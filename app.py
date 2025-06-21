import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import certifi
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from slack_sdk import WebClient
from datetime import datetime
import re
from dotenv import load_dotenv
import sqlite3
from sqlite3 import Error
from google.cloud import secretmanager
import json
import threading
import time
from datetime import datetime, timedelta
from slack_sdk.errors import SlackApiError
from google.api_core.exceptions import ResourceExhausted
from contextlib import contextmanager

os.environ['SSL_CERT_FILE'] = certifi.where()

# Get Google Cloud Project ID from environment
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

def access_secret_version(secret_id, version_id="latest", project_id=None):
    """
    Access the payload of the given secret version.
    """
    if project_id is None:
        project_id = GOOGLE_CLOUD_PROJECT
        if not project_id:
            raise ValueError("Google Cloud Project ID not set. Please set GOOGLE_CLOUD_PROJECT environment variable.")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    try:
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        return payload
    except Exception as e:
        logging.error(f"Failed to access secret {secret_id}: {e}", exc_info=True)
        # Depending on the secret, you might want to raise the error or return None
        # For critical secrets like tokens, raising an error might be better to halt execution.
        raise  # Re-raise the exception to make it clear that secret fetching failed.

# Load secrets from Secret Manager
# For multi-tenancy, these would be dynamically determined, perhaps using a workspace_id.
# For now, we use a placeholder or default. In a real multi-tenant app,
# you'd fetch SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET based on the incoming request's team_id.
# GEMINI_API_KEY might be global or per-tenant. SLACK_APP_TOKEN is usually global for Socket Mode.

# It's good practice to define secret IDs as constants
SLACK_SIGNING_SECRET_ID = "SLACK_SIGNING_SECRET"
GEMINI_API_KEY_ID = "GEMINI_API_KEY"
SLACK_BOT_TOKEN_ID = "SLACK_BOT_TOKEN"
SLACK_APP_TOKEN_ID = "SLACK_APP_TOKEN" # Typically used for Socket Mode connection

try:
    # Attempt to load GOOGLE_CLOUD_PROJECT first if not already set (e.g. local testing without .env for this var)
    if not GOOGLE_CLOUD_PROJECT:
        load_dotenv() # Load .env for local development if GOOGLE_CLOUD_PROJECT is not set
        GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") # Try loading from .env

    # If GOOGLE_CLOUD_PROJECT is still not set, it means we are likely in an environment
    # where it should have been provided (like Cloud Run) or local dev is misconfigured.
    if not GOOGLE_CLOUD_PROJECT:
        logging.warning("GOOGLE_CLOUD_PROJECT environment variable not found. Secret fetching will likely fail if not running locally with application default credentials.")
        # For local development where ADC is set up, project_id might be inferred by the client library.
        # However, explicitly setting it is safer.
        # If you still want to try to load other secrets from .env for local dev:
        load_dotenv() # Ensure .env is loaded for local fallback
        SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
        SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
        if not all([SLACK_SIGNING_SECRET, GEMINI_API_KEY, SLACK_BOT_TOKEN, SLACK_APP_TOKEN]):
            logging.error("Failed to load secrets from .env file as a fallback during local development.")
            # Decide on behavior: raise error, or try to continue (might fail later)
            # For now, let it try to continue, it will fail when Slack clients are initialized if tokens are None.
    else:
        # Productive path: Load secrets from Google Cloud Secret Manager
        SLACK_SIGNING_SECRET = access_secret_version(SLACK_SIGNING_SECRET_ID)
        GEMINI_API_KEY = access_secret_version(GEMINI_API_KEY_ID)
        SLACK_BOT_TOKEN = access_secret_version(SLACK_BOT_TOKEN_ID)
        SLACK_APP_TOKEN = access_secret_version(SLACK_APP_TOKEN_ID) # For Socket Mode

except Exception as e:
    logging.error(f"Critical error during secret initialization: {e}", exc_info=True)
    # Fallback for local development if Secret Manager access fails (e.g. no auth, wrong project)
    # This helps in local testing without full cloud setup, but ensure GOOGLE_CLOUD_PROJECT is NOT set
    # or this block won't be hit as intended if GOOGLE_CLOUD_PROJECT is set but SM access fails.
    if not GOOGLE_CLOUD_PROJECT: # Only try .env if GOOGLE_CLOUD_PROJECT was never set (true local dev)
        logging.info("Attempting to load secrets from .env file for local development due to Secret Manager access failure.")
        load_dotenv()
        SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
        SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
        if not all([SLACK_SIGNING_SECRET, GEMINI_API_KEY, SLACK_BOT_TOKEN, SLACK_APP_TOKEN]):
            logging.error("Failed to load secrets from .env file as a fallback. Application may not work.")
            # Consider raising an exception here if secrets are absolutely critical for startup
    else:
        # If GOOGLE_CLOUD_PROJECT was set, it implies a cloud environment or intentional local test with SM.
        # Failure here is more critical.
        raise RuntimeError(f"Failed to initialize secrets from Secret Manager and GOOGLE_CLOUD_PROJECT was set. Error: {e}")


# Global variables
SEARCH_LIMIT = 3000 # Max messages to search
CHUNK_SIZE = 500 # Messages per Gemini call
DELAY_BETWEEN_CALLS = 20 # Seconds to wait between Gemini calls
MAX_CHUNKS = 6 # Hard cap to avoid runaway loops
BACKFILL_INTERVAL = 1800 # 30 minutes
SEARCH_TRIGGERS = ["search", "find", "look up", "look for", "show me"]
ENABLE_MENTION_INJECTION = True

# Initialize Slack clients
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

# Cache the bot's user ID once
try:
    BOT_USER_ID = client.auth_test()['user_id']
except Exception as e:
    logging.error(f"Failed to fetch bot user ID: {e}")
    BOT_USER_ID = None

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)

logger = logging.getLogger("slack_bot")

# Suppress excessive logging from external libraries
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

genai.configure(api_key=GEMINI_API_KEY)

def get_model():
    try:
        # Use the latest Gemini 2.0 Flash model
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logger.error("Failed to load gemini-2.0-flash, falling back to gemini-1.5-flash", exc_info=True)
        # Fallback to 1.5-flash only if still available, but this will be deprecated soon
        return genai.GenerativeModel('gemini-1.5-flash')

model = get_model()

DATABASE_FILE = "slack_messages.db"

@contextmanager
def db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database with messages table"""
    with db_connection() as conn:
        c = conn.cursor()
        
        # Enable Write-Ahead Logging (WAL) and Optimize PRAGMAs
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
       
        # Main messages table
        c.execute('''CREATE TABLE IF NOT EXISTS messages
        (ts TEXT PRIMARY KEY,
        text TEXT,
        user_id TEXT,
        username TEXT,
        channel_id TEXT,
        channel_name TEXT,
        thread_ts TEXT,
        reactions TEXT,
        attachments TEXT,
        permalink TEXT,
        is_bot BOOLEAN,
        deleted BOOLEAN DEFAULT FALSE)''')
        
        # Last seen timestamps per channel
        c.execute('''CREATE TABLE IF NOT EXISTS last_seen
        (channel_id TEXT PRIMARY KEY,
        last_ts TEXT)''')
        
        # Whether messages have been responded to
        c.execute('''CREATE TABLE IF NOT EXISTS responded_messages
        (ts TEXT PRIMARY KEY)''')
        conn.commit()

init_db()

def get_slack_entity(entity_type, entity_id, cache, fetch_fn):
    if entity_id in cache:
        return cache[entity_id]
    try:
        if not entity_id:
            return f"Unknown {entity_type.title()}"
        entity_info = fetch_fn(entity_id)
        name = entity_info.get('name') or entity_info.get('real_name') or entity_id
        cache[entity_id] = name
        return name
    except Exception:
        return f"Unknown {entity_type.title()}"

user_name_cache = {}
channel_name_cache = {}

def get_username(user_id):
    return get_slack_entity(
        "user", user_id, user_name_cache,
        lambda uid: client.users_info(user=uid)['user']
    )

def get_channel_name(channel_id):
    return get_slack_entity(
        "channel", channel_id, channel_name_cache,
        lambda cid: client.conversations_info(channel=cid)['channel']
    )

def build_mention_map():
    mention_map = {}
    users = client.users_list()["members"]
    for user in users:
        profile = user.get("profile", {})
        real_name = profile.get("real_name", "").strip()
        display_name = profile.get("display_name", "").strip()

        if not real_name:
            continue

        first_name = real_name.split()[0]

        # Add both full name and first name
        mention_map[real_name] = user["id"]
        mention_map[first_name] = user["id"]

        # Optional: add display name if different
        if display_name and display_name != first_name and display_name != real_name:
            mention_map[display_name] = user["id"]

    return mention_map

def assemble_message_data(msg, channel_info):
    """Build a dict with all message fields for DB upsert/insert."""
    ts = msg.get('ts')
    channel_id = channel_info['id']
    
    # # Fetch extended channel details
    # try:
    #     channel_details = client.conversations_info(channel=channel_id)["channel"]
    #     # is_dm = channel_details.get("is_im", False)
    #     # is_private = channel_details.get("is_private", False)
    # except SlackApiError as e:
    #     logger.error(f"[CHANNEL] Error fetching channel info: {e}")
    #     # is_dm = False
    #     # is_private = False

    return {
        'ts': ts,
        'text': msg.get('text'),
        'user_id': msg.get('user'),
        'username': get_username(msg.get('user')) if msg.get('user') else 'Unknown User',
        'channel_id': channel_id,
        'channel_name': channel_info.get('name') or channel_info.get('user') or 'unknown',
        'thread_ts': msg.get('thread_ts'),
        'reactions': json.dumps(msg.get('reactions', [])),
        'attachments': json.dumps(msg.get('attachments', [])),
        'permalink': safe_get_permalink(client, channel_id, ts),
        'is_bot': int('bot_id' in msg)
    }

def build_sources_list(messages):
    """
    Given a list of message dicts, build a formatted sources list for Slack.
    """
    if not messages:
        return ""
    lines = ["*Sources:*"]
    seen = set()
    for msg in messages:
        key = (msg['ts'], msg['channel_id'])
        if key in seen:
            continue
        seen.add(key)
        # Build link if possible
        ts = msg['ts']
        channel = msg.get('channel_name', 'unknown')
        author = get_username(msg['user_id']) if msg.get('user_id') else "Unknown"
        text_preview = (msg.get('text', '') or '')[:80].replace('\n', ' ')
        try:
            dt = datetime.fromtimestamp(float(ts))
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            timestamp_str = ts
        permalink = msg.get('permalink')
        if permalink:
            link = f"<{permalink}|link>"
        else:
            link = "(unavailable)"
        deleted = msg.get('deleted', 0)
        deleted_note = " _(deleted)_" if deleted else ""
        lines.append(f"- [{timestamp_str}, @{author}, #{channel}]{deleted_note}: \"{text_preview}\" {link}")
    return "\n".join(lines)

def filter_search_messages(messages):
    """
    Return only messages that are not deleted and not sent by bots.
    Handles Slack API format and common DB fields.
    """
    filtered = []
    for msg in messages:
        # Ignore deleted messages
        if msg.get('subtype') == 'message_deleted':
            continue
        if msg.get('deleted', False):
            continue
        if msg.get('is_deleted', False):
            continue
        # Ignore messages sent by bots
        if msg.get('subtype') == 'bot_message' or msg.get('bot_id') or (msg.get('message', {}).get('bot_id')):
            continue
        filtered.append(msg)
    return filtered

def save_or_update_message(msg, channel_info, action="upsert", source="event"):
    """
    Insert, update, or mark a message as deleted in the database.

    Args:
        msg (dict): Slack message dictionary (must include 'ts').
        channel_info (dict): Channel info dictionary (must include 'id').
        action (str): "upsert" for insert/update, "delete" for soft-delete.
        source (str): Event source, for logging/audit.
    """
    ts = msg.get('ts')
    channel_id = channel_info['id']
    
    if action == "delete":
        logger.info(f"[DEBUG] Attempting to mark message ts={ts} as deleted in channel={channel_id}.")
        with db_connection() as conn:
            c = conn.cursor()
            c.execute('UPDATE messages SET deleted = 1 WHERE ts = ? AND channel_id = ?', (ts, channel_id))
            conn.commit()
        logger.info(f"[DB] Marked message ts={ts} in channel {channel_id} as deleted.")
        return

    # Upsert logic
    text = msg.get('text')
    if text is None or text == "":
        logger.info(f"[DB] Skipping upsert for message ts={ts} in channel {channel_id} due to empty or None text.")
        return

    msg_data = assemble_message_data(msg, channel_info)
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM messages WHERE ts = ? AND channel_id = ?", (ts, channel_id))
        exists = c.fetchone() is not None
        if exists:
            c.execute('''
                UPDATE messages SET
                    text=:text,
                    user_id=:user_id,
                    username=:username,
                    channel_name=:channel_name,
                    thread_ts=:thread_ts,
                    reactions=:reactions,
                    attachments=:attachments,
                    permalink=:permalink,
                    is_bot=:is_bot,
                    deleted=0
                WHERE ts = :ts AND channel_id = :channel_id
            ''', msg_data)
        else:
            logger.info(f"[DEBUG] Attempting to upsert message ts={ts} in channel={channel_id}.")
            c.execute('''
                INSERT INTO messages (
                    ts, text, user_id, username, channel_id, channel_name,
                    thread_ts, reactions, attachments, permalink, is_bot, deleted
                ) VALUES (
                    :ts, :text, :user_id, :username, :channel_id, :channel_name,
                    :thread_ts, :reactions, :attachments, :permalink, :is_bot, 0
                )
            ''', msg_data)
        c.execute('INSERT OR REPLACE INTO last_seen (channel_id, last_ts) VALUES (?, ?)', (channel_id, ts))
        logger.info("[DB] Successfully added message to log.")
        conn.commit()

def get_thread_and_context_messages(channel_id, thread_ts=None, limit=CHUNK_SIZE):
    if thread_ts:
        return client.conversations_replies(channel=channel_id, ts=thread_ts).get('messages', [])
    else:
        return list(reversed(client.conversations_history(channel=channel_id, limit=limit).get('messages', [])))

def safe_get_permalink(client, channel_id, ts):
    try:
        response = client.chat_getPermalink(channel=channel_id, message_ts=ts)
        if response.get('ok'):
            return response.get('permalink')
        else:
            logger.warning(f"Failed to get permalink for ts={ts} in channel={channel_id}: {response.get('error')}")
            return None
    except SlackApiError as e:
        if hasattr(e, "response") and e.response.get('error') == 'message_not_found':
            logger.warning(f"Message not found for ts={ts} in channel={channel_id}. It may have been deleted.")
            return None
        else:
            logger.error(f"Slack API error when getting permalink: {e}", exc_info=True)
            return None
    except Exception as e:
        if 'message_not_found' in str(e):
            logger.warning(f"Message not found for ts={ts} in channel={channel_id}. It may have been deleted.")
            return None
        else:
            logger.error(f"Slack API error when getting permalink: {e}", exc_info=True)
            return None

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def log_interaction(user_id, channel_id, message):
    user_name = get_username(user_id)
    channel_name = get_channel_name(channel_id)
    logger.info(f"[INTERACTION] User:{user_name} | Channel:{channel_name} | Message:'{message}'")

def log_mention(user_id, channel_id, message):
    user_name = get_username(user_id)
    channel_name = get_channel_name(channel_id)
    logger.info(f"[MENTION] From user {user_name} in channel {channel_name}: {message}")

def has_responded(ts):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT 1 FROM responded_messages WHERE ts = ?', (ts,))
        result = c.fetchone()
    return result is not None

def mark_as_responded(ts):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT OR IGNORE INTO responded_messages (ts) VALUES (?)', (ts,))
        conn.commit()

def inject_mentions(text):
    # Replace longer names first to avoid partial name collisions
    for name in sorted(MENTION_MAP, key=len, reverse=True):
        user_id = MENTION_MAP[name]
        pattern = r'\b' + re.escape(name) + r'\b'
        replacement = f'<@{user_id}>'
        text = re.sub(pattern, replacement, text)
    return text

def is_search_request(message):
    return any(trigger in message.lower() for trigger in SEARCH_TRIGGERS) if message else False

def extract_search_terms_and_instruction(message, logger=None):
    extraction_prompt = f"""Extract from this message:
1. Keywords to search for (comma-separated)
2. Instruction for presenting results (after ";")

Example response for "Find docs about AI and summarize key points":
"AI, documentation; summarize key points"

Message: "{message}"
Response:"""
    try:
        response = generate_response(extraction_prompt, safe=True).strip()
        if ";" in response:
            terms_part, instruction = response.split(";", 1)
            return [t.strip() for t in terms_part.split(",")], instruction.strip()
        else:
            return [response], "Summarize the findings"
    except Exception as e:
        if logger:
            logger.error(f"Search term extraction failed: {e}")
        return [message], "Summarize the findings"

def is_message_too_old(ts, hours=24):
    try:
        msg_time = datetime.fromtimestamp(float(ts))
        return datetime.now() - msg_time > timedelta(hours=hours)
    except Exception:
        return True  # If parsing fails, treat as too old

def store_message_with_check(msg, channel_info, source="event"):
    channel_id = channel_info['id']
    text = msg.get('text')
    ts = msg.get('ts')

    # Fetch full channel info to detect DMs and private channels
    try:
        channel_details = client.conversations_info(channel=channel_id)["channel"]
        is_dm = channel_details.get("is_im", False)
        is_private = channel_details.get("is_private", False)
    except SlackApiError as e:
        logger.error(f"[CHANNEL] Error fetching channel info: {e}")
        # Default to storing if detection fails
        is_dm = False
        is_private = False

    # Skip DMs and private channels
    if is_dm or is_private:
        logger.info(f"[DB] Skipping message in {'DM' if is_dm else 'private channel'} {channel_id}")
        return

    # Existing logic for handling deletions/upserts
    if text is None and source == "deleted_event":
        logger.info(f"[DB] Marking message ts={ts} in channel={channel_id} as deleted.")
        save_or_update_message(msg, channel_info, action="delete", source=source)
        return

    logger.info(f"[DB] Upserting message ts={ts} in channel={channel_id} (source={source}).")
    save_or_update_message(msg, channel_info, action="upsert", source=source)

def search_messages_local(terms, channel_id=None):
    """Search local database for messages containing any of the terms (OR logic)"""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            query = '''
            SELECT * FROM messages
            WHERE deleted = 0
            AND ({})
            '''.format(' OR '.join(['text LIKE ?'] * len(terms)))
            params = [f'%{term}%' for term in terms]
            if channel_id:
                query += ' AND channel_id = ?'
                params.append(channel_id)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        return results
    except Error as e:
        logger.error(f"[DB] Search error: {e}", exc_info=True)
        return []

def get_last_seen(channel_id):
    """Get last processed timestamp for a channel"""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT last_ts FROM last_seen WHERE channel_id = ?', (channel_id,))
            result = c.fetchone()
            last_ts = result['last_ts'] if result else '0'
        return last_ts
    except Error as e:
        logger.error(f"[DB] Error getting last seen: {e}", exc_info=True)
        return '0'

def message_exists(ts, channel_id):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM messages WHERE ts=? AND channel_id=?", (ts, channel_id))
        return c.fetchone() is not None

def periodic_backfill(interval=BACKFILL_INTERVAL):
    """Periodic backfill scheduler with auto-vacuum and null entry cleanup"""
    while True:
        try:
            backfill_and_process_mentions()
            
            # Database maintenance: VACUUM and remove null/empty entries
            with db_connection() as conn:
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                conn.execute("PRAGMA incremental_vacuum")
                logger.info("[BACKFILL] Automatic VACUUM complete")
                conn.execute("DELETE FROM messages WHERE text IS NULL OR text = ''")
                conn.commit()
                logger.info("[BACKFILL] Removed NULL/empty text entries")

        except Exception as e:
            logger.error(f"[BACKFILL] Periodic backfill error: {e}", exc_info=True)
        
        time.sleep(interval)

def backfill_and_process_mentions():
    """Backfill missed messages and process mentions (with pagination), skipping already logged messages."""
    try:
        logger.info("[BACKFILL] Starting backfill process")
        channels = client.conversations_list(types="public_channel")["channels"]

        for channel in channels:
            channel_id = channel['id']
            if not channel.get('is_member', False):
                continue

            last_ts = get_last_seen(channel_id)
            logger.debug(f"[BACKFILL] Backfilling {channel['name']} from {last_ts}")

            cursor = None
            while True:
                response = client.conversations_history(
                    channel=channel_id,
                    oldest=last_ts,
                    limit=1000,
                    cursor=cursor
                )

                for msg in response['messages']:
                    ts = msg.get('ts')
                    if not ts:
                        continue

                    # Skip bot join/system messages
                    if "has joined the channel" in msg.get('text', '').lower():
                        continue

                    # Skip if message is too old
                    if is_message_too_old(ts):
                        continue

                    # Skip if already responded
                    if has_responded(ts):
                        continue

                    # Skip if already logged in DB
                    with db_connection() as conn:
                        c = conn.cursor()
                        c.execute("SELECT 1 FROM messages WHERE ts=? AND channel_id=?", (ts, channel_id))
                        if c.fetchone():
                            logger.info(f"[BACKFILL] Skipping already-logged message ts={ts} in channel={channel_id}")
                            continue

                    # Store message
                    store_message_with_check(msg, channel)

                    # Check for mentions
                    if f"<@{BOT_USER_ID}>" in msg.get('text', ''):
                        log_mention(msg.get('user'), channel_id, msg.get('text'))
                        handle_app_mention({
                            'user': msg.get('user'),
                            'channel': channel_id,
                            'text': msg.get('text'),
                            'ts': ts,
                            'thread_ts': msg.get('thread_ts')
                        }, lambda text, **kwargs: client.chat_postMessage(
                            channel=channel_id,
                            text=text,
                            thread_ts=msg.get('thread_ts') if msg.get('thread_ts') else None
                        ), client, logger
                    )
                    mark_as_responded(ts)

                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break

        logger.info("[BACKFILL] Backfill complete")

    except Exception as e:
        logger.error(f"[BACKFILL] Critical error: {e}", exc_info=True)

@app.event("message")
def handle_all_messages(event, say, logger):
    # Ignore bot messages and channel join messages to avoid loops and noise
    if event.get("subtype") in ("bot_message", "channel_join"):
        return

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text", "")
    logger.info(f"[DEBUG] Received event: {event}")

    if "has joined the channel" in text.lower():
        return

    # Store message in database
    try:
        channel_info = client.conversations_info(channel=channel_id)["channel"]
        store_message_with_check(event, channel_info)
    except Exception as e:
        logger.error(f"[DB] Error storing message: {e}", exc_info=True)

    # Handle message deletions
    if event.get("subtype") == "message_deleted":
        channel_id = event.get("channel")
        # Use deleted_ts, not ts
        deleted_ts = event.get("deleted_ts")
        msg = {'ts': deleted_ts}
        channel_info = {'id': channel_id}
        save_or_update_message(msg, channel_info, action="delete")
        logger.info(f"[DB] Marked message ts={deleted_ts} as deleted in channel {channel_id}.")
        return
    
    # Handle message edits
    if event.get("subtype") == "message_changed":
        channel_id = event.get("channel")
        edited_msg = event["message"]
        # Fetch channel info as before
        channel_info = client.conversations_info(channel=channel_id)["channel"]
        # Store the edited message, using the original ts
        store_message_with_check(edited_msg, channel_info)
        return

    # Handle Direct Messages (DMs)
    if channel_id and channel_id.startswith("D"):
        history = client.conversations_history(channel=channel_id, limit=CHUNK_SIZE)
        messages = list(reversed(history.get("messages", [])))
        if messages:
            primary_message = messages[-1].get("text", "")
            context_messages = [
                f"{get_username(msg.get('user', ''))}: {msg.get('text', '')}"
                for msg in messages[:-1] if msg.get('user') and msg.get('text')
            ]
            context = {
                "primary_message": primary_message,
                "previous_messages": "\n".join(context_messages)
            }
            prompt = build_context_prompt(context)
            response_text = generate_response(prompt)
            formatted = format_for_slack(response_text or "")
            say(formatted, mrkdwn=True)
        return  # Prevent further processing for DMs

@app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    ts = event.get("ts")
    user = event.get("user")
    subtype = event.get("subtype")
    bot_id = event.get("bot_id")

    # Ignore bot messages and already processed messages
    if subtype == "bot_message" or bot_id is not None:
        return

    if has_responded(ts):
        logger.info(f"Skipping already processed mention: {ts}")
        return

    mark_as_responded(ts)

    if event.get("subtype") == "channel_join":
        logger.info("[MENTION] Ignored app_mention event with subtype channel_join.")
        return

    original_text = event.get('text', '')
    channel = event.get('channel')
    user = event.get('user')

    log_mention(user, channel, original_text)

    try:
        if BOT_USER_ID:
            cleaned_text = original_text.replace(f"<@{BOT_USER_ID}>", '').strip()
            event['text'] = cleaned_text
            logger.debug(f"[MENTION] Cleaned text passed to interaction handler: '{cleaned_text}'")
        else:
            logger.warning("[MENTION] BOT_USER_ID is not set — skipping mention cleanup.")

        # Pass client and logger to handle_bot_interaction
        handle_bot_interaction(event, say, client, logger)
    except Exception as e:
        logger.error(f"[MENTION] Error processing mention: {e}", exc_info=True)

def format_source_for_thread(msg, idx=None):
    """
    Format a source message for posting as a threaded reply.
    """
    try:
        ts_float = float(msg['ts'])
        dt = datetime.fromtimestamp(ts_float)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        timestamp_str = msg['ts']

    author = get_username(msg['user_id']) if msg.get('user_id') else "Unknown"
    channel = msg.get('channel_name', 'unknown')
    text_preview = (msg.get('text', '') or '')[:200].replace('\n', ' ')
    permalink = msg.get('permalink')
    deleted = msg.get('deleted', 0)
    deleted_note = " _(deleted)_" if deleted else ""

    link = f"<{permalink}|link>" if permalink else "(unavailable)"
    prefix = f"{idx}. " if idx else ""

    return (f"{prefix}[{timestamp_str}, @{author}, #{channel}]{deleted_note}: "
            f"\"{text_preview}\" {link}")


# Add this constant near the top of your app.py (with other constants)
MAX_SOURCES = 10  # Adjust this number as needed

def handle_search_request(keywords, instruction, user_id, channel_id, say, client, logger, source="command"):
    logger.info(f"[SEARCH][{source.upper()}] Handling request - Keywords: {keywords} | Instruction: {instruction}")

    raw_results = search_messages_across_channels(keywords, logger=logger, search_limit=SEARCH_LIMIT)
    results = filter_search_messages(raw_results)

    if not results:
        say(f"No messages found for: {', '.join(keywords)}", channel=channel_id)
        return

    summaries = []
    sources = results

    try:
        for i, chunk in enumerate(chunk_list(results, CHUNK_SIZE)):
            if i >= MAX_CHUNKS:
                summaries.append("Too many results. Please narrow your search.")
                break
            logger.info(f"[SEARCH][CHUNK] Summarizing chunk {i+1}/{MAX_CHUNKS} with {len(chunk)} messages.")
            context_snippets = [format_message_for_context(r) for r in chunk]
            prompt = (
                f"You are Clip Eastwood, a Slack-based creative assistant. "
                f"Below are Slack messages about {', '.join(keywords)}. "
                f"{instruction}\n\n"
                f"{'-'*20}\n"
                f"{chr(10).join(context_snippets)}"
            )
            summary = generate_response(prompt, safe=True)
            summaries.append(summary)
            time.sleep(DELAY_BETWEEN_CALLS)
    except ResourceExhausted:
        say("Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later.", channel=channel_id)
        return

    if len(summaries) > 1:
        logger.info(f"[SEARCH][FINAL] Summarizing {len(summaries)} chunk summaries into final result.")
        final_prompt = (
            f"Summarize the following summaries into a single cohesive answer:\n\n"
            f"{'-'*20}\n"
            f"{chr(10).join(summaries)}"
        )
        try:
            final_summary = generate_response(final_prompt, safe=True)
        except ResourceExhausted:
            say("Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later.", channel=channel_id)
            return
    else:
        final_summary = summaries[0] if summaries else "No summary available."

    # --- Limit sources to MAX_SOURCES and prepare note ---
    MAX_SOURCES = 10
    sources_to_show = sources[:MAX_SOURCES]
    sources_note = ""
    if len(sources) > MAX_SOURCES:
        sources_note = f"_{len(sources)} sources found. Showing the {MAX_SOURCES} most recent in thread._"
    elif sources:
        sources_note = f"_{len(sources)} sources found._"
    else:
        sources_note = "_No relevant sources found._"

    # Post summary with sources note
    context_header = (
        f"*Requestor:* <@{user_id}>\n"
        f"*Keywords:* {', '.join(keywords)}\n"
        f"*Context:* {instruction}\n\n"
        f"{sources_note}\n\n"
    )
    summary_post = client.chat_postMessage(
        channel=channel_id,
        text=context_header + "\n" + format_for_slack(final_summary or "")
    )
    parent_ts = summary_post["ts"]

    # Post sources (if any)
    if sources_to_show:
        BATCH_SIZE = 5
        for batch_start in range(0, len(sources_to_show), BATCH_SIZE):
            batch = sources_to_show[batch_start:batch_start + BATCH_SIZE]
            batch_lines = []
            for idx, msg in enumerate(batch, start=batch_start + 1):
                batch_lines.append(format_source_for_thread(msg, idx))
            batch_text = "\n".join(batch_lines)
            client.chat_postMessage(
                channel=channel_id,
                text=batch_text,
                thread_ts=parent_ts
            )
    else:
        client.chat_postMessage(
            channel=channel_id,
            text="No relevant sources found.",
            thread_ts=parent_ts
        )

@app.command("/ai-search")
def handle_search_command(ack, respond, command, say, logger, client):
    ack()
    text = command.get("text", "").strip()
    user_id = command.get("user_id")
    channel_id = command.get("channel_id")

    if ";" in text:
        terms_part, instruction = text.split(";", 1)
        instruction = instruction.strip()
    else:
        terms_part = text
        instruction = "Summarize the findings"

    keywords = [t.strip() for t in terms_part.split(",") if t.strip()]

    # Immediate feedback to user
    respond(f"Searching for {', '.join(keywords)} across channels and fulfilling your request to '{instruction}'. This may take a few minutes...")

    handle_search_request(
        keywords=keywords,
        instruction=instruction,
        user_id=user_id,
        channel_id=channel_id,
        say=say,
        client=client,
        logger=logger,
        source="command"
    )

def search_messages_local_flat(limit=None, channel_id=None):
    """Fetch all (or recent) messages from local DB, with optional limit."""
    try:
        with db_connection() as conn:
            c = conn.cursor()
            query = "SELECT * FROM messages WHERE deleted = 0"
            params = []
            if channel_id:
                query += " AND channel_id = ?"
                params.append(channel_id)
            query += " ORDER BY ts DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        return results
    except Error as e:
        logger.error(f"[DB] Search error: {e}", exc_info=True)
        return []

def search_messages_across_channels(search_terms, logger=None, search_limit=None):
    """
    Search for messages containing any of the search terms (OR logic).
    Returns a list of relevant messages with full metadata.
    """
    try:
        results = []
        with db_connection() as conn:
            c = conn.cursor()
            query = ("SELECT * FROM messages "
                     "WHERE deleted = 0 AND (" +
                     " OR ".join(["text LIKE ?" for _ in search_terms]) +
                     ") ORDER BY ts DESC")
            params = [f'%{term}%' for term in search_terms]
            if search_limit:
                query += ' LIMIT ?'
                params.append(search_limit)
            c.execute(query, params)
            rows = c.fetchall()
            results = [dict(row) for row in rows]
        # Remove duplicates (by ts + channel_id)
        seen = set()
        unique_results = []
        for r in results:
            key = (r["channel_id"], r["ts"])
            if key not in seen:
                unique_results.append(r)
                seen.add(key)
        return unique_results
    except Exception as e:
        if logger:
            logger.error(f"[SEARCH] Error searching messages: {e}", exc_info=True)
        return []

def handle_bot_interaction(event, say, client, logger):
    """Handles @mentions and DMs"""
    start_time = time.time()
    user_id = event.get('user')
    channel_id = event.get('channel')
    message_text = event.get('text', '')
    primary_message = event.get('text')
    thread_ts = event.get('thread_ts')

    # Ignore bot messages
    if event.get('bot_id') or event.get('subtype') == 'bot_message':
        return

    # Route search requests
    if is_search_request(message_text):
        keywords, instruction = extract_search_terms_and_instruction(message_text, logger)
        handle_search_request(
            keywords=keywords,
            instruction=instruction,
            user_id=user_id,
            channel_id=channel_id,
            say=say,
            client=client,
            logger=logger,
            source="mention"
        )
        return

    log_interaction(user_id, channel_id, primary_message)
    context_messages = []

    try:
        messages = get_thread_and_context_messages(channel_id, thread_ts, limit=CHUNK_SIZE)
        messages = filter_search_messages(messages)
        logger.debug(f"[CONTEXT] Fetched {len(messages)} messages for context (thread_ts={thread_ts})")

        for msg in messages:
            if 'text' in msg and msg.get('user'):
                try:
                    sender = get_username(msg['user'])
                    timestamp = datetime.fromtimestamp(float(msg['ts'])).isoformat()
                    context_messages.append(f"[{timestamp}] {sender}: {msg['text']}")
                except Exception as e:
                    logger.warning(f"[CONTEXT] Skipped message due to user info error: {e}")

        context = {
            "primary_message": primary_message,
            "previous_messages": "\n".join(context_messages[-CHUNK_SIZE:])
        }

        prompt = build_context_prompt(context)
        response_text = generate_response(prompt)
        logger.info(f"[RESPONSE] AI generated response for user {user_id}")
        formatted = format_for_slack(response_text or "")
        say(formatted, thread_ts=thread_ts if thread_ts else None, mrkdwn=True)

    except Exception as e:
        logger.error(f"[CONTEXT] Error fetching context or generating response: {e}", exc_info=True)
        say(f"Hey there, <@{user_id}>! I ran into a problem gathering context.")

    duration = time.time() - start_time
    logger.info(f"[PERF] Interaction completed in {duration:.2f}s for user {user_id}")


def get_thread_and_context_messages(channel_id, thread_ts=None, limit=CHUNK_SIZE):
    """Fetch messages with pagination, handling both threads and channel history"""
    messages = []
    cursor = None
    
    try:
        if thread_ts:
            # Threaded conversation
            while True:
                response = client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    cursor=cursor,
                    limit=min(limit, 200)  # Slack's max per page
                )
                messages.extend(response.get('messages', []))
                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor or len(messages) >= limit:
                    break
        else:
            # Channel history
            response = client.conversations_history(
                channel=channel_id,
                limit=limit,
                oldest=datetime.now().timestamp() - 86400  # Last 24h
            )
            messages = list(reversed(response.get('messages', [])))
            
        return messages[:limit]  # Hard cap
    except SlackApiError as e:
        logger.error(f"Error fetching messages: {e}", exc_info=True)
        return []

def build_context_prompt(context):
    """Prompt for context-rich user interactions (direct mentions, DMs)."""
    return f"""You are Clip Eastwood, a Slack-based creative assistant with a gruff, witty, and cowboy flair.
Your top priority is to follow user instructions quickly and accurately, using clarity and insight.
Be helpful, insightful, and capable of creative tasks and effective business communications (lists, memos, letters, summaries, newsletters, blogs, articles, and multi-paragraph write-ups and summaries).
Respond in markdown, and maintain your signature style only in communicating with users, not in content output.

Instructions:
- Use the recent messages to understand any names, tasks, or references.
- Do NOT repeat or summarize the context block—just use it to inform your reply.
- Avoid repeating questions the user already answered.
- Be warm and human, but don't delay the task.
- Use wit and charm only to enhance clarity or delight, not to stall.

User request:
\"\"\"{context['primary_message']}\"\"\"

Channel context:
\"\"\"{context['previous_messages']}\"\"\"

Respond clearly, confidently, and in a format that fits the request (lists, memos, letters, summaries, newsletters, blogs, articles, and multi-paragraph write-ups and summaries).
Prioritize completion.
"""

def build_search_prompt(terms, context_snippets, context_instruction):
    """Prompt for search/summarization features."""
    return (
        f"You are Clip Eastwood, a Slack-based creative assistant. "
        f"Below are Slack messages about {', '.join(terms)}. "
        f"{context_instruction}\n\n"
        f"{'-'*20}\n"
        f"{chr(10).join(context_snippets)}"
    )

def generate_response(prompt, safe=True, max_retries=2):
    """
    Calls Gemini (or other LLM) with a prompt and returns a formatted Slack response.
    """
    for attempt in range(max_retries if safe else 1):
        try:
            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            text = getattr(response, "text", None)
            if not text:
                return "I couldn't generate a response."
            return format_for_slack(text)
        except ResourceExhausted as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) * 15
                logger.warning(f"API quota exceeded, retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Gemini API quota exceeded: {e}", exc_info=True)
                return "Sorry, the AI service is temporarily unavailable due to quota limits. Please try again later."
        except Exception as e:
            logger.error(f"AI generation failed: {e}", exc_info=True)
            return "Sorry, I couldn't generate a response due to an internal error."

def generate_content(prompt):
    response = model.generate_content(
        prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    if not response or not hasattr(response, 'text') or not response.text:
        logging.error("Response blocked or invalid.")
        if hasattr(response, 'safety_ratings'):
            logging.info(f"Safety ratings: {response.safety_ratings}")
        return "Sorry, I couldn't generate a response."
    
    generated_text = response.text.strip('\"')  # Remove leading and trailing quotation marks
    logging.debug(f"Generated text: {generated_text}")
    return generated_text

def format_for_slack(text, do_inject_mentions=True):
    """End-to-end formatting pipeline for Gemini -> Slack"""
    # Cleanup
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip('"\'')

    # Markdown conversion
    text = re.sub(r'^#+\s(.+)$', r'*\1*', text, flags=re.MULTILINE)  # Headers
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)  # Bold
    text = re.sub(r'`([^`]+)`', r'`\1`', text)  # Inline code
    
    # Link formatting
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<\2|\1>', text)

    # Mention injection
    if do_inject_mentions and ENABLE_MENTION_INJECTION:
        text = inject_mentions(text)

    return text.strip()

MENTION_MAP = build_mention_map()

def format_message_for_context(msg):
    """
    Format a message for use in context or as a source reference.
    Includes timestamp, author, channel, and permalink.
    """
    # Format timestamp
    try:
        ts_float = float(msg['ts'])
        dt = datetime.fromtimestamp(ts_float)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        timestamp_str = msg['ts']

    author = get_username(msg['user_id']) if msg.get('user_id') else "Unknown"
    channel = msg.get('channel_name', 'unknown')
    text = msg.get('text', '')

    # Add reactions
    reactions = ""
    try:
        rx = json.loads(msg['reactions']) if msg['reactions'] else []
        if rx:
            reactions = " [Reactions: " + " ".join([f":{r['name']}|{r['count']}" for r in rx]) + "]"
    except Exception:
        pass

    # Add attachments
    attachments = ""
    try:
        att = json.loads(msg['attachments']) if msg['attachments'] else []
        file_links = []
        for a in att:
            if 'title' in a and 'title_link' in a:
                file_links.append(f"{a['title']}: {a['title_link']}")
            elif 'name' in a and 'url_private' in a:
                file_links.append(f"{a['name']}: {a['url_private']}")
        if file_links:
            attachments = f" [Attachments: {'; '.join(file_links)}]"
    except Exception:
        pass

    # Permalink (if available and not deleted)
    permalink = msg.get('permalink')
    deleted = msg.get('deleted', 0)
    deleted_note = " _(deleted)_" if deleted else ""

    # Compose
    return f"[{timestamp_str}, @{author}, #{channel}]{deleted_note}: {text}{reactions}{attachments}"

def join_all_channels():
    channels = client.conversations_list(types="public_channel")["channels"]
    for channel in channels:
        channel_id = channel["id"]
        try:
            client.conversations_join(channel=channel_id)
        except Exception as e:
            # Ignore if already in channel or can't join
            pass

def remove_null_text_entries():
    """Remove any messages with NULL text from the database."""
    with db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE text IS NULL;")
        conn.commit()
    logger.info("[BACKFILL] Removed all messages with NULL text from the database.")

# Start your app
if __name__ == "__main__":
    join_all_channels()
    
    # Start periodic backfill in background FIRST
    threading.Thread(target=periodic_backfill, daemon=True).start()
    
    # THEN start your Slack app (this blocks)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()