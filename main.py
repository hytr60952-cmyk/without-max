import asyncio 
import time
import json
import httpx
import random
import io
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

# ✅ Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# ================= Config =================
CACHE = {}
CACHE_TTL = 240  # 4 minutes

TELEGRAM_BOT_TOKEN = "7652042264:AAGc6DQ-OkJ8PaBKJnc_NkcCseIwmfbHD-c"
TELEGRAM_CHAT_ID = "5029478739"

# ✅ Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("instagram-scraper")

# ✅ Stats & Session Management
STATS = {
    "last_alerts": [],
    "total_requests": 0,
    "session_refreshes": 0
}

# ================= Global HTTPX Client (Session Reuse + Cookies) =================
GLOBAL_CLIENT: httpx.AsyncClient | None = None
REQUEST_COUNT = 0  # Global request counter for session refresh

# Tune limits for connection reuse
HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)

# A realistic header templates pool — more fields to mimic real browsers
BASE_HEADER_TEMPLATES = [
    {
        "x-ig-app-id": "936619743392459",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    },
    {
        "x-ig-app-id": "936619743392459",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/605.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    },
    {
        "x-ig-app-id": "936619743392459",
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image.webp,image/apng,*/*;q=0.8",
        "Accept-Language": "hi-IN,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "Origin": "https://www.instagram.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
    },
]

def build_request_headers():
    """
    Returns a realistic header dict built from templates with small randomizations
    (like Accept-Language variants or small referer tweaks) to appear less bot-like.
    """
    base = random.choice(BASE_HEADER_TEMPLATES).copy()
    # Small variance: randomize Accept-Language quality or add mild query referer
    if random.random() < 0.2:
        base["Accept-Language"] = base.get("Accept-Language", "en-US,en;q=0.9") + ",fr;q=0.7"
    # Occasionally add a faux "x-ig-www-claim" or other IG-ish header
    if random.random() < 0.15:
        base["x-ig-www-claim"] = "0"
    # Add a mild timestamp-like suffix to Referer sometimes to mimic navigation
    if random.random() < 0.1:
        base["Referer"] = base["Referer"] + f"?ts={int(time.time() % 10000)}"
    return base

# ================= NEW: Professional Session Management =================
async def refresh_instagram_session():
    """
    Regularly refresh Instagram session to avoid detection
    Creates new client with fresh cookies and visits homepage
    """
    global GLOBAL_CLIENT, REQUEST_COUNT
    
    logger.info("🔄 Refreshing Instagram session...")
    
    # Close old client if exists
    if GLOBAL_CLIENT:
        try:
            await GLOBAL_CLIENT.aclose()
        except Exception as e:
            logger.warning(f"Error closing old client: {e}")
    
    # Create new client with fresh cookies
    GLOBAL_CLIENT = httpx.AsyncClient(
        timeout=15.0,
        limits=HTTPX_LIMITS,
        follow_redirects=True,
        trust_env=False
    )
    
    # Visit Instagram homepage to get fresh cookies
    try:
        homepage_headers = build_request_headers()
        homepage_headers["Sec-Fetch-Dest"] = "document"
        homepage_headers["Sec-Fetch-Mode"] = "navigate"
        
        await GLOBAL_CLIENT.get("https://www.instagram.com/", headers=homepage_headers)
        logger.info("✅ New Instagram session created with fresh cookies")
        
        # Update stats
        STATS["session_refreshes"] += 1
        
    except Exception as e:
        logger.warning(f"⚠️ Session refresh - homepage visit failed: {e}")
        # Client still created, just without initial cookies

# ================= NEW: Professional Human-like Delays =================
async def human_like_delay(request_type="scrape"):
    """
    Implement realistic human browsing patterns with random delays
    Different delay strategies for different types of requests
    """
    if request_type == "scrape":
        # Main scraping delays - most realistic pattern
        delay_strategies = [
            random.uniform(1.2, 3.5),    # Normal browsing: 1.2-3.5s
            random.uniform(0.8, 2.2),    # Quick consecutive: 0.8-2.2s  
            random.uniform(2.5, 4.5),    # Longer break: 2.5-4.5s
            random.uniform(0.5, 1.5)     # Rare fast: 0.5-1.5s
        ]
    elif request_type == "image":
        # Image requests can be faster
        delay_strategies = [
            random.uniform(0.7, 2.0),
            random.uniform(0.4, 1.2),
            random.uniform(1.5, 3.0)
        ]
    else:
        # Default
        delay_strategies = [random.uniform(1.0, 3.0)]
    
    delay = random.choice(delay_strategies)
    
    # Add micro-variations (±10%)
    delay_variation = delay * random.uniform(-0.1, 0.1)
    final_delay = max(0.3, delay + delay_variation)  # Minimum 0.3s
    
    await asyncio.sleep(final_delay)
    return final_delay

# ================= Utils =================
def format_error_message(username: str, attempt: int, error: str, status_code: int = None):
    base = f"❌ ERROR | User: {username}\n🔁 Attempt: {attempt}"
    if status_code:
        return f"{base}\n📡 Status: {status_code} ({error})"
    else:
        return f"{base}\n⚠️ Exception: {error}"

async def cache_cleaner():
    while True:
        now = time.time()
        expired_keys = [k for k, v in CACHE.items() if v["expiry"] < now]
        for k in expired_keys:
            CACHE.pop(k, None)
        await asyncio.sleep(60)

async def notify_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, data=payload)
        # store in stats
        STATS["last_alerts"].append({"time": time.time(), "msg": message})
        STATS["last_alerts"] = STATS["last_alerts"][-10:]  # keep last 10
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")

async def handle_error(status_code: int, detail: str, notify_msg: str = None):
    if notify_msg:
        await notify_telegram(notify_msg)
    raise HTTPException(status_code=status_code, detail=detail)

# ================= Lifespan: create global client & start cache cleaner =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_CLIENT
    # Initial session creation
    await refresh_instagram_session()

    # Start cache cleaner
    cleaner_task = asyncio.create_task(cache_cleaner())
    try:
        yield
    finally:
        cleaner_task.cancel()
        if GLOBAL_CLIENT:
            await GLOBAL_CLIENT.aclose()
        GLOBAL_CLIENT = None

# ================= App Init =================
app = FastAPI(lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ================= Helper for getting client =================
def get_client() -> httpx.AsyncClient:
    if GLOBAL_CLIENT is None:
        # Fallback: create a short-lived client if lifespan hasn't run
        return httpx.AsyncClient(timeout=15.0, limits=HTTPX_LIMITS, follow_redirects=True, trust_env=False)
    return GLOBAL_CLIENT

# ================= API Logic =================
async def scrape_user(username: str, max_retries: int = 3):
    global REQUEST_COUNT
    
    username = username.lower()
    cached = CACHE.get(username)
    if cached and cached["expiry"] > time.time():
        return cached["data"]

    url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
    client = get_client()

    # ✅ NEW: Human-like delay before requests
    delay_used = await human_like_delay("scrape")
    logger.debug(f"⏳ Human delay: {delay_used:.2f}s before scraping {username}")

    # ✅ NEW: Session refresh every 25 requests
    REQUEST_COUNT += 1
    STATS["total_requests"] += 1
    
    if REQUEST_COUNT % 25 == 0:
        logger.info(f"🔄 Auto-refreshing session after {REQUEST_COUNT} requests")
        await refresh_instagram_session()
        # Small delay after session refresh
        await asyncio.sleep(random.uniform(2, 4))

    for attempt in range(1, max_retries + 1):
        headers = build_request_headers()
        try:
            resp = await client.get(url, headers=headers)
            status = resp.status_code

            # If successful json
            if status == 200:
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    # Malformed JSON — maybe a block; treat as retryable
                    msg = format_error_message(username, attempt, "Invalid JSON", status)
                    logger.warning(msg)
                    await notify_telegram(msg)
                    # human-like delay before retry
                    await human_like_delay("scrape")
                    continue

                user = data.get("data", {}).get("user")
                if not user:
                    await handle_error(404, "User not found", f"⚠️ User not found: {username}")

                user_data = {
                    "username": user.get("username"),
                    "real_name": user.get("full_name"),
                    "profile_pic": user.get("profile_pic_url_hd"),
                    "followers": user.get("edge_followed_by", {}).get("count"),
                    "following": user.get("edge_follow", {}).get("count"),
                    "post_count": user.get("edge_owner_to_timeline_media", {}).get("count"),
                    "bio": user.get("biography"),
                }

                CACHE[username] = {"data": user_data, "expiry": time.time() + CACHE_TTL}
                logger.info(f"✅ Successfully scraped {username} (Attempt {attempt})")
                return user_data

            # If rate-limited or temporarily blocked
            elif status in (429, 403, 401):
                msg = format_error_message(username, attempt, "Rate-limited/Forbidden", status)
                logger.warning(msg)
                await notify_telegram(msg)

                # Exponential backoff with jitter + human-like pattern
                backoff = (2 ** (attempt - 1)) * random.uniform(1.0, 2.5)
                # For 403 maybe longer wait before retrying
                if status == 403:
                    backoff *= 1.5
                logger.info(f"⏳ Backoff delay: {backoff:.2f}s for {username}")
                await asyncio.sleep(backoff)
                continue

            elif status == 404:
                await handle_error(404, "User not found", f"⚠️ User not found: {username}")

            else:
                msg = format_error_message(username, attempt, "Request Failed", status)
                logger.warning(msg)
                await notify_telegram(msg)
                # human-like delay before retry
                await human_like_delay("scrape")
                continue

        except httpx.RequestError as e:
            msg = format_error_message(username, attempt, str(e))
            logger.warning(msg)
            await notify_telegram(msg)
            # backoff on network errors too
            backoff = (2 ** (attempt - 1)) * random.uniform(0.5, 1.5)
            await asyncio.sleep(backoff)

    await handle_error(502, "All attempts failed", f"🚨 All attempts failed for {username}")

# ================= Routes =================
@app.get("/scrape/{username}")
@limiter.limit("8/minute")  # Reduced for safety
async def get_user(username: str, request: Request):
    return await scrape_user(username)

@app.get("/proxy-image/")
@limiter.limit("10/minute")
async def proxy_image(request: Request, url: str, max_retries: int = 3):
    client = get_client()
    
    # ✅ NEW: Human-like delay for image requests
    delay_used = await human_like_delay("image")
    logger.debug(f"⏳ Image delay: {delay_used:.2f}s")

    for attempt in range(1, max_retries + 1):
        headers = build_request_headers()
        try:
            resp = await client.get(url, headers=headers, timeout=15.0)
            if resp.status_code == 200:
                # determine content-type safely
                ctype = resp.headers.get("content-type", "image/jpeg")
                logger.info(f"✅ Image proxied successfully: {url[:50]}...")
                return StreamingResponse(io.BytesIO(resp.content), media_type=ctype)
            elif resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Image not found")
            elif resp.status_code in (429, 403):
                msg = format_error_message("proxy-image", attempt, "Image fetch blocked", resp.status_code)
                logger.warning(msg)
                await notify_telegram(msg)
                await asyncio.sleep((2 ** (attempt - 1)) * random.uniform(0.5, 1.8))
                continue
            else:
                msg = format_error_message("proxy-image", attempt, "Image fetch failed", resp.status_code)
                logger.warning(msg)
                await notify_telegram(msg)
                await human_like_delay("image")
                continue

        except httpx.RequestError as e:
            msg = format_error_message("proxy-image", attempt, str(e))
            logger.warning(msg)
            await notify_telegram(msg)
            await asyncio.sleep((2 ** (attempt - 1)) * random.uniform(0.5, 1.5))

    raise HTTPException(status_code=502, detail="All attempts failed for image fetch")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(), 
        "total_requests": STATS["total_requests"],
        "session_refreshes": STATS["session_refreshes"],
        "cache_size": len(CACHE)
    }

@app.head("/health")
async def health_check_head():
    return JSONResponse(content=None, status_code=200)

@app.get("/stats")
async def stats():
    # Provide cookie summary for debugging
    client = get_client()
    cookie_count = len(client.cookies.jar) if hasattr(client.cookies, "jar") else len(client.cookies)
    return {
        "cache_size": len(CACHE),
        "total_requests": STATS["total_requests"],
        "session_refreshes": STATS["session_refreshes"],
        "last_alerts": STATS["last_alerts"],
        "cookie_count": cookie_count,
        "request_count": REQUEST_COUNT
    }

@app.post("/refresh-session")
async def manual_refresh_session():
    """Manual endpoint to refresh Instagram session"""
    await refresh_instagram_session()
    return {"status": "session_refreshed", "session_refreshes": STATS["session_refreshes"]}
