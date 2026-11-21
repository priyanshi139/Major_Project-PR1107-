"""Authentication helpers: PBKDF2 password hashing, signed JWT access tokens, login throttling."""
import hashlib, hmac, os, secrets, time
from pathlib import Path
import jwt

ITERATIONS = 200_000
TOKEN_HOURS = 8
ALGO = "HS256"


def hash_password(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return f"pbkdf2${ITERATIONS}${salt}${h}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it)).hex()
        return hmac.compare_digest(calc, h)
    except Exception:
        return False


def load_secret(artifacts_dir) -> str:
    env = os.environ.get("HEALTHTWIN_SECRET")
    if env:
        return env
    p = Path(artifacts_dir) / ".secret_key"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    s = secrets.token_hex(32)
    p.write_text(s, encoding="utf-8")
    return s


def make_token(secret: str, user: dict) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user["user_id"]), "role": user["role"], "iat": now, "exp": now + TOKEN_HOURS * 3600}, secret, algorithm=ALGO)


def read_token(secret: str, token: str) -> dict:
    return jwt.decode(token, secret, algorithms=[ALGO])


class Throttle:
    """Blocks a username after `limit` failed logins within `window` seconds (in-memory)."""

    def __init__(self, limit=5, window=300):
        self.limit, self.window, self.fails = limit, window, {}

    def blocked(self, name) -> bool:
        now = time.time()
        self.fails[name] = [t for t in self.fails.get(name, []) if now - t < self.window]
        return len(self.fails[name]) >= self.limit

    def fail(self, name):
        self.fails.setdefault(name, []).append(time.time())

    def ok(self, name):
        self.fails.pop(name, None)
