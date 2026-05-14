#src/auth.py

import os
import time
import json
import hmac
import base64
import hashlib
from typing import Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException
from pydantic import BaseModel

load_dotenv()


ADMIN_USERNAME = os.getenv("STROKE_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("STROKE_ADMIN_PASSWORD", "admin123")

#Token signing secret.
AUTH_SECRET = os.getenv(
    "STROKE_AUTH_SECRET",
    "stroke-ai-triage-secret-local",
)

TOKEN_EXPIRE_SECONDS = int(os.getenv("STROKE_TOKEN_EXPIRE_SECONDS", "86400"))  #24hours


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64d(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str) -> str:
    sig = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64e(sig)


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "role": "admin",
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
    }

    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64e(payload_json)
    sig_b64 = _sign(payload_b64)

    return f"{payload_b64}.{sig_b64}"


def verify_token(token: str) -> dict:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token format")

    expected_sig = _sign(payload_b64)

    if not hmac.compare_digest(sig_b64, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    try:
        payload = json.loads(_b64d(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")

    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return payload


def require_admin(authorization: Optional[str] = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.replace("Bearer ", "", 1).strip()
    payload = verify_token(token)

    return payload


def login_admin(req: AdminLoginRequest):
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin username or password",
        )

    token = create_token(req.username)

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": "admin",
        "expires_in": TOKEN_EXPIRE_SECONDS,
    }