from __future__ import annotations
import os
import requests
from typing import Dict, Any

def sign_in_email_password(email: str, password: str) -> Dict[str, Any]:
    api_key = os.environ.get("FIREBASE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FIREBASE_API_KEY not set. Run in Demo mode or set API key.")

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Auth failed: {r.text}")
    return r.json()
