import secrets

from fastapi import Header, HTTPException
from jose import jwt, JWTError

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
from datetime import datetime, timedelta
TOKENS = {}

def validate_token(authorization: str = Header("Authorization")):
    try:
        # Extract the token from the "Authorization" header
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid token format")
        token = authorization[len("Bearer "):]
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if token not in TOKENS or TOKENS[token] < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token expired or invalid")
    except JWTError as e:
        print(e)
        raise HTTPException(status_code=401, detail="Invalid token")