from datetime import datetime, timedelta

from fastapi import APIRouter
from jose import jwt

from routers.dependencies import SECRET_KEY, ALGORITHM, TOKENS

TOKEN_EXPIRATION_HOURS = 3600

router = APIRouter()

@router.get("/")
def generate_token():
    expiration = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
    token = jwt.encode({
        "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    TOKENS[token] = expiration
    return {"token": token, "expiration": expiration}
