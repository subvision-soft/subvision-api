from fastapi import FastAPI, HTTPException, Header, Depends
from jose import jwt, JWTError
from datetime import datetime, timedelta
import secrets
import base64
import numpy as np
import cv2
import io

# Constants
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 30

app = FastAPI()

# Temporary token storage (for demo purposes)
tokens = {}

# Generate a temporary token
@app.post("/generate-token")
def generate_token():
    expiration = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    token = jwt.encode({"exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    tokens[token] = expiration
    return {"token": token, "expires_in": TOKEN_EXPIRATION_MINUTES}

# Validate a token
def validate_token(authorization: str = Header(...)):
    try:
        # Extract the token from the "Authorization" header
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid token format")
        token = authorization[len("Bearer "):]
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if token not in tokens or tokens[token] < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token expired or invalid")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoint to process the image
@app.post("/process-image")
def process_image(
    image_data: bytes,
    authorization: str = Depends(validate_token)
):
    try:
        # Convert raw bytes into an image (e.g., if it's base64 encoded)

        decoded_data = base64.b64decode(image_data)
        # Convert bytes data to a NumPy array
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)

        # Decode the NumPy array into an image using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        # Perform your image processing here


        return {"message": "Image processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
