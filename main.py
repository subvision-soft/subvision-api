import base64
import secrets
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends, Response
from fastapi.params import Cookie
from jose import jwt, JWTError
from pydantic import BaseModel

from target_detection import get_sheet_coordinates

# Constants
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = 3600

app = FastAPI()

# Temporary token storage (for demo purposes)
tokens = {}
class ImageRequest(BaseModel):
    image_data: str

# Generate a temporary token
@app.get("/generate-token")
def generate_token(response: Response):
    expiration = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
    token = jwt.encode({
        "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    tokens[token] = expiration
    # use Set-Cookie header to store the token in the browser
    response.set_cookie(key="token", value=token, expires=TOKEN_EXPIRATION_HOURS)
    return None


# Validate a token
def validate_token(token: str = Cookie(...)):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if token not in tokens or tokens[token] < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token expired or invalid")
    except JWTError as e:
        print(e)
        raise HTTPException(status_code=401, detail="Invalid token")


# Endpoint to process the image
@app.post("/detect-target")
def detect_target(
        request: ImageRequest,
        token: str = Depends(validate_token)
):
    try:
        # Decode the base64 image string
        decoded_data = base64.b64decode(request.image_data)
        # Convert bytes to NumPy array
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)
        # Decode the image using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        # save the image to disk
        cv2.imwrite("image.jpg", image)
        if image is None:
            raise ValueError("Invalid image format.")
        # Perform your image processing here
        return get_sheet_coordinates(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
