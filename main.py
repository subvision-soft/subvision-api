import base64
import secrets
import traceback
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends, Response
from fastapi.params import Cookie
from jose import jwt, JWTError
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from target_detection import get_sheet_coordinates, process_image

# Constants
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = 3600

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins like ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Or restrict methods like ["GET", "POST"]
    allow_headers=["*"],  # Or restrict headers like ["Content-Type", "Authorization"]
)

# Temporary token storage (for demo purposes)
tokens = {}


class ImageRequest(BaseModel):
    image_data: str


# Generate a temporary token
@app.get("/generate-token")
def generate_token():
    expiration = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
    token = jwt.encode({
        "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    tokens[token] = expiration
    return {"token": token, "expiration": expiration}


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
    except JWTError as e:
        print(e)
        raise HTTPException(status_code=401, detail="Invalid token")


# Endpoint to process the image
@app.post("/detect-target")
def detect_target(
        request: ImageRequest,
        authorization: str = Depends(validate_token)
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
        # benchmarking
        start = datetime.now()
        coordinates = get_sheet_coordinates(image)
        print(f"Time taken: {datetime.now() - start}")
        return coordinates
    except Exception as e:
        # trace the error
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/target-score")
def target_score(
        request: ImageRequest,
        authorization: str = Depends(validate_token)
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
        return process_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
