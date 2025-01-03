import base64
import json
import os
import secrets
import traceback
from datetime import datetime, timedelta

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends, Response
from jose import jwt, JWTError
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from target_detection import get_sheet_coordinates, process_image

from notion_client import Client

notion_token = os.environ.get('NOTION_TOKEN')

notion_database_id = os.environ.get('NOTION_DATABASE_ID')

import socket

# Constants
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = 3600

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/news")
def get_news():
    # Initialize the Notion client
    client = Client(auth=notion_token)

    # Get the database
    db_rows = client.databases.query(database_id=notion_database_id)
    write_dict_to_file_as_json(db_rows, 'news.json')
    results = []
    for row in db_rows['results']:
        title = safe_get(row, 'properties.Titre.title.0.text.content')
        string_date = safe_get(row, 'properties.Date.date.start')
        date = datetime.strptime(string_date, '%Y-%m-%d')
        description = safe_get(row, 'properties.Description.rich_text.0.text.content')
        illustration = safe_get(row, 'properties.Illustration.files.0.file.url')
        results.append({
            'title': title,
            'date': date,
            'description': description,
            'illustration': illustration
        })

    return results


def write_dict_to_file_as_json(content, file_name):
    content_as_json_str = json.dumps(content)

    with open(file_name, 'w') as f:
        f.write(content_as_json_str)


def safe_get(data, dot_chained_keys):
    '''
        {'a': {'b': [{'c': 1}]}}
        safe_get(data, 'a.b.0.c') -> 1
    '''
    keys = dot_chained_keys.split('.')
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[int(key)]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None
    return data


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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
    )
