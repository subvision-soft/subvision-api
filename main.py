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


SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_HOURS = 3600

# Le cache à l'ancienne :)
news_cache = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokens = {}


class ImageRequest(BaseModel):
    image_data: str

@app.get("/generate-token")
def generate_token():
    expiration = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
    token = jwt.encode({
        "exp": expiration}, SECRET_KEY, algorithm=ALGORITHM)
    tokens[token] = expiration
    return {"token": token, "expiration": expiration}


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


@app.post("/detect-target")
def detect_target(
        request: ImageRequest,
        authorization: str = Depends(validate_token)
):
    try:
        decoded_data = base64.b64decode(request.image_data)
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format.")
        return get_sheet_coordinates(image)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")



def extract_news(notion_json):
    extracted_data = []
    for item in notion_json.get("results", []):
        properties = item.get("properties", {})
        title = properties.get("Titre", {}).get("title", [])
        title_text = title[0]["plain_text"] if title else ""
        date = properties.get("Date", {}).get("date", {}).get("start", "")
        date_obj = datetime.strptime(date, "%Y-%m-%d").date() if date else None
        illustration_files = properties.get("Illustration", {}).get("files", [])
        illustration = illustration_files[0]["file"]["url"] if illustration_files else ""
        description_rich_text = properties.get("Description", {}).get("rich_text", [])
        description_html = "".join([
            f"<b>{t['text']['content']}</b>" if t["annotations"]["bold"] else
            f"<i>{t['text']['content']}</i>" if t["annotations"]["italic"] else
            t["text"]["content"]
            for t in description_rich_text
        ]).replace("\n", "<br>")

        extracted_data.append({
            "title": title_text,
            "date": date_obj,
            "illustration": illustration,
            "description": description_html
        })

    return extracted_data

def fetch_news():
    client = Client(auth=notion_token)
    db_rows = client.databases.query(database_id=notion_database_id)
    notion_data = json.loads(json.dumps(db_rows))  # Replace with actual JSON
    parsed_data = extract_news(notion_data)
    return {
        "timestamp": datetime.now(),
        "data": parsed_data
    }

@app.get("/news")
def get_news():
    global news_cache
    if news_cache is None or datetime.now() - news_cache["timestamp"] > timedelta(minutes=5):
        news_cache = fetch_news()
    return news_cache["data"]


def safe_get(data, dot_chained_keys):
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
        decoded_data = base64.b64decode(request.image_data)
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image format.")
        return process_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
    )
