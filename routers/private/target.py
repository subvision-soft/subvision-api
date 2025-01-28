import base64
import traceback

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from target_detection import get_sheet_coordinates, process_image

router = APIRouter()
class ImageRequest(BaseModel):
    image_data: str
@router.post("/detect")
def detect_target(
        request: ImageRequest,
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


@router.post("/process")
def target_score(
        request: ImageRequest,
):
    try:
        decoded_data = base64.b64decode(request.image_data)
        np_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        image = cv2.flip(image, 0)

        if image is None:
            raise ValueError("Invalid image format.")
        return process_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
