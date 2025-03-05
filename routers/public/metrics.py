from fastapi import APIRouter, Request
import time
from collections import defaultdict, deque
from statistics import mean

router = APIRouter()

@router.get("/")
async def get_average_response_times():
    averages = {endpoint: mean(times) for endpoint, times in response_times.items() if times}
    return {"average_response_times": averages}
