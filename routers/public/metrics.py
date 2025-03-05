from fastapi import FastAPI, Request
import time
from collections import defaultdict, deque
from statistics import mean

app = FastAPI()

MAX_RECORDS = 100
response_times = defaultdict(lambda: deque(maxlen=MAX_RECORDS))

@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()

    endpoint = request.url.path
    duration = end_time - start_time

    response_times[endpoint].append(duration)

    return response

@app.get("/metrics")
async def get_average_response_times():
    averages = {endpoint: mean(times) for endpoint, times in response_times.items() if times}
    return {"average_response_times": averages}
