
import uvicorn
from fastapi import FastAPI,  Depends
from fastapi.middleware.cors import CORSMiddleware

from routers.dependencies import validate_token
from routers.private import target
from routers.public import token, news, metrics

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

app.include_router(
    token.router,
    prefix="/token",
    tags=["token"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    target.router,
    prefix="/target",
    tags=["target"],
    dependencies=[Depends(validate_token)],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    news.router,
    prefix="/news",
    tags=["news"],
    responses={404: {"description": "Not found"}})

app.include_router(
    metric.router,
    prefix="/metrics",
        tags=["metrics"],
    responses={404: {"description": "Not found"}},
)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
    )
