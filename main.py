
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
    metrics.router,
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
