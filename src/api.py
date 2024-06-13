from fastapi import FastAPI
from router.semantic_router import SemanticRouter


app = FastAPI()
semantic_router = SemanticRouter()

@app.post(
    "/chat",
    response_model=str,
    response_model_exclude_none=True,
    status_code=200,
)
def text(request_body: dict):
    return semantic_router.handler(request_body["content"])