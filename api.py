"""FastAPI backend exposing the ReAct agent response predictor."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import react_agent
import retriever

app = FastAPI(title="Agent Response Predictor", version="1.0")


class Message(BaseModel):
    role: str = Field(..., description="'customer' or 'agent'")
    text: str


class PredictRequest(BaseModel):
    conversation_history: list[Message] = Field(default_factory=list)
    customer_message: str


class Source(BaseModel):
    thread_id: str | None = None
    subject: str | None = None
    score: float | None = None


class PredictResponse(BaseModel):
    predicted_response: str
    confidence: str
    reasoning_trace: list[dict]
    sources: list[dict]


@app.get("/health")
def health():
    try:
        s = retriever.stats()
        return {"status": "ok", "milvus": s}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {e}")


@app.get("/stats")
def stats():
    return retriever.stats()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    history = [{"role": m.role, "text": m.text} for m in req.conversation_history]
    try:
        result = react_agent.predict(history, req.customer_message)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return result


if __name__ == "__main__":
    import uvicorn

    from config import get_config

    cfg = get_config()
    uvicorn.run(app, host=cfg.API_HOST, port=cfg.API_PORT)
