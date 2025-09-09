from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional

app = FastAPI(title="MCP Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContextRequest(BaseModel):
    context: str
    model_name: Optional[str] = None

class PredictionRequest(BaseModel):
    input_data: str
    context: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "MCP Server is running"}

@app.post("/load_context")
async def load_context(request: ContextRequest):
    # Load model context logic here
    return {"status": "Context loaded", "model": request.model_name or "default"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Model prediction logic here
    return {"prediction": "Sample prediction based on context"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)