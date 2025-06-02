import asyncio
import uvicorn
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from clap_agent_executor import ClapAgentA2AExecutorFastAPIStyle

A2A_SERVER_HOST = "localhost"
A2A_SERVER_PORT = 9999

class A2AAgentRequest(BaseModel): 
    message: str = Field(..., description="The message/query for the agent")
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = Field(None)

class A2AAgentResponse(BaseModel): 
    message: str = Field(..., description="The agent's response")
    status: str = Field(default="success")
    data: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = Field(None)

clap_executor_instance: Optional[ClapAgentA2AExecutorFastAPIStyle] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clap_executor_instance
    print("CLAP A2A FastAPI Server (RAG): Lifespan startup...")
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=dotenv_path) 

    clap_executor_instance = ClapAgentA2AExecutorFastAPIStyle()
    await clap_executor_instance.setup_rag_agent() 
    print("CLAP A2A FastAPI Server (RAG): ClapAgentA2AExecutorFastAPIStyle initialized and RAG setup.")
    yield
    print("CLAP A2A FastAPI Server (RAG): Lifespan shutdown...")
    if clap_executor_instance and hasattr(clap_executor_instance, 'close_resources'):
        await clap_executor_instance.close_resources()
    print("CLAP A2A FastAPI Server (RAG): Resources closed.")

app = FastAPI(
    title="CLAP RAG Agent (A2A FastAPI Style)",
    description="Exposes a CLAP RAG agent via A2A using FastAPI.",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"],
)

AGENT_CARD_DATA = {
    "name": "CLAP Finance Info Agent (A2A FastAPI)",
    "description": "A CLAP agent with RAG capabilities, providing information about Finance based on a specific knowledge base.",
    "url": f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}/",
    "version": "0.2.0", # Updated version
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"], 
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "clap_query_holbox_ai_info", 
            "name": "Query Holbox AI Information (CLAP RAG Agent)",
            "description": "Answers questions based on the content of a pre-loaded Machine Learning textbook.",
            "tags": ["clap", "rag", "machine-learning", "document-analysis"],
            "examples": [
                "What are the main products of Holbox AI?",
                "Tell me about Holbox AI's mission."
            ],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"]
        }
    ]
}

@app.get("/.well-known/agent.json", response_model=Dict[str, Any])
async def get_agent_card():
    return AGENT_CARD_DATA

@app.post("/", response_model=A2AAgentResponse)
async def handle_a2a_message_send(request: A2AAgentRequest = Body(...)):
    global clap_executor_instance
    if not clap_executor_instance or not clap_executor_instance._initialized: 
        raise HTTPException(status_code=503, detail="RAG Agent executor not ready or failed initialization.")
    
    print(f"CLAP A2A FastAPI Server (RAG): Received query: '{request.message[:100]}...'")
    try:
        response_content_str = await clap_executor_instance.execute(
            user_input=request.message,
        )
        return A2AAgentResponse(message=response_content_str, status="success", session_id=request.session_id)
    except Exception as e:
        print(f"CLAP A2A FastAPI Server (RAG): Error processing RAG request: {e}")
        return A2AAgentResponse(
            message=f"Error processing your RAG request: {str(e)}", status="error",
            data={"error_type": type(e).__name__}, session_id=request.session_id
        )

if __name__ == "__main__":
    print(f"Starting CLAP RAG A2A Server (FastAPI Style) on http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}")
    uvicorn.run(app, host=A2A_SERVER_HOST, port=A2A_SERVER_PORT, log_level="info")
