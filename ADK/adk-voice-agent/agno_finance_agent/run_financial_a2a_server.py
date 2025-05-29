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

from financial_agno_executor import AgnoFinancialA2AExecutorFastAPI

A2A_SERVER_HOST = "localhost"
A2A_SERVER_PORT = 10000 

class AgnoAgentApiRequest(BaseModel):
    message: str = Field(..., description="The user query or context for the financial brief")

class AgnoAgentApiResponse(BaseModel):
    message: str 
    status: str 
    data: Optional[Dict[str, Any]] = None 

agno_executor_instance: Optional[AgnoFinancialA2AExecutorFastAPI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agno_executor_instance
    print("Agno Financial Agent FastAPI Server: Lifespan startup...")
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=dotenv_path) 
    print(f"Agno Financial Agent FastAPI Server: .env loaded from {dotenv_path}")

    agno_executor_instance = AgnoFinancialA2AExecutorFastAPI()
    print("Agno Financial Agent FastAPI Server: AgnoFinancialA2AExecutorFastAPI initialized.")
    yield
    print("Agno Financial Agent FastAPI Server: Lifespan shutdown...")
    if agno_executor_instance and hasattr(agno_executor_instance, 'close_resources'):
        await agno_executor_instance.close_resources()
    print("Agno Financial Agent FastAPI Server: Resources closed.")

app = FastAPI(
    title="Agno Financial Agent (A2A-like via FastAPI)",
    description="Exposes an Agno-powered financial agent using a direct FastAPI endpoint.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

AGENT_CARD_DATA = {
    "name": "Agno Financial Report Agent (FastAPI)",
    "description": "Generates morning market briefs using CoinGecko, YFinance, and an LLM.",
    "url": f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}/", 
    "version": "1.0.0",
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"], 
    "capabilities": {"streaming": False}, 
    "skills": [{
            "id": "generate_financial_brief",
            "name": "Generate Financial Market Brief",
            "description": "Creates a daily financial brief using market data and news.",
            "tags": ["finance", "market-report", "crypto", "stocks", "agno"],
            "examples": ["What's the morning market brief?", "Get financial report for Asia tech."],
            "inputModes": ["text/plain"],
            "outputModes": ["text/plain"]
    }]
}

@app.get("/.well-known/agent.json", response_model=Dict[str, Any])
async def get_agent_card_endpoint(): 
    return AGENT_CARD_DATA

@app.post("/", response_model=AgnoAgentApiResponse) 
async def generate_financial_brief_endpoint(request: AgnoAgentApiRequest = Body(...)):
    global agno_executor_instance
    if not agno_executor_instance:
        raise HTTPException(status_code=503, detail="Financial Agent executor not initialized.")
    
    print(f"Agno Financial Agent FastAPI Server: Received POST to / : '{request.message[:100]}...'")
    try:
        report_text = await agno_executor_instance.generate_brief_directly(
            user_input=request.message
        )
        if "Error" in report_text[:20]: 
            return AgnoAgentApiResponse(message=report_text, status="error")
        return AgnoAgentApiResponse(message=report_text, status="success")
    except Exception as e:
        print(f"Agno Financial Agent FastAPI Server: Error processing request: {e}")
        return AgnoAgentApiResponse(
            message=f"Internal server error: {str(e)}",
            status="error",
            data={"error_type": type(e).__name__}
        )

if __name__ == "__main__":
    print(f"Starting Agno Financial Agent Server (FastAPI Style) on http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}")
    uvicorn.run(app, host=A2A_SERVER_HOST, port=A2A_SERVER_PORT, log_level="info")