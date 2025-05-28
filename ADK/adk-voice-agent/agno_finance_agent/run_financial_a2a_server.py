# HOLBOXATHON/agno_financial_agent/run_financial_agent_server_fastapi.py
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

# Import your new executor
from financial_agno_executor import AgnoFinancialA2AExecutorFastAPI

# --- Configuration ---
A2A_SERVER_HOST = "localhost"
A2A_SERVER_PORT = 10000 # Port for this Agno Financial Agent server

# --- Request/Response Pydantic Models for the FastAPI endpoint ---
class AgnoAgentApiRequest(BaseModel):
    message: str = Field(..., description="The user query or context for the financial brief")
    # Add other fields if your ADK client tool will send them

class AgnoAgentApiResponse(BaseModel):
    message: str # The generated financial brief or an error message
    status: str # "success" or "error"
    data: Optional[Dict[str, Any]] = None # For any extra data or error details

# --- Lifespan Management ---
agno_executor_instance: Optional[AgnoFinancialA2AExecutorFastAPI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agno_executor_instance
    print("Agno Financial Agent FastAPI Server: Lifespan startup...")
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=dotenv_path) # For GROQ_API_KEY etc.
    print(f"Agno Financial Agent FastAPI Server: .env loaded from {dotenv_path}")

    agno_executor_instance = AgnoFinancialA2AExecutorFastAPI()
    # No async setup for the executor itself, its __init__ handles Agno agent setup
    print("Agno Financial Agent FastAPI Server: AgnoFinancialA2AExecutorFastAPI initialized.")
    yield
    print("Agno Financial Agent FastAPI Server: Lifespan shutdown...")
    if agno_executor_instance and hasattr(agno_executor_instance, 'close_resources'):
        await agno_executor_instance.close_resources()
    print("Agno Financial Agent FastAPI Server: Resources closed.")

# --- FastAPI App Creation ---
app = FastAPI(
    title="Agno Financial Agent (A2A-like via FastAPI)",
    description="Exposes an Agno-powered financial agent using a direct FastAPI endpoint.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"], # ADK Voice Agent origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Agent Card (still useful for discovery if ADK tool fetches it) ---
AGENT_CARD_DATA = {
    "name": "Agno Financial Report Agent (FastAPI)",
    "description": "Generates morning market briefs using CoinGecko, YFinance, and an LLM.",
    "url": f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}/", # Points to the POST endpoint
    "version": "1.0.0",
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"], # Will return the report as text
    "capabilities": {"streaming": False}, # This direct FastAPI is not streaming A2A events
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
async def get_agent_card_endpoint(): # Renamed to avoid conflict
    return AGENT_CARD_DATA

# --- Main API Endpoint ---
@app.post("/", response_model=AgnoAgentApiResponse) # POST to the root
async def generate_financial_brief_endpoint(request: AgnoAgentApiRequest = Body(...)):
    global agno_executor_instance
    if not agno_executor_instance:
        raise HTTPException(status_code=503, detail="Financial Agent executor not initialized.")
    
    print(f"Agno Financial Agent FastAPI Server: Received POST to / : '{request.message[:100]}...'")
    try:
        report_text = await agno_executor_instance.generate_brief_directly(
            user_input=request.message
        )
        if "Error" in report_text[:20]: # Basic check if the agent returned an error string
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