import httpx
import json
import os

CLAP_A2A_SERVER_BASE_URL = os.getenv("CLAP_A2A_SERVER_URL", "http://localhost:9999")

async def call_clap_agent_via_a2a(user_query: str) -> str:
    print(f"\n[ADK A2A TOOL LOG (FastAPI Style)] Calling CLAP A2A agent.")
    print(f"[ADK A2A TOOL LOG (FastAPI Style)] Target URL: {CLAP_A2A_SERVER_BASE_URL}/") 
    print(f"[ADK A2A TOOL LOG (FastAPI Style)] User Query: '{user_query}'")

    payload = {
        "message": user_query,
        "context": {}, 
        
    }

    async with httpx.AsyncClient(timeout=120.0) as http_client: 
        try:
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] Sending POST to {CLAP_A2A_SERVER_BASE_URL}/")
            response = await http_client.post(
                f"{CLAP_A2A_SERVER_BASE_URL}/", 
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"}
            )
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] Raw HTTP Status: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for 4xx/5xx responses

            response_data = response.json() # Expects direct JSON (A2AAgentResponse)
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] Response JSON: {json.dumps(response_data, indent=2)}")

            if response_data.get("status") == "success" and "message" in response_data:
                final_text = response_data["message"]
                print(f"[ADK A2A TOOL LOG (FastAPI Style)] Success. Returning: '{final_text[:200]}...'")
                return final_text
            else:
                error_msg = response_data.get("message", "Unknown error structure from CLAP FastAPI A2A server.")
                print(f"[ADK A2A TOOL LOG (FastAPI Style)] Error in response: {error_msg}")
                return f"Error from CLAP agent: {error_msg}"

        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            error_msg = f"Error: CLAP A2A server returned HTTP {e.response.status_code}. Response: {error_text[:200]}"
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] {error_msg}")
            return error_msg
        except httpx.RequestError as e:
            error_msg = f"Error: Could not connect to CLAP A2A service (HTTP RequestError). {type(e).__name__}: {e}"
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Error interacting with CLAP A2A agent (General Exception). {type(e).__name__}: {e}"
            print(f"[ADK A2A TOOL LOG (FastAPI Style)] {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

async def get_knowledge_from_clap_agent(query_for_rag: str) -> str:
    # Docstring remains the same
    """
    Delegates a complex query, suitable for Retrieval Augmented Generation (RAG)
    or specialized CLAP agent tools, to a remote CLAP agent accessible via A2A protocol.
    Use this for questions that require deep knowledge lookups or specific CLAP capabilities.

    Args:
        query_for_rag: The user's full query or question that needs to be answered by the CLAP RAG agent.
    
    Returns:
        The answer retrieved from the CLAP agent, or an error message.
    """
    return await call_clap_agent_via_a2a(user_query=query_for_rag)