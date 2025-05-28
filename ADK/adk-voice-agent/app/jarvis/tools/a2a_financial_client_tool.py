# HOLBOXATHON/ADK/adk-voice-agent/app/jarvis/tools/a2a_financial_client_tool.py
import httpx
import json
import os
# No A2AClient or A2A types needed for direct HTTP call

# URL of your Agno Financial A2A Server (FastAPI style)
AGNO_A2A_FINANCIAL_SERVER_BASE_URL = os.getenv("AGNO_A2A_FINANCIAL_SERVER_URL", "http://localhost:10000")

async def _execute_agno_http_call(user_query_for_brief: str) -> str: # Renamed
    tool_log_prefix = "[ADK Tool - _execute_agno_http_call]"
    target_url = f"{AGNO_A2A_FINANCIAL_SERVER_BASE_URL}/" # Assuming POST to root
    
    print(f"\n{tool_log_prefix} Calling Agno Financial Server (direct FastAPI POST).")
    print(f"{tool_log_prefix} Target URL: {target_url}")
    print(f"{tool_log_prefix} User Query/Context: '{user_query_for_brief}'")

    # This payload matches AgnoAgentApiRequest in run_financial_agent_server_fastapi.py
    payload = {
        "message": user_query_for_brief,
        "context": {}, 
        "session_id": None 
    }
    print(f"{tool_log_prefix} Request Payload: {json.dumps(payload)}")

    async with httpx.AsyncClient(timeout=180.0) as http_client:
        try:
            print(f"{tool_log_prefix} Sending POST to {target_url}...")
            response = await http_client.post(
                target_url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"}
            )
            print(f"{tool_log_prefix} Raw HTTP Status: {response.status_code}")
            
            # Check for non-2xx status codes first
            if response.status_code != 200:
                error_text = response.text
                error_msg = f"Error: Agno server returned HTTP {response.status_code}. Response: {error_text[:200]}"
                print(f"{tool_log_prefix} {error_msg}")
                return error_msg # Return the server's error message if available

            response_data = response.json() 
            print(f"{tool_log_prefix} Response JSON: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

            # Process the simple JSON response (matches AgnoAgentApiResponse)
            if response_data.get("status") == "success" and "message" in response_data:
                final_text = response_data["message"]
                print(f"{tool_log_prefix} Success. Returning report: '{final_text[:200]}...'")
                return final_text
            else:
                error_msg_detail = response_data.get("message", "Unknown error structure from Agno server.")
                print(f"{tool_log_prefix} Error in response data: {error_msg_detail}")
                return f"Error from Agno Financial Agent: {error_msg_detail}"

        except httpx.HTTPStatusError as e: # Should be caught by status_code check now
            error_text = e.response.text if e.response else "No response body"
            error_msg = f"Error: Agno server returned HTTP {e.response.status_code}. Response: {error_text[:200]}"
            print(f"{tool_log_prefix} {error_msg}")
            return error_msg
        except httpx.RequestError as e: # Catches network errors, timeouts
            error_msg = f"Error: Could not connect to Agno service (HTTP RequestError). {type(e).__name__}: {e}"
            print(f"{tool_log_prefix} {error_msg}")
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"Error: Could not decode JSON response from Agno server. {type(e).__name__}: {e}. Response text: {response.text[:200]}"
            print(f"{tool_log_prefix} {error_msg}")
            return error_msg
        except Exception as e: 
            error_msg = f"Error in tool for Agno agent. {type(e).__name__}: {e}"
            print(f"{tool_log_prefix} {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

async def get_financial_market_brief(user_prompt_context: str = "Generate standard morning brief.") -> str:
    """
    Retrieves the morning financial market brief by calling a specialized Agno agent
    via a direct HTTP POST request to its FastAPI endpoint.

    Args:
        user_prompt_context: Optional. Specific focus for the brief.
    
    Returns:
        A string containing the financial market brief, or an error message.
    """
    return await _execute_agno_http_call(user_query_for_brief=user_prompt_context)