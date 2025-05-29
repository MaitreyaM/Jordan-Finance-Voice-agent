import httpx
import json
import os

AGNO_A2A_FINANCIAL_SERVER_BASE_URL = os.getenv("AGNO_A2A_FINANCIAL_SERVER_URL", "http://localhost:10000")

async def _execute_agno_http_call(user_query_for_brief: str) -> str:
    tool_log_prefix = "[ADK Tool - _execute_agno_http_call]"
    target_url = f"{AGNO_A2A_FINANCIAL_SERVER_BASE_URL}/"
    
    print(f"\n{tool_log_prefix} Calling Agno Financial Server (direct FastAPI POST).")
    print(f"{tool_log_prefix} Target URL: {target_url}")
    print(f"{tool_log_prefix} User Query/Context: '{user_query_for_brief}'")

    payload = {
        "message": user_query_for_brief,
        
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
            
            if response.status_code != 200:
                error_text = response.text
                error_msg = f"Error: Agno server returned HTTP {response.status_code}. Response: {error_text[:200]}"
                print(f"{tool_log_prefix} {error_msg}")
                return error_msg

            response_data = response.json() 
            print(f"{tool_log_prefix} Response JSON: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

            response_status = response_data.get("status")
            response_message = response_data.get("message", "No message content from Agno server.")

            if response_status == "success":
                print(f"{tool_log_prefix} Success. Returning report: '{response_message[:200]}...'")
                return response_message
            elif response_status == "auth_required": 
                print(f"{tool_log_prefix} AuthRequired. Returning message: '{response_message[:200]}...'")
               
                return response_message 
            else: 
                error_msg_detail = response_message
                print(f"{tool_log_prefix} Error or unexpected status '{response_status}' in response data: {error_msg_detail}")
                return f"Error from Agno Financial Agent (status: {response_status}): {error_msg_detail}"

        except httpx.HTTPStatusError as e:
            error_text = e.response.text if e.response else "No response body"
            error_msg = f"Error: Agno server returned HTTP {e.response.status_code}. Response: {error_text[:200]}"
            print(f"{tool_log_prefix} {error_msg}")
            return error_msg
        except httpx.RequestError as e: 
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
    Retrieves the morning financial market brief or handles Zerodha actions by calling a specialized Agno agent
    via a direct HTTP POST request to its FastAPI endpoint.

    Args:
        user_prompt_context: Specific focus for the brief or Zerodha command.
    
    Returns:
        A string containing the financial market brief, Zerodha data, authentication URL, or an error message.
    """
    return await _execute_agno_http_call(user_query_for_brief=user_prompt_context)