# ADK/adk-voice-agent/app/main.py
import asyncio
import base64
import json
import os
from pathlib import Path
from typing import AsyncIterable, List 
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect # Import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from google.adk.agents import LiveRequestQueue, Agent 
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

from jarvis.agent import get_jarvis_agent_definition # Your AGENT_INSTRUCTION is here

load_dotenv()
APP_NAME = "ADK Streaming Jordon"
session_service = InMemorySessionService()

# URLs for MCP Servers
GMAIL_MCP_SERVER_URL = "http://localhost:8001/sse" 
LOCAL_ZERODHA_MCP_SSE_URL = "http://localhost:8002/sse"
# Placeholder for Alpha Vantage if you re-add it. Ensure it's an SSE URL.
# ALPHA_VANTAGE_SSE_MCP_URL = "YOUR_ALPHA_VANTAGE_SSE_MCP_URL_HERE" 

fetched_gmail_mcp_tools: List = []
fetched_zerodha_mcp_tools: List = []
# fetched_alphavantage_mcp_tools: List = [] # Commented out for now
gmail_mcp_exit_stack = None
zerodha_mcp_exit_stack = None
# alphavantage_mcp_exit_stack = None # Commented out for now

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fetched_gmail_mcp_tools, gmail_mcp_exit_stack
    global fetched_zerodha_mcp_tools, zerodha_mcp_exit_stack
    # global fetched_alphavantage_mcp_tools, alphavantage_mcp_exit_stack # Commented out

    print("FastAPI app starting up...")
    try:
        # 1. Gmail MCP Toolset
        print(f"Initializing MCPToolset for Gmail server at {GMAIL_MCP_SERVER_URL}...")
        tools_gmail, exit_stack_gmail = await MCPToolset.from_server(
            connection_params=SseServerParams(url=GMAIL_MCP_SERVER_URL)
        )
        fetched_gmail_mcp_tools = tools_gmail
        gmail_mcp_exit_stack = exit_stack_gmail
        if fetched_gmail_mcp_tools:
            print(f"Successfully fetched {len(fetched_gmail_mcp_tools)} tools from Gmail MCP server: {[tool.name for tool in fetched_gmail_mcp_tools]}.")
        else:
            print("Gmail MCPToolset.from_server returned an empty list of tools or failed.")

        # 2. Zerodha MCP Toolset (Local SSE)
        print(f"Initializing MCPToolset for local Zerodha SSE server at {LOCAL_ZERODHA_MCP_SSE_URL}...")
        tools_zerodha, exit_stack_zerodha = await MCPToolset.from_server(
            connection_params=SseServerParams(url=LOCAL_ZERODHA_MCP_SSE_URL)
        )
        fetched_zerodha_mcp_tools = tools_zerodha
        zerodha_mcp_exit_stack = exit_stack_zerodha
        if fetched_zerodha_mcp_tools:
            print(f"Successfully fetched {len(fetched_zerodha_mcp_tools)} tools from local Zerodha MCP server: {[tool.name for tool in fetched_zerodha_mcp_tools]}.")
        else:
            print("Local Zerodha MCPToolset.from_server returned an empty list of tools or failed.")
        
        # 3. Alpha Vantage MCP Toolset (SSE) - Placeholder if you add it back
        # print(f"Initializing MCPToolset for Alpha Vantage SSE server at {ALPHA_VANTAGE_SSE_MCP_URL}...")
        # try:
        #     tools_av, exit_stack_av = await MCPToolset.from_server(
        #         connection_params=SseServerParams(url=ALPHA_VANTAGE_SSE_MCP_URL)
        #     )
        #     fetched_alphavantage_mcp_tools = tools_av
        #     alphavantage_mcp_exit_stack = exit_stack_av
        #     if fetched_alphavantage_mcp_tools:
        #         print(f"Successfully fetched {len(fetched_alphavantage_mcp_tools)} tools from Alpha Vantage MCP: {[tool.name for tool in fetched_alphavantage_mcp_tools]}.")
        #         # Optional: Print schemas for debugging
        #         # print(f"--- Alpha Vantage Tool Schemas ---")
        #         # for tool_instance in fetched_alphavantage_mcp_tools:
        #         #     if hasattr(tool_instance, '_get_declaration'):
        #         #         print(f"Tool: {tool_instance.name}, Schema: {tool_instance._get_declaration().parameters}")
        #     else:
        #         print("Alpha Vantage MCPToolset.from_server returned an empty list of tools or failed.")
        # except Exception as e:
        #     print(f"Error initializing Alpha Vantage MCPToolset: {e}. Alpha Vantage tools will be unavailable.")
        
        yield 
    
    except Exception as e:
        print(f"Error during MCPToolset initializations in lifespan: {e}")
        # raise # Optionally re-raise to halt startup on critical MCP failure
    finally:
        print("FastAPI app shutting down...")
        if gmail_mcp_exit_stack:
            print("Closing Gmail MCP server connection...")
            await gmail_mcp_exit_stack.aclose()
        if zerodha_mcp_exit_stack:
            print("Closing local Zerodha MCP server connection...")
            await zerodha_mcp_exit_stack.aclose()
        # if alphavantage_mcp_exit_stack: 
        #     print("Closing Alpha Vantage MCP server connection...")
        #     await alphavantage_mcp_exit_stack.aclose()

app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root_html_endpoint(): 
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

def start_agent_session(session_id: str, is_audio: bool = False):
    session = session_service.create_session(
        app_name=APP_NAME, user_id=session_id, session_id=session_id,
    )
    # Combine tools from successfully connected MCPs
    all_dynamic_mcp_tools = fetched_gmail_mcp_tools + fetched_zerodha_mcp_tools 
    # If you add Alpha Vantage back:
    # all_dynamic_mcp_tools += fetched_alphavantage_mcp_tools
    
    agent_definition = get_jarvis_agent_definition(all_dynamic_mcp_tools)
    jarvis_agent_instance = Agent(**agent_definition)
    runner = Runner(
        app_name=APP_NAME, agent=jarvis_agent_instance, session_service=session_service,
    )
    modality = "AUDIO" if is_audio else "TEXT"
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck"))
    )
    config = {"response_modalities": [modality], "speech_config": speech_config}
    if is_audio:
        config["output_audio_transcription"] = {} 
        print(f"RunConfig for audio session {session_id}: {config}")
    else:
        print(f"RunConfig for text session {session_id}: {config}")
        
    run_config = RunConfig(**config)
    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live( 
        session=session, live_request_queue=live_request_queue, run_config=run_config,
    )
    return live_events, live_request_queue

async def agent_to_client_messaging(
    websocket: WebSocket, live_events: AsyncIterable[Event | None]
):
    """Agent to client communication (original logic for partials)"""
    try:
        async for event in live_events:
            if event is None:
                continue
            if event.turn_complete or event.interrupted:
                message = {
                    "turn_complete": event.turn_complete,
                    "interrupted": event.interrupted,
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT TO CLIENT]: {message}")
                continue
            
            # Original logic for processing parts - crucial for streaming
            if event.content and event.content.parts:
                part = event.content.parts[0] # Assuming one main part for streaming
                if not isinstance(part, types.Part):
                    continue

                if part.text and event.partial: # Key: This was the original logic for text streaming
                    message = {
                        "mime_type": "text/plain", "data": part.text, "role": "model",
                    }
                    await websocket.send_text(json.dumps(message))
                    print(f"[AGENT TO CLIENT]: text/plain (partial): {part.text}")
                elif part.text and not event.partial: # Handle complete text part
                    message = {
                        "mime_type": "text/plain", "data": part.text, "role": "model",
                    }
                    await websocket.send_text(json.dumps(message))
                    print(f"[AGENT TO CLIENT]: text/plain (complete): {part.text}")


                is_audio = (
                    part.inline_data
                    and part.inline_data.mime_type
                    and part.inline_data.mime_type.startswith("audio/pcm")
                )
                if is_audio:
                    audio_data = part.inline_data.data
                    if audio_data:
                        message = {
                            "mime_type": "audio/pcm",
                            "data": base64.b64encode(audio_data).decode("ascii"),
                            "role": "model",
                        }
                        await websocket.send_text(json.dumps(message))
                        print(f"[AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")
    except WebSocketDisconnect:
        print(f"agent_to_client_messaging: WebSocket disconnected for {websocket.client}.")
    except Exception as e:
        print(f"Error in agent_to_client_messaging for {websocket.client}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()


async def client_to_agent_messaging(
    websocket: WebSocket, live_request_queue: LiveRequestQueue
):
    """Client to agent communication (original logic)"""
    try:
        while True:
            message_json = await websocket.receive_text()
            message = json.loads(message_json)
            mime_type = message["mime_type"]
            data = message["data"]
            role = message.get("role", "user")
            if mime_type == "text/plain":
                content = types.Content(role=role, parts=[types.Part.from_text(text=data)])
                live_request_queue.send_content(content=content)
                print(f"[CLIENT TO AGENT PRINT]: {data}")
            elif mime_type == "audio/pcm":
                decoded_data = base64.b64decode(data)
                live_request_queue.send_realtime(
                    types.Blob(data=decoded_data, mime_type=mime_type)
                )
                if len(decoded_data) > 0 : print(f"[CLIENT TO AGENT]: audio/pcm chunk received ({len(decoded_data)} bytes)")
            else:
                # Instead of raising ValueError, which would break the client_to_agent_messaging loop,
                # log a warning and continue. The client might send unsupported types.
                print(f"Warning: Mime type '{mime_type}' not supported by client_to_agent_messaging. Message ignored.")
    except WebSocketDisconnect:
        print(f"client_to_agent_messaging: WebSocket disconnected for {websocket.client}.")
    except Exception as e:
        print(f"Error in client_to_agent_messaging for {websocket.client}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    is_audio: str = Query(...),
):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"Client #{session_id} ({client_host}) connected, audio mode: {is_audio}")

    live_events, live_request_queue = start_agent_session(
        session_id, is_audio == "true"
    )

    agent_to_client_task = asyncio.create_task(
        agent_to_client_messaging(websocket, live_events), name=f"a2c_{session_id}"
    )
    client_to_agent_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue), name=f"c2a_{session_id}"
    )
    
    # Using the simpler asyncio.gather from your initial working version
    try:
        await asyncio.gather(agent_to_client_task, client_to_agent_task)
    except WebSocketDisconnect:
        print(f"WebSocket for client #{session_id} disconnected (caught in gather).")
    except Exception as e:
        print(f"Error in asyncio.gather for client #{session_id}: {type(e).__name__} - {e}")
    finally:
        print(f"Cleaning up tasks for client #{session_id}...")
        if not agent_to_client_task.done():
            agent_to_client_task.cancel()
        if not client_to_agent_task.done():
            client_to_agent_task.cancel()
        await asyncio.gather(agent_to_client_task, client_to_agent_task, return_exceptions=True)
        print(f"Client #{session_id} ({client_host}) processing finished.")