import asyncio
import base64
import json
import os
from pathlib import Path
from typing import AsyncIterable, List # Python's built-in List for type hinting
from contextlib import asynccontextmanager # For FastAPI lifespan

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ADK specific imports - ensure Agent or LlmAgent is the correct one for instantiation
from google.adk.agents import LiveRequestQueue, Agent # Or from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams

from jarvis.agent import get_jarvis_agent_definition, GMAIL_MCP_SERVER_URL


load_dotenv()
APP_NAME = "ADK Streaming example"
session_service = InMemorySessionService()

fetched_mcp_tools: List = [] 
mcp_toolset_exit_stack = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fetched_mcp_tools, mcp_toolset_exit_stack
    print("FastAPI app starting up... Initializing MCPToolset.")
    try:
        # MCPToolset.from_server returns a tuple: (list_of_tools, async_exit_stack)
        tools_from_mcp, exit_stack = await MCPToolset.from_server(
            connection_params=SseServerParams(
                url=GMAIL_MCP_SERVER_URL,
                # Add headers here if your MCP server requires them (e.g., for authentication)
                # headers={"Authorization": "Bearer YOUR_TOKEN"}
            )
            # Optional: tool_filter to select only specific tools from the MCP server
            # tool_filter=['send_email_tool', 'fetch_recent_emails']
        )
        fetched_mcp_tools = tools_from_mcp
        mcp_toolset_exit_stack = exit_stack # Store the exit_stack for cleanup
        if fetched_mcp_tools:
            print(f"Successfully fetched {len(fetched_mcp_tools)} tools from MCP server: {[tool.name for tool in fetched_mcp_tools]}.")
        else:
            # This case could happen if connection fails or server has no tools
            print("MCPToolset.from_server returned an empty list of tools or failed to fetch.")
        yield 
    except Exception as e:
        print(f"Error during MCPToolset initialization in lifespan: {e}")
        
        raise
    finally:
        print("FastAPI app shutting down...")
        if mcp_toolset_exit_stack:
            print("Closing MCP server connection via exit_stack...")
            await mcp_toolset_exit_stack.aclose()
            print("MCP server connection closed.")

# Create FastAPI app with the lifespan manager
app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root_html_endpoint(): 
    """Serves the index.html"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# This function was originally synchronous, and it should remain so if it doesn't 'await' anything.
# The fetched_mcp_tools are populated by the async lifespan manager before this is called.
def start_agent_session(session_id: str, is_audio: bool = False):
    """Starts an agent session"""
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        session_id=session_id,
    )

    agent_definition = get_jarvis_agent_definition(fetched_mcp_tools)

    # Instantiate the agent.
    # If your original code used `from jarvis.agent import root_agent` and root_agent was an LlmAgent,
    # you should use LlmAgent here for consistency.
    # The `adk_mcp.md` uses `LlmAgent`.
    # from google.adk.agents.llm_agent import LlmAgent # Potentially needed
    # For now, using Agent as per your main.py import: from google.adk.agents import Agent
    jarvis_agent_instance = Agent(**agent_definition)

    runner = Runner(
        app_name=APP_NAME,
        agent=jarvis_agent_instance,
        session_service=session_service,
    )

    modality = "AUDIO" if is_audio else "TEXT"
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck"))
    )
    config = {"response_modalities": [modality], "speech_config": speech_config}
    if is_audio:
        config["output_audio_transcription"] = {}
    run_config = RunConfig(**config)

    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live( # This call is synchronous
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    return live_events, live_request_queue


async def agent_to_client_messaging(
    websocket: WebSocket, live_events: AsyncIterable[Event | None]
):
    """Agent to client communication"""
    while True:
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
            part = event.content and event.content.parts and event.content.parts[0]
            if not part:
                continue
            if not isinstance(part, types.Part):
                continue
            if part.text and event.partial:
                message = {
                    "mime_type": "text/plain",
                    "data": part.text,
                    "role": "model",
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT TO CLIENT]: text/plain: {part.text}")
            is_audio = (
                part.inline_data
                and part.inline_data.mime_type
                and part.inline_data.mime_type.startswith("audio/pcm")
            )
            if is_audio:
                audio_data = part.inline_data and part.inline_data.data
                if audio_data:
                    message = {
                        "mime_type": "audio/pcm",
                        "data": base64.b64encode(audio_data).decode("ascii"),
                        "role": "model",
                    }
                    await websocket.send_text(json.dumps(message))
                    print(f"[AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")


async def client_to_agent_messaging(
    websocket: WebSocket, live_request_queue: LiveRequestQueue
):
    """Client to agent communication"""
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
            print(f"[CLIENT TO AGENT]: audio/pcm: {len(decoded_data)} bytes")
        else:
            raise ValueError(f"Mime type not supported: {mime_type}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    is_audio: str = Query(...),
):
    await websocket.accept()
    print(f"Client #{session_id} connected, audio mode: {is_audio}")

    live_events, live_request_queue = start_agent_session( # Call the synchronous version
        session_id, is_audio == "true"
    )

    agent_to_client_task = asyncio.create_task(
        agent_to_client_messaging(websocket, live_events)
    )
    client_to_agent_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue)
    )
    await asyncio.gather(agent_to_client_task, client_to_agent_task)
    print(f"Client #{session_id} disconnected")