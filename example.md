Directory structure:
└── version_5_a2a_sdk/
    ├── README.md
    ├── main.py
    ├── .gitignore
    ├── agents/
    │   └── tell_time_agent/
    │       ├── __init__.py
    │       ├── __main__.py
    │       ├── agent.py
    │       ├── agent_executor.py
    │       └── __pycache__/
    └── client/
        ├── client.py
        └── __pycache__/


Files Content:

================================================
FILE: version_5_a2a_sdk/README.md
================================================
# 🕒 version_5_a2a_sdk

## 🌟 Purpose
This version demonstrates a **minimal educational setup** using Google's [Agent-to-Agent (A2A)](https://github.com/google/A2A) protocol via the official **`a2a-python` SDK**.

It includes:
- A single **TellTime agent** that returns the current system time
- A single **A2A client** that sends messages and receives streaming responses
- A clear example of multi-turn streaming via task updates

This version is ideal as a **first hands-on project** to understand A2A SDK usage.

---

## 🚀 Features

- ✅ Minimal working agent with LangChain + Gemini + time tool
- ✅ Fully async A2A client using the SDK
- ✅ Streaming responses with task update events
- ✅ Supports multi-turn interactions
- ✅ Clean, modular folder structure

---

## 📦 Project Structure

```bash
version_5_a2a_sdk/
→ agents/
    → tell_time_agent/
        agent.py             # LangChain-based TellTime agent logic
        agent_executor.py    # Executor that connects the agent to A2A runtime
        __main__.py          # Starts the agent server (entry point)
        __init__.py          # Required to treat this folder as a module

→ client/
    client.py               # A2A SDK client that streams messages to the agent

main.py                    # Optional runner stub
README.md                  # You're reading it!
````

---

## 🛠️ Prerequisites

* Python 3.13+
* [`uv`](https://github.com/astral-sh/uv) for clean environment setup
* A valid `GOOGLE_API_KEY` for Gemini

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/theailanguage/a2a_samples.git
cd version_5_a2a_sdk
uv init --python python3.13
uv venv
source .venv/bin/activate
uv add a2a-sdk langchain langgraph google-genai httpx python-dotenv langchain-google-genai uvicorn click rich
uv sync --all-groups
touch .env
echo "GOOGLE_API_KEY=your_key_here" > .env
```

---

## 🧪 Running the Project

### 🟢 Step 1: Start the TellTime Agent Server

```bash
uv run python3 -m agents.tell_time_agent
```

This launches the agent server at `http://localhost:10000`.

### 🟡 Step 2: Run the A2A Client

```bash
uv run python3 client/client.py
```

This lets you:

* Ask one-shot queries like *"What time is it?"*
* Watch streaming updates in real time
* Handle follow-up input for multi-turn dialogs

---

## 🤔 How the Agent Works (agent.py)

* Uses LangChain ReAct agent with Gemini (Flash model)
* Has one tool:

```python
@tool
def get_time_now():
    return {"current_time": datetime.now().strftime("%H:%M:%S")}
```

* Responds with a structured format:

```python
{"status": "completed", "message": "The current time is ..."}
```

---

## 💡 What You'll Learn

| Concept               | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| A2A SDK usage         | Connect, send, and stream tasks to an agent                    |
| AgentExecutor pattern | How the agent gets plugged into A2A's server loop              |
| Streaming flow        | Watch how `working` → `completed` messages are delivered       |
| Multi-turn handling   | Learn how agents can request clarification/input from the user |

---

Happy coding ✨


================================================
FILE: version_5_a2a_sdk/main.py
================================================
def main():
    print("Hello from version-5-a2a-sdk!")


if __name__ == "__main__":
    main()



================================================
FILE: version_5_a2a_sdk/.gitignore
================================================
.env
.venv
.python-version
pyproject.toml
uv.lock



================================================
FILE: version_5_a2a_sdk/agents/tell_time_agent/__init__.py
================================================



================================================
FILE: version_5_a2a_sdk/agents/tell_time_agent/__main__.py
================================================
# =============================================================================
# agents/tell_time_agent/main.py
# =============================================================================
# Purpose:
# This file starts the A2A-compatible agent server.
# It sets up environment, configures the task execution handler, agent card,
# and launches a Starlette-based web server for incoming agent tasks.
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os                      # Provides access to environment variables
import sys                     # Used for exiting if setup is incomplete

import click                   # Helps define command-line interface for running the server
import httpx                   # HTTP client used for async push notifications
from dotenv import load_dotenv  # Loads .env file for environment variables

# Import the agent logic and its executor
from .agent import TellTimeAgent                # Defines the actual agent logic
from .agent_executor import TellTimeAgentExecutor  # Bridges the agent with A2A server

# Import A2A SDK components to create a working agent server
from a2a.server.apps import A2AStarletteApplication  # Main application class based on Starlette
from a2a.server.request_handlers import DefaultRequestHandler  # Default logic for handling tasks
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore  # In-memory task manager and notifier
from a2a.types import AgentCard, AgentSkill, AgentCapabilities  # Agent metadata definitions

# -----------------------------------------------------------------------------
# Load environment variables from .env file if present
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Main entry point to launch the agent server
# -----------------------------------------------------------------------------
@click.command()
@click.option('--host', 'host', default='localhost')     # Host where the agent will listen (default: localhost)
@click.option('--port', 'port', default=10000)            # Port where the agent will listen (default: 10000)
def main(host: str, port: int):
    # Check if the required API key is set in environment
    if not os.getenv('GOOGLE_API_KEY'):
        print("GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)  # Exit the program if API key is missing

    # Create HTTP client (used for push notifications)
    client = httpx.AsyncClient()

    # Set up the request handler for processing incoming tasks
    handler = DefaultRequestHandler(
        agent_executor=TellTimeAgentExecutor(),  # Hook in our custom agent
        task_store=InMemoryTaskStore(),          # Use in-memory store to manage task state
        push_notifier=InMemoryPushNotifier(client),  # Enable server push updates (e.g., via webhook)
    )

    # Set up the A2A server application using agent card and handler
    server = A2AStarletteApplication(
        agent_card=build_agent_card(host, port),  # Provide agent capabilities and skills
        http_handler=handler,                     # Attach the request handler
    )

    # Start the server using uvicorn async server
    import uvicorn
    uvicorn.run(server.build(), host=host, port=port)

# -----------------------------------------------------------------------------
# Defines the metadata card for this agent
# -----------------------------------------------------------------------------
def build_agent_card(host: str, port: int) -> AgentCard:
    return AgentCard(
        name="TellTime Agent",                                      # Human-readable name of the agent
        description="Tells the current system time.",               # Short description
        url=f"http://{host}:{port}/",                               # Full URL where the agent is reachable
        version="1.0.0",                                            # Version of the agent
        capabilities=AgentCapabilities(streaming=True, pushNotifications=True),  # Supported features
        defaultInputModes=TellTimeAgent.SUPPORTED_CONTENT_TYPES,    # Accepted input content types
        defaultOutputModes=TellTimeAgent.SUPPORTED_CONTENT_TYPES,   # Returned output content types
        skills=[                                                     # Skills this agent supports (currently one)
            AgentSkill(
                id="tell_time",                                     # Unique ID for the skill
                name="Get Current Time",                           # Display name
                description="Tells the current system time in HH:MM:SS format.",
                tags=["time", "clock"],                             # Useful tags for search/filtering
                examples=["What time is it?", "Tell me the current time."],  # Example user prompts
            )
        ],
    )

# -----------------------------------------------------------------------------
# This ensures the server starts when you run `python -m agents.tell_time_agent.main`
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()



================================================
FILE: version_5_a2a_sdk/agents/tell_time_agent/agent.py
================================================
# =============================================================================
# agents/tell_time_agent/agent.py
# =============================================================================
# Purpose:
# This file defines the TellTimeAgent.
# - It uses LangChain ReAct agent with Gemini (via langchain-google-genai)
# - It supports streaming responses
# - It defines a simple tool: get_time_now()
# - It handles structured responses with support for multi-turn logic
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from datetime import datetime                      # To get the current system time
from typing import Any, Literal, AsyncIterable     # Type annotations for cleaner, safer code
import logging                                     # To optionally print internal debug/info messages

from pydantic import BaseModel                     # Used to define response validation schemas
from langchain_core.messages import AIMessage, ToolMessage  # Message types from LangChain for interpreting outputs
from langchain_core.runnables.config import RunnableConfig  # To configure LangChain's agent calls
from langchain_core.tools import tool              # To define a callable tool for the agent
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini LLM integration
from langgraph.checkpoint.memory import MemorySaver         # In-memory store to manage multi-turn conversation state
from langgraph.prebuilt import create_react_agent           # To build a full LangGraph ReAct agent from components

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)               # Setup logging for debugging (not actively used)
memory = MemorySaver()                             # Simple in-memory storage for graph state

# -----------------------------------------------------------------------------
# Tool Definition: get_time_now()
# -----------------------------------------------------------------------------
@tool
def get_time_now() -> dict[str, str]:
    """Returns the current system time in HH:MM:SS format."""
    return {"current_time": datetime.now().strftime("%H:%M:%S")}

# -----------------------------------------------------------------------------
# ResponseFormat schema: validates what the agent returns
# -----------------------------------------------------------------------------
class ResponseFormat(BaseModel):
    status: Literal["completed", "input_required", "error"]  # Structured status of the agent reply
    message: str                                              # The message that will be shown to the user

# -----------------------------------------------------------------------------
# TellTimeAgent Class Definition
# -----------------------------------------------------------------------------
class TellTimeAgent:
    """
    LangChain ReAct agent that answers time-related queries.
    - Only uses the get_time_now tool
    - Responds based on structured format
    - Powered by Gemini Flash model
    """

    # Instruction given to the agent LLM
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for time-related queries. "
        "Use the 'get_time_now' tool when users ask for the current time to get the time in HH:MM:SS format. "
        "Convert this time to the requested format by the user on your own. You are allowed to do that"
    )

    # Response formatting guidelines expected by the agent
    RESPONSE_FORMAT_INSTRUCTION = (
        "Use 'completed' if the task is done, 'input_required' if clarification is needed, "
        "and 'error' if something fails. Always include a user-facing message."
    )

    def __init__(self):
        # Initialize the Gemini LLM model (fast variant)
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        # Register the available tools for this agent
        self.tools = [get_time_now]

        # Create a complete ReAct-style agent graph using LangGraph
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

    # -----------------------------------------------------------------------------
    # The `stream` method streams partial updates from the agent in real-time.
    # It returns responses step-by-step as the agent works, instead of waiting for everything at once.
    # -----------------------------------------------------------------------------

    # Define an asynchronous method that returns an iterable (like a loop) of dictionaries.
    # Each dictionary is a partial update from the agent.
    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        """
        This function is used when a user sends a message to the agent.
        Instead of waiting for a single response, it gives us updates as they happen.

        - 'query' is the user’s question or command (e.g., "What time is it?")
        - 'session_id' is a unique ID for this user's interaction (to maintain context)
        - It yields updates such as "Looking up time...", "Processing...", and the final result.
        """

        # --------------------------------------------------------------
        # Set up a configuration that ties this request to a session.
        # LangGraph needs a session/thread ID to track the conversation.
        # --------------------------------------------------------------
        config: RunnableConfig = {
            "configurable": {
                "thread_id": session_id  # Unique ID to separate one user conversation from another
            }
        }

        # --------------------------------------------------------------
        # This is the input format LangGraph expects: a list of messages.
        # Each message is a tuple: ("who", "what they said").
        # Here, we send just one message from the "user".
        # --------------------------------------------------------------
        inputs = {"messages": [("user", query)]}

        # --------------------------------------------------------------
        # Begin streaming the agent's thinking steps using LangGraph.
        # Each 'item' is a step in the reasoning (like a thought bubble).
        # 'stream_mode="values"' tells it to yield useful results only.
        # --------------------------------------------------------------
        for item in self.graph.stream(inputs, config, stream_mode="values"):

            # ----------------------------------------------------------
            # Get the most recent message from the list of all messages.
            # The agent might add multiple messages during the thinking.
            # We only care about the last one for status.
            # ----------------------------------------------------------
            message = item["messages"][-1]

            # ----------------------------------------------------------
            # If the message is from the AI and includes tool usage,
            # that means the agent is about to call a tool like get_time_now.
            # ----------------------------------------------------------
            if isinstance(message, AIMessage) and message.tool_calls:
                yield {  # Yield means "send this result immediately" before continuing
                    "is_task_complete": False,         # Not done yet
                    "require_user_input": False,       # No follow-up question to the user (yet)
                    "content": "Looking up the current time...",  # What to display on the UI or CLI
                }

            # ----------------------------------------------------------
            # If the message is from the tool (like get_time_now),
            # this means the agent just received the result and is working on formatting it.
            # ----------------------------------------------------------
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,         # Still not done yet
                    "require_user_input": False,       # No clarification from the user is needed
                    "content": "Processing the time result...",  # Let the user know progress
                }

        # --------------------------------------------------------------
        # Once the stream ends (no more partial steps), send the final result.
        # This could be a completed message or a follow-up question.
        # --------------------------------------------------------------
        yield self._final_response(config)

    # -----------------------------------------------------------------------------
    # This private method gives the final structured result after the stream ends.
    # It reads the agent’s final decision and returns a dictionary with flags.
    # -----------------------------------------------------------------------------
    def _final_response(self, config: RunnableConfig) -> dict[str, Any]:
        """
        After all streaming messages are done, this function checks what the agent finally decided.
        It uses the config to find the saved response (called 'structured_response').
        """

        # Get the internal memory state from the LangGraph session
        state = self.graph.get_state(config)

        # Pull out the structured result (should match the ResponseFormat schema)
        structured = state.values.get("structured_response")

        # --------------------------------------------------------------
        # If the structured result is valid, use its status and message.
        # The agent gives us:
        #   - status = "completed" or "input_required" or "error"
        #   - message = the final output or a clarification question
        # --------------------------------------------------------------
        if isinstance(structured, ResponseFormat):
            if structured.status == "completed":
                return {
                    "is_task_complete": True,              # Mark this as done
                    "require_user_input": False,           # No further input needed
                    "content": structured.message,         # Show the user the final result
                }
            if structured.status in ("input_required", "error"):
                return {
                    "is_task_complete": False,             # Not done yet
                    "require_user_input": True,            # Ask the user to clarify
                    "content": structured.message,         # The question or error to show
                }

        # --------------------------------------------------------------
        # If the agent response was broken or missing — handle gracefully
        # --------------------------------------------------------------
        print("[DEBUG] structured response:", structured)  # Print for debugging in the console

        return {
            "is_task_complete": False,                     # Don't mark this task as complete
            "require_user_input": True,                    # Ask the user to rephrase
            "content": "Unable to process your request at the moment. Please try again.",  # Default fallback message
        }
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]  # Declares formats this agent can handle for input/output



================================================
FILE: version_5_a2a_sdk/agents/tell_time_agent/agent_executor.py
================================================
# =============================================================================
# agents/tell_time_agent/agent_executor.py
# =============================================================================
# Purpose:
# This file defines the "executor" that acts as a bridge between the A2A server
# and the underlying TellTime agent. It listens to tasks and dispatches them to
# the agent, then sends back task updates and results through the event queue.
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from .agent import TellTimeAgent  # Imports the TellTimeAgent class from the same directory

# Importing base classes from the A2A SDK to define agent behavior
from a2a.server.agent_execution import AgentExecutor  # Base class for defining agent task executor logic
from a2a.server.agent_execution import RequestContext  # Holds information about the incoming user query and context

# EventQueue is used to push updates back to the A2A server (e.g., task status, results)
from a2a.server.events.event_queue import EventQueue

# Importing event and status types for responding to client
from a2a.types import (
    TaskArtifactUpdateEvent,  # Event for sending result artifacts back to the client
    TaskStatusUpdateEvent,   # Event for sending status updates (e.g., working, completed)
    TaskStatus,              # Object that holds the current status of the task
    TaskState,               # Enum that defines states: working, completed, input_required, etc.
)

# Utility functions to create standardized message and artifact formats
from a2a.utils import (
    new_agent_text_message,  # Creates a message object from agent to client
    new_task,                # Creates a new task object from the initial message
    new_text_artifact        # Creates a textual result artifact
)

# -----------------------------------------------------------------------------
# TellTimeAgentExecutor: Connects the agent logic to A2A server infrastructure
# -----------------------------------------------------------------------------

class TellTimeAgentExecutor(AgentExecutor):  # Define a new executor by extending A2A's AgentExecutor
    """
    This class connects the TellTimeAgent to the A2A server runtime. It implements
    the `execute` function to run tasks and push updates to the event queue.
    """

    def __init__(self):  # Constructor for the executor class
        self.agent = TellTimeAgent()  # Creates an instance of the TellTimeAgent for handling queries

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # This method is called when a new task is received

        query = context.get_user_input()  # Extracts the actual text of the user's message
        task = context.current_task      # Gets the task object if it already exists

        if not context.message:          # Ensure the message is not missing
            raise Exception('No message provided')  # Raise an error if something's wrong

        if not task:                     # If no existing task, this is a new interaction
            task = new_task(context.message)       # Create a new task based on the message
            event_queue.enqueue_event(task)        # Enqueue the new task to notify the A2A server

        # Use the agent to handle the query via async stream
        async for event in self.agent.stream(query, task.contextId):

            if event['is_task_complete']:  # If the task has been successfully completed
                # Send the result artifact to the A2A server
                event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        taskId=task.id,                 # ID of the task
                        contextId=task.contextId,       # ID of the context (conversation thread)
                        artifact=new_text_artifact(     # The result artifact
                            name='current_result',      # Name of the artifact
                            description='Result of request to agent.',  # Description
                            text=event['content'],      # The actual result text
                        ),
                        append=False,                   # Not appending to previous result
                        lastChunk=True,                 # This is the final chunk of the result
                    )
                )
                # Send final status update: task is completed
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,                 # ID of the task
                        contextId=task.contextId,       # Context ID
                        status=TaskStatus(state=TaskState.completed),  # Mark task as completed
                        final=True,                     # This is the last status update
                    )
                )

            elif event['require_user_input']:  # If the agent needs more information from user
                # Enqueue an input_required status with a message
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,                 # ID of the task
                        contextId=task.contextId,       # Context ID
                        status=TaskStatus(
                            state=TaskState.input_required,  # Set state as input_required
                            message=new_agent_text_message(  # Provide a message asking for input
                                event['content'],             # Message content
                                task.contextId,               # Context ID
                                task.id                       # Task ID
                            ),
                        ),
                        final=True,                     # Input_required is a final state until user responds
                    )
                )

            else:  # The task is still being processed (working)
                # Enqueue a status update showing ongoing work
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,                 # Task ID
                        contextId=task.contextId,       # Context ID
                        status=TaskStatus(
                            state=TaskState.working,    # Mark as still working
                            message=new_agent_text_message(
                                event['content'],       # Current progress or log
                                task.contextId,         # Context ID
                                task.id                  # Task ID
                            ),
                        ),
                        final=False,                    # More updates may follow
                    )
                )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Optional method to cancel long-running tasks (not supported here)
        raise Exception('Cancel not supported')  # Raise error since this agent doesn’t support canceling




================================================
FILE: version_5_a2a_sdk/client/client.py
================================================
# =============================================================================
# client/client.py
# =============================================================================
# Purpose:
# This file defines a dynamic async client built on top of the official
# A2A Python SDK. It can:
# - Detect agent capabilities (streaming or not)
# - Send queries in a loop
# - Handle single-turn or multi-turn conversations
# - Automatically pick between streaming and non-streaming flows
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import asyncio                      # Provides support for asynchronous programming and I/O operations
import json                         # Allows encoding and decoding JSON data
import traceback                    # Prints detailed tracebacks in case of errors
from uuid import uuid4              # Generates unique message IDs
from typing import Any              # Allows function arguments and variables to accept any type

import click                        # Library to easily create command-line interfaces
import httpx                        # Async HTTP client for sending requests to agents
from rich import print as rprint    # Enhanced print function to support colors and formatting
from rich.syntax import Syntax      # Used to highlight JSON output in the terminal

# Import the official A2A SDK client and related types
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,                      # Metadata about the agent
    SendMessageRequest,             # For sending regular (non-streaming) messages
    SendStreamingMessageRequest,    # For sending streaming messages
    MessageSendParams,              # Structure to hold message content
    SendMessageSuccessResponse,     # Represents a successful response from the agent
    Task,                           # Task object representing the agent's work unit
    TaskState,                      # Enum describing current task state (working, complete, etc.)
    GetTaskRequest,                 # Used to request status of a task
    TaskQueryParams,                # Parameters needed to fetch a specific task
)

# -----------------------------------------------------------------------------
# Helper: Create a message payload in expected A2A format
# -----------------------------------------------------------------------------
def build_message_payload(text: str, task_id: str | None = None, context_id: str | None = None) -> dict[str, Any]:
    # Constructs a dictionary payload that matches A2A message format
    return {
        "message": {
            "role": "user",  # The role of the message sender
            "parts": [{"kind": "text", "text": text}],  # The actual message content
            "messageId": uuid4().hex,  # Unique message ID for tracking
            **({"taskId": task_id} if task_id else {}),  # Include taskId only if it's a follow-up
            **({"contextId": context_id} if context_id else {}),  # Include contextId for continuity
        }
    }

# -----------------------------------------------------------------------------
# Helper: Pretty print JSON objects using syntax coloring
# -----------------------------------------------------------------------------
def print_json_response(response: Any, title: str) -> None:
    # Displays a formatted and color-highlighted view of the response
    print(f"\n=== {title} ===")  # Section title for clarity
    try:
        if hasattr(response, "root"):  # Check if response is wrapped by SDK
            data = response.root.model_dump(mode="json", exclude_none=True)
        else:
            data = response.model_dump(mode="json", exclude_none=True)

        json_str = json.dumps(data, indent=2, ensure_ascii=False)  # Convert dict to pretty JSON string
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)  # Apply syntax highlighting
        rprint(syntax)  # Print it with color
    except Exception as e:
        # Print fallback text if something fails
        rprint(f"[red bold]Error printing JSON:[/red bold] {e}")
        rprint(repr(response))

# -----------------------------------------------------------------------------
# Handles sending one non-streaming message and optionally a follow-up
# -----------------------------------------------------------------------------
async def handle_non_streaming(client: A2AClient, text: str):
    # Build and send the first message
    request = SendMessageRequest(params=MessageSendParams(**build_message_payload(text)))
    result = await client.send_message(request)  # Wait for agent reply
    print_json_response(result, "Agent Reply")  # Print the reply

    # If agent needs more input, prompt user again
    if isinstance(result.root, SendMessageSuccessResponse):
        task = result.root.result  # Extract task
        if task.status.state == TaskState.input_required:
            follow_up = input("\U0001F7E1 Agent needs more input. Your reply: ")
            follow_up_req = SendMessageRequest(
                params=MessageSendParams(**build_message_payload(follow_up, task.id, task.contextId))
            )
            follow_up_resp = await client.send_message(follow_up_req)
            print_json_response(follow_up_resp, "Follow-up Response")

# -----------------------------------------------------------------------------
# Handles streaming message and recursively continues if more input is needed
# -----------------------------------------------------------------------------
async def handle_streaming(client: A2AClient, text: str, task_id: str | None = None, context_id: str | None = None):
    # Construct streaming request payload
    request = SendStreamingMessageRequest(params=MessageSendParams(**build_message_payload(text, task_id, context_id)))

    # Track latest task/context ID to support multi-turn
    latest_task_id = None
    latest_context_id = None
    input_required = False

    # Process each streamed update
    async for update in client.send_message_streaming(request):
        print_json_response(update, "Streaming Update")  # Print each update as it comes

        # Extract context/task from current update
        if hasattr(update.root, "result"):
            result = update.root.result
            if hasattr(result, "contextId"):
                latest_context_id = result.contextId
            if hasattr(result, "status") and result.status.state == TaskState.input_required:
                latest_task_id = result.taskId
                input_required = True

    # If input was required, get response from user and continue conversation
    if input_required and latest_task_id and latest_context_id:
        follow_up = input("\U0001F7E1 Agent needs more input. Your reply: ")
        await handle_streaming(client, follow_up, latest_task_id, latest_context_id)

# -----------------------------------------------------------------------------
# Loop for querying the agent repeatedly
# -----------------------------------------------------------------------------
async def interactive_loop(client: A2AClient, supports_streaming: bool):
    print("\nEnter your query below. Type 'exit' to quit.")  # Print instructions for user
    while True:
        query = input("\n\U0001F7E2 Your query: ").strip()  # Get user input
        if query.lower() in {"exit", "quit"}:
            print("\U0001F44B Exiting...")  # Say goodbye
            break
        # Choose path based on agent's capability
        if supports_streaming:
            await handle_streaming(client, query)
        else:
            await handle_non_streaming(client, query)

# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
@click.command()
@click.option("--agent-url", default="http://localhost:10000", help="URL of the A2A agent to connect to")
def main(agent_url: str):
    asyncio.run(run_main(agent_url))  # Launch async event loop with provided agent URL

# -----------------------------------------------------------------------------
# Async runner: sets up client, agent card, and launches the loop
# -----------------------------------------------------------------------------
async def run_main(agent_url: str):
    print(f"Connecting to agent at {agent_url}...")  # Let user know we're starting connection
    try:
        async with httpx.AsyncClient() as session:  # Use async context to keep session open
            client = await A2AClient.get_client_from_agent_card_url(session, agent_url)  # Create A2A client
            client.httpx_client.timeout = 60  # Increase timeout for long operations

            res = await session.get(f"{agent_url}/.well-known/agent.json")  # Get agent metadata
            agent_card = AgentCard.model_validate(res.json())  # Validate the structure of the metadata
            supports_streaming = agent_card.capabilities.streaming  # Check if agent can stream

            rprint(f"[green bold]✅ Connected. Streaming supported:[/green bold] {supports_streaming}")  # Confirm success
            await interactive_loop(client, supports_streaming)  # Start conversation loop

    except Exception:
        traceback.print_exc()  # Show full error trace
        print("❌ Failed to connect or run. Ensure the agent is live and reachable.")  # Friendly error message

# -----------------------------------------------------------------------------
# Execute main only when run as script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()  # Run main CLI logic



