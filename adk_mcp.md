A guide to Google ADK and MCP integration with an external server
May 15, 2025
Wei Yih Yap
Generative AI Field Solutions Architect

Alan Blount
Product Manager

Try Gemini 2.5
Our most intelligent model is now available on Vertex AI

Try now
For AI-powered agents to perform useful, real-world tasks, they need to reliably access tools and up-to-the-minute information that lives outside the base model. Anthropic’s Model Context Protocol (MCP) is designed to address this, providing a standardized way for agents to retrieve that crucial, external context needed to inform their responses and actions.

This is vital for developers who need to build and deploy sophisticated agents that can leverage enterprise data or public tools. But integrating agents built with Google's Agent Development Kit (ADK) to communicate effectively with an MCP server, especially one hosted externally, might present some integration challenges.

Today, we’ll guide you through developing ADK agents that connect to external MCP servers, initially using Server-Sent Events (SSE). We’ll take an example of an ADK agent leveraging MCP to access Wikipedia articles, which is a common use case to retrieve external specialised data. We will also introduce Streamable HTTP, the next-generation transport protocol designed to succeed SSE for MCP communications.

A quick refresher
Before we start, let's make sure we all understand the following terms:

SSE enables servers to push data to clients over a persistent HTTP connection. In a typical setup for MCP, this involved using two distinct endpoints: one for the client to send requests to the server (usually via HTTP POST) and a separate endpoint where the client would establish an SSE connection (HTTP GET) to receive streaming responses and server-initiated messages.

MCP is an open standard designed to standardize how Large Language Models (LLMs) interact with external data sources, APIs and resources as agent tools, MCP aims to replace the current landscape of fragmented, custom integrations with a universal, standardized framework.

Streamable HTTP utilizes a single HTTP endpoint for both sending requests from the client to the server, and receiving responses and notifications from the server to the client.

$300 in free credit to try Google Cloud developer tools
Build and test your proof of concept with $300 in free credit for new customers. Plus, all customers get free monthly usage of 20+ products, including AI APIs.

Start building for free
Step 1: Create an MCP server 
You need the following python packages installed in your virtual environment before proceeding. We will be using the uv tool in this blog.

lang-py
"beautifulsoup4==4.12.3",
"google-adk==0.3.0",
"html2text==2024.2.26",
"mcp[cli]==1.5.0",
"requests==2.32.3"
Here’s an explanation of the Python code server.py:

It creates an instance of an MCP server using FastMCP

It defines a tool called extract_wikipedia_article decorated with @mcp.tool

It configures an SSE transport mechanism SseServerTransport to enable real-time communication, typically for the MCP server interactions.

It creates a web application instance using the Starlette framework and defines two routes, message and sse.

You can read more about SSE transport protocol here.

lang-py
# File server.py
​
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from html2text import html2text
​
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount
​
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport
​
# Create an MCP server instance with an identifier ("wiki")
mcp = FastMCP("wiki")
​
@mcp.tool()
def extract_wikipedia_article(url: str) -> str:
    """
    Retrieves and processes a Wikipedia article from the given URL, extracting
    the main content and converting it to Markdown format.
​
    Usage:
        extract_wikipedia_article("https://en.wikipedia.org/wiki/Gemini_(chatbot)")
    """
    try:
        if not url.startswith("http"):
            raise ValueError("URL must begin with http or https protocol.")
​
        response = requests.get(url, timeout=8)
        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Unable to access the article. Server returned status: {response.status_code}"
                )
            )
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="The main article content section was not found at the specified Wikipedia URL."
                )
            )
        markdown_text = html2text(str(content_div))
        return markdown_text
​
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"An unexpected error occurred: {str(e)}")) from e
​
sse = SseServerTransport("/messages/")
​
async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())
​
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)
​
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
To start the server, you can run the command uv run server.py.

Bonus tip, to debug the server using MCP Inspector, execute the command uv run mcp dev server.py.

https://storage.googleapis.com/gweb-cloudblog-publish/images/01-mcp-inspector.max-1800x1800.png
Step 2: Attach the MCP server while creating ADK agents
The following explains the Python code in the file agent.py:

Uses MCPToolset.from_server with SseServerParams to establish a SSE connection to a URI endpoint. For this demo we will use http://localhost:8001/sse, but in production this would be a remote server.

Create an ADK Agent and call get_tools_async to get the tools from the MCP server.

lang-py
# File agent.py
​
import asyncio
import json
from typing import Any
​
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import (
    InMemoryArtifactService,  # Optional
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    SseServerParams,
)
from google.genai import types
from rich import print
load_dotenv()
​
async def get_tools_async():
    """Gets tools from the File System MCP Server."""
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=SseServerParams(
            url="http://localhost:8001/sse",
        )
    )
    print("MCP Toolset created successfully.")
    return tools, exit_stack
​
async def get_agent_async():
    """Creates an ADK Agent equipped with tools from the MCP Server."""
    tools, exit_stack = await get_tools_async()
    print(f"Fetched {len(tools)} tools from MCP server.")
    root_agent = LlmAgent(
        model="gemini-2.0-flash",
        name="assistant",
        instruction="""Help user extract and summarize the article from wikipedia link.
        Use the following tools to extract wikipedia article:
        - extract_wikipedia_article
​
        Once you retrieve the article, always summarize it in a few sentences for the user.
        """,
        tools=tools,
    )
    return root_agent, exit_stack
​
root_agent = get_agent_async()
Step 3: Test your agent
We will use the ADK developer tool to test the agent.

Create the following directory structure:

lang-py
. # <--Your current directory
├── adk-agent
│   ├── __init__.py
│   └── agent.py
├── .env
The content for __init__.py and .env are as follows:

lang-py
# .env
GOOGLE_GENAI_USE_VERTEXAI="True"
GOOGLE_CLOUD_PROJECT=<YOUR_PROJECT_ID>
GOOGLE_CLOUD_LOCATION="us-central1"
lang-py
# __init__.py
from . import agent
Start the UI with the following command:

lang-py
uv run adk web
This will open up the ADK developer tool interface as shown below:

https://storage.googleapis.com/gweb-cloudblog-publish/images/02-adk-web.max-1800x1800.png
Streamable HTTP
It is worth noting that in March 2025, MCP released a new transport protocol called Streamable HTTP. The Streamable HTTP transport allows a server to function as an independent process managing multiple client connections via HTTP POST and GET requests. Servers can optionally implement Server-Sent Events (SSE) for streaming multiple messages, enabling support for basic MCP servers as well as more advanced servers with streaming and server-initiated communication.

The following code demonstrates how to implement a Streamable HTTP server, where the tool extract_wikipedia_article will return a dummy string to simplify the code.

lang-py
# File server.py
​
import contextlib
import logging
from collections.abc import AsyncIterator
​
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send
import uvicorn
​
logger = logging.getLogger(__name__)
​
​
app = Server("mcp-streamable-http-stateless-demo")
​
​
@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    # Check if the tool is extract-wikipedia-article
    if name == "extract-wikipedia-article":
        # Return dummy content for the Wikipedia article
        return [
            types.TextContent(
                type="text",
                text="This is the article ...",
            )
        ]
​
    # For other tools, keep the existing notification logic
    ctx = app.request_context
    interval = arguments.get("interval", 1.0)
    count = arguments.get("count", 5)
    caller = arguments.get("caller", "unknown")
​
    # Send the specified number of notifications with the given interval
    for i in range(count):
        await ctx.session.send_log_message(
            level="info",
            data=f"Notification {i + 1}/{count} from caller: {caller}",
            logger="notification_stream",
            related_request_id=ctx.request_id,
        )
        if i < count - 1:  # Don't wait after the last notification
            await anyio.sleep(interval)
​
    return [
        types.TextContent(
            type="text",
            text=(
                f"Sent {count} notifications with {interval}s interval"
                f" for caller: {caller}"
            ),
        )
    ]
​
​
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="extract-wikipedia-article",
            description=("Extracts the main content of a Wikipedia article"),
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the Wikipedia article to extract",
                    },
                },
            },
        )
    ]
​
​
session_manager = StreamableHTTPSessionManager(
    app=app,
    event_store=None,
    stateless=True,
)
​
​
async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)
​
​
@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Context manager for session manager."""
    async with session_manager.run():
        logger.info("Application started with StreamableHTTP session manager!")
        try:
            yield
        finally:
            logger.info("Application shutting down...")
​
​
app = Starlette(
    debug=True,
    routes=[
        Mount("/mcp", app=handle_streamable_http),
    ],
    lifespan=lifespan,
)
​
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
You can start the Streamable HTTP MCP server using by running the following:

lang-py
# Start the server
uv run server.py
To debug with MCP Inspector, select Streamable HTTP and fill in the MCP Server URL http://localhost:3000/mcp.

https://storage.googleapis.com/gweb-cloudblog-publish/images/03-streamable-http.max-2100x2100.png
Authentication
For production deployments of MCP servers, robust authentication is a critical security consideration. As this field is under active development at the time of writing, we recommend referring to the MCP specification on Authentication for more information.

For an enterprise grade API governance system which, similar to MCP, can generate agent tools:

Apigee centralizes and manages any APIs, with full control, versioning, and governance 

API Hub organizes metadata for any API and documentation 

Application Integrations support many existing API connections with user access control support 

ADK supports these Google Cloud Managed Tools with about the same number of lines of code

Get started 
To get started today, read the documentation for ADK. You can create your own Agent with