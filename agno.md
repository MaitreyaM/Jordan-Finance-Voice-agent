What is Agno?
Agno is a lightweight, high-performance library for building Agents.

It helps you progressively build the 5 levels of Agentic Systems:

Level 1: Agents with tools and instructions.
Level 2: Agents with knowledge and storage.
Level 3: Agents with memory and reasoning.
Level 4: Teams of Agents with collaboration and coordination.
Level 5: Agentic Workflows with state and determinism.
Here's a Investment Research Agent that analyzes stocks, reasoning through each step:

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Claude(id="claude-3-7-sonnet-latest"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True),
    ],
    instructions=[
        "Use tables to display data",
        "Only output the report, no other text",
    ],
    markdown=True,
)
agent.print_response("Write a report on NVDA", stream=True, show_full_reasoning=True, stream_intermediate_steps=True)
 reasoning_finance_agent.mp4 
Key features
Agno is simple, fast and model-agnostic. Here are some key features:

Model Agnostic: Agno Agents can connect to 23+ model providers, no lock-in.
Lightning Fast: - Lightning Fast: Agents instantiate in ~3μs and use ~5Kib memory on average (see performance for more details).
Reasoning is a first class citizen: Make your Agents "think" and "analyze" using Reasoning Models, ReasoningTools or our custom chain-of-thought approach.
Natively Multi Modal: Agno Agents are natively multi modal, they can take in text, image, audio and video and generate text, image, audio and video as output.
Advanced Multi Agent Architecture: Agno provides an industry leading multi-agent architecture (Agent Teams) with 3 different modes: route, collaborate and coordinate.
Agentic Search built-in: Give your Agents the ability to search for information at runtime using one of 20+ vector databases. Get access to state-of-the-art Agentic RAG that uses hybrid search with re-ranking. Fully async and highly performant.
Long-term Memory & Session Storage: Agno provides plug-n-play Storage & Memory drivers that give your Agents long-term memory and session storage.
Pre-built FastAPI Routes: Agno provides pre-built FastAPI routes to serve your Agents, Teams and Workflows.
Structured Outputs: Agno Agents can return fully-typed responses using model provided structured outputs or json_mode.
Monitoring: Monitor agent sessions and performance in real-time on agno.com.
Building Agents with Agno
If you're new to Agno, start by building your first Agent, chat with it on the playground and finally, monitor it on agno.com.

After that, checkout the Examples Gallery and build real-world applications with Agno.

Installation
pip install -U agno
What are Agents?
Agents are AI programs that operate autonomously.

The core of an Agent is a model, tools and instructions.
Agents also have memory, knowledge, storage and the ability to reason.
Read more about each of these in the docs.

Let's build a few Agents to see how they work.

Example - Reasoning Agent
Let's start with a Reasoning Agent so we get a sense of Agno's capabilities.

Save this code to a file: reasoning_agent.py.

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=Claude(id="claude-3-7-sonnet-latest"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        ),
    ],
    instructions=[
        "Use tables to display data",
        "Only output the report, no other text",
    ],
    markdown=True,
)
agent.print_response(
    "Write a report on NVDA",
    stream=True,
    show_full_reasoning=True,
    stream_intermediate_steps=True,
)
Then create a virtual environment, install dependencies, export your ANTHROPIC_API_KEY and run the agent.

uv venv --python 3.12
source .venv/bin/activate

uv pip install agno anthropic yfinance

export ANTHROPIC_API_KEY=sk-ant-api03-xxxx

python reasoning_agent.py
We can see the Agent is reasoning through the task, using the ReasoningTools and YFinanceTools to gather information. This is how the output looks like:

 reasoning_finance_agent.mp4 
Now let's walk through the simple -> tools -> knowledge -> teams of agents flow.

Example - Basic Agent
The simplest Agent is just an inference task, no tools, no memory, no knowledge.

from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    markdown=True
)
agent.print_response("Tell me about a breaking news story from New York.", stream=True)
To run the agent, install dependencies and export your OPENAI_API_KEY.

pip install agno openai

export OPENAI_API_KEY=sk-xxxx

python basic_agent.py
View this example in the cookbook

Example - Agent with tools
This basic agent will obviously make up a story, lets give it a tool to search the web.

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
agent.print_response("Tell me about a breaking news story from New York.", stream=True)
Install dependencies and run the Agent:

pip install duckduckgo-search

python agent_with_tools.py
Now you should see a much more relevant result.

View this example in the cookbook

Example - Agent with knowledge
Agents can store knowledge in a vector database and use it for RAG or dynamic few-shot learning.

Agno agents use Agentic RAG by default, which means they will search their knowledge base for the specific information they need to achieve their task.

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
agent.print_response("What is the history of Thai curry?", stream=True)
Install dependencies and run the Agent:

pip install lancedb tantivy pypdf duckduckgo-search

python agent_with_knowledge.py
View this example in the cookbook

Example - Multi Agent Teams
Agents work best when they have a singular purpose, a narrow scope and a small number of tools. When the number of tools grows beyond what the language model can handle or the tools belong to different categories, use a team of agents to spread the load.

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.team import Team

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Team(
    mode="coordinate",
    members=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    success_criteria="A comprehensive financial news report with clear sections and data-driven insights.",
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("What's the market outlook and financial performance of AI semiconductor companies?", stream=True)
Install dependencies and run the Agent team:

pip install duckduckgo-search yfinance

python agent_team.py


## AGNO A2A EXAMPLE

Directory structure:
└── a2a/
    ├── __init__.py
    └── basic_agent/
        ├── README.md
        ├── __init__.py
        ├── __main__.py
        ├── basic_agent.py
        └── test_client.py


Files Content:

================================================
FILE: cookbook/examples/a2a/__init__.py
================================================



================================================
FILE: cookbook/examples/a2a/basic_agent/README.md
================================================
# Basic Agno A2A Agent Example

Basic Agno A2A Agent example that uses A2A to send and receive messages to/from an agent.

## Getting started

1. Clone a2a python repository: https://github.com/google/a2a-python

2. Install the a2a library in your virtual environment which has Agno installed

   ```bash
   pip install .
   ```

3. Start the server

   ```bash
   python cookbook/examples/a2a/basic_agent
   ```

4. Run the test client in a different terminal

   ```bash
   python cookbook/examples/a2a/basic_agent/test_client.py
   ```

## Notes

- The test client sends a message to the server and prints the response.
- The server uses the `BasicAgentExecutor` to execute the message and send the response back to the client.
- Streaming is not yet functional.
- The server runs on `http://localhost:9999` by default.



================================================
FILE: cookbook/examples/a2a/basic_agent/__init__.py
================================================



================================================
FILE: cookbook/examples/a2a/basic_agent/__main__.py
================================================
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from basic_agent import BasicAgentExecutor

if __name__ == "__main__":
    skill = AgentSkill(
        id="agno_agent",
        name="Agno Agent",
        description="Agno Agent",
        tags=["Agno agent"],
        examples=["hi", "hello"],
    )

    agent_card = AgentCard(
        name="Agno Agent",
        description="Agno Agent",
        url="http://localhost:9999/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
        authentication=AgentAuthentication(schemes=["public"]),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=BasicAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    import uvicorn

    uvicorn.run(server.build(), host="0.0.0.0", port=9999, timeout_keep_alive=10)



================================================
FILE: cookbook/examples/a2a/basic_agent/basic_agent.py
================================================
from typing import List

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Part, TextPart
from a2a.utils import new_agent_text_message
from agno.agent import Agent, Message, RunResponse
from agno.models.openai import OpenAIChat
from typing_extensions import override

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)


class BasicAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(self):
        self.agent = agent

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message: Message = Message(role="user", content="")
        for part in context.message.parts:
            if isinstance(part, Part):
                if isinstance(part.root, TextPart):
                    message.content = part.root.text
                    break

        result: RunResponse = await self.agent.arun(message)
        event_queue.enqueue_event(new_agent_text_message(result.content))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("Cancel not supported")



================================================
FILE: cookbook/examples/a2a/basic_agent/test_client.py
================================================
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,  # noqa: F401
)


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        client = await A2AClient.get_client_from_agent_card_url(
            httpx_client, "http://localhost:9999"
        )
        send_message_payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": "Hello! What can you tell me about the weather in Tokyo?",
                    }
                ],
                "messageId": uuid4().hex,
            },
        }
        request = SendMessageRequest(params=MessageSendParams(**send_message_payload))

        response = await client.send_message(request)
        print(response.model_dump(mode="json", exclude_none=True))

        # streaming_request = SendStreamingMessageRequest(
        #     params=MessageSendParams(**send_message_payload)
        # )

        # stream_response = client.send_message_streaming(streaming_request)
        # async for chunk in stream_response:
        #     print(chunk.model_dump(mode='json', exclude_none=True))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


