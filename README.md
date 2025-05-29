# Jordon: Advanced Voice-Controlled Financial and Productivity Assistant

Jordon is a sophisticated, voice-controlled AI assistant built to demonstrate the integration of multiple cutting-edge AI technologies and agentic frameworks. It serves as a central hub for managing personal productivity (Calendar, Gmail), accessing financial market information (briefs, RAG-based explanations), and interacting with a Zerodha trading account.

This project showcases the orchestration of Google's Agent Development Kit (ADK), the custom-built CLAP multi-agent Python framework, the Agno agent framework, and various Model Context Protocol (MCP) integrations.


<p align="center">
  <video width="800" controls>
    <source src="demo.mp4" type="video/mp4">
    Your browser does not support HTML5 video.
  </video>
</p>


## Key Features

*   **Voice-Controlled Interface:** Interactive and responsive UI for natural language interaction (text and speech).
*   **Comprehensive Productivity Suite:**
    *   **Google Calendar Management:** View, create, edit, and delete calendar events.
    *   **Gmail Integration:** Send emails and fetch recent email summaries.
*   **Advanced Financial Capabilities:**
    *   **Zerodha Trading Account Interaction:** Securely log in, view holdings, check positions, get margins, and place orders.
    *   **Financial Market Briefs:** Receive synthesized daily market briefs covering Bitcoin (via CoinGecko), major tech stocks, and market sentiment (powered by an Agno agent team using YFinance).
    *   **Financial Knowledge Base (RAG):** Ask for explanations of financial terms and concepts, with answers retrieved from a curated document base (`financial_glossary.pdf`) using the CLAP RAG agent.
*   **Multi-Agent System:**
    *   **Orchestrator:** Jordon (Google ADK agent) acts as the central orchestrator.
    *   **Specialized Agents:**
        *   CLAP RAG Agent: For deep knowledge retrieval.
        *   Agno Agent Team: For dynamic financial brief synthesis.
    *   **MCP Services:** Dedicated local MCP servers for Gmail and Zerodha, demonstrating real-world tool integration.
*   **Modular & Extensible Architecture:** Designed with clear separation of concerns for each component.

## Architecture Overview

Jordon operates as a distributed system of interconnected services:

1.  **Jordon ADK Voice Agent (FastAPI + ADK + Gemini LLM):**
    *   The primary user interface and main orchestrator.
    *   Handles voice input/output via WebSockets.
    *   Uses local Python functions for Google Calendar.
    *   Connects to Gmail and Zerodha MCP servers via ADK `MCPToolset` (SSE).
    *   Delegates complex queries to CLAP and Agno agents via HTTP calls.

2.  **Local Gmail MCP Server (Starlette + FastMCP + SSE):**
    *   Exposes Gmail functionalities (`send_email_tool`, `fetch_recent_emails`) over MCP.

3.  **Local Zerodha MCP Server (Starlette + FastMCP + SSE):** 
    *   A modified version of the `aptro/zerodha-mcp` server, adapted to run locally via SSE.
    *   Exposes Zerodha trading tools (`check_and_authenticate`, `get_holdings`, etc.).
    *   Includes a local FastAPI instance on port 5000 to handle the Kite Connect OAuth redirect.
    *   PLEASE CLONE THIS ZERODHA MCP IN THE MCP FOLDER TO RUN LOCALLY : https://github.com/aptro/zerodha-mcp

4.  **Agno Financial Brief Agent Server (FastAPI + Agno Team + Groq/Gemini LLMs):**
    *   Hosts an Agno `Team` of agents.
    *   One member agent uses `YFinanceTools` for stock news/prices.
    *   Another member agent uses a custom tool for CoinGecko crypto prices.
    *   A coordinator agent synthesizes this data into market briefs.
    *   Accessed by Jordon via an HTTP "A2A-like" call.

5.  **CLAP RAG Agent Server (FastAPI + CLAP Agent + ChromaDB + Gemini LLM):**
    *   Hosts a CLAP `Agent` for Retrieval Augmented Generation.
    *   Uses ChromaDB and SentenceTransformer embeddings for a knowledge base built from `financial_glossary.pdf`.
    *   Includes a `duckduckgo_search` tool as a fallback.
    *   Accessed by Jordon via an HTTP "A2A-like" call.

**Simplified Data Flow:**

Use code with caution.

Markdown

[User (Voice/Web UI)] <--(WebSocket)--> [Jordon ADK Voice Agent (FastAPI @ Port 8000)]

| |

| +-- (Local Python Call) --> [Google Calendar API]

| |

| +-- (MCP/SSE) --> [Local Gmail MCP Server (Starlette @ Port 8001)]

| |

| +-- (MCP/SSE) --> [Local Zerodha MCP Server (Starlette @ Port 8002)]

| | (Auth FastAPI @ Port 5000)

| |

| +-- (HTTP POST) --> [Agno Financial Brief Server (FastAPI @ Port 10000)]

| |

+--------------------------------------(HTTP POST) --> [CLAP RAG Q&A Server (FastAPI @ Port 9999)]

## Technology Stack

*   **Core Agent Framework (Orchestrator):** Google Agent Development Kit (ADK v0.5.0)
*   **Voice Agent Web Server:** FastAPI, Uvicorn, WebSockets
*   **Specialized Agent Frameworks:**
    *   CLAP (Custom Python Multi-Agent Framework)
    *   Agno
*   **LLMs:** Google Gemini Flash, Groq Llama3 (8B & 70B)
*   **Model Context Protocol (MCP):** `mcp` library, `FastMCP` for server implementation, SSE transport.
*   **Retrieval Augmented Generation (RAG):**
    *   Vector Database: ChromaDB
    *   Embedding Model: SentenceTransformers (default `all-MiniLM-L6-v2`)
    *   Document Loaders: PyPDF, custom text processing.
*   **Financial Data Tools:**
    *   Kite Connect API (via local Zerodha MCP)
    *   YFinance (via Agno `YFinanceTools`)
    *   CoinGecko API (via custom Agno tool)
*   **Other Tools:** Google Calendar API, Gmail API (via MCP), DuckDuckGo Search.
*   **Programming Language:** Python 3.10
*   **UI:** HTML, CSS, JavaScript (for the ADK voice agent's frontend).

## Prerequisites

*   Python 3.10
*   Conda (recommended for environment management)
*   **API Keys & Credentials:**
    *   Google API Key (for Gemini LLM used by ADK Jordon & CLAP RAG Agent) - store in relevant `.env` files.
    *   Groq API Key (for Groq LLMs used by Agno Agent) - store in Agno agent's `.env` file.
    *   Zerodha Kite Connect API Key & API Secret - store in the local Zerodha MCP server's `.env` file.
    *   Gmail SMTP Username & App Password (for Gmail MCP server) - store in Gmail MCP's `.env` file.
*   **Google Cloud Project:** With Calendar API and Speech-to-Text API enabled.
*   **`credentials.json` for Google Calendar:** Obtained from Google Cloud Console for OAuth.
*   **Zerodha Developer App:** Configured with the correct Redirect URI (`http://127.0.0.1:5000/zerodha/auth/redirect`).
*   **RAG Document:** `financial_glossary.pdf` placed in the `clap_a2a_integration/` directory.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone YOUR_GITHUB_REPOSITORY_URL
    cd YOUR_PROJECT_DIRECTORY
    ```

2.  **Create Conda Environment (Recommended):**
    ```bash
    conda create -n holbox python=3.10 -y
    conda activate holbox
    ```

3.  **Install Dependencies:**
    *   **ADK Voice Agent (Jordon):**
        ```bash
        cd ADK/adk-voice-agent/
        pip install -r requirements.txt
        cd ../.. 
        ```
    *   **Gmail MCP Server:** (Dependencies likely covered by ADK's `requirements.txt` or install `mcp fastapi uvicorn python-dotenv starlette requests`)
        Located in `mcps/gmail_mcp.py`.
    *   **Local Zerodha MCP Server:**
        Navigate to your cloned `aptro/zerodha-mcp` directory.
        ```bash
        # Ensure you are in the aptro/zerodha-mcp directory
        pip install -r requirements.txt # Or uv pip install kiteconnect fastapi uvicorn python-dotenv httpx "mcp>=1.3.0"
        ```
    *   **Agno Financial Brief Agent Server:**
        Navigate to `ADK/adk-voice-agent/agno_finance_agent/` (or wherever it's located in your final structure).
        ```bash
        pip install -r requirements.txt # Ensure Agno, Groq SDK, Google Gemini SDK, httpx, requests are listed
        ```
    *   **CLAP RAG Agent Server:**
        Navigate to `clap_a2a_integration/`.
        ```bash
        pip install -r requirements.txt # Ensure CLAP, ChromaDB, SentenceTransformers, Google Gemini SDK are listed
        ```
    *(Note: Consolidate requirements into top-level files if structure changes)*

4.  **Configure Environment Variables (`.env` files):**
    *   **`ADK/adk-voice-agent/.env`:**
        *   `GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY`
        *   `CLAP_A2A_SERVER_URL="http://localhost:9999"`
        *   `AGNO_A2A_FINANCIAL_SERVER_URL="http://localhost:10000"`
    *   **`mcps/gmail_mcp_config/.env` (or wherever `gmail_mcp.py` loads from):**
        *   `SMTP_USERNAME=your_gmail_address`
        *   `SMTP_PASSWORD=your_gmail_app_password`
    *   **`local_zerodha_mcp_directory/.env` (root of your `aptro/zerodha-mcp` clone):**
        *   `KITE_API_KEY=YOUR_ZERODHA_KITE_API_KEY`
        *   `KITE_API_SECRET=YOUR_ZERODHA_KITE_API_SECRET`
    *   **`ADK/adk-voice-agent/agno_finance_agent/.env`:**
        *   `GROQ_API_KEY=YOUR_GROQ_API_KEY`
        *   `GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY` (if Gemini is used by Agno coordinator)
        *   `AGNO_DATA_FETCHER_MODEL_GROQ=llama3-8b-8192` (or your preferred model)
        *   `AGNO_COORDINATOR_MODEL_GEMINI=gemini-1.5-flash-latest` (or your preferred model)
    *   **`clap_a2a_integration/.env`:**
        *   `GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY` (for CLAP's LLM)

5.  **Google Calendar API Setup:**
    *   Place your downloaded `credentials.json` in the `ADK/adk-voice-agent/` directory.
    *   Run the setup script from within that directory:
        ```bash
        cd ADK/adk-voice-agent/
        python setup_calendar_auth.py 
        cd ../..
        ```
    *   Follow the browser authentication flow. This will store `calendar_token.json` in `~/.credentials/`.

6.  **Zerodha API App:**
    *   Ensure your app in the Kite Developer console has the **Redirect URI** set to `http://127.0.0.1:5000/zerodha/auth/redirect`.

7.  **CLAP RAG Document:**
    *   Ensure `financial_glossary.pdf` (or your chosen PDF) is in the `clap_a2a_integration/` directory. The ChromaDB will be built on the first run of the CLAP server if it doesn't exist.

## Running the System

Each server component needs to be run in a separate terminal. Ensure your `holbox` conda environment is activated for each.

1.  **Gmail MCP Server:**
    ```bash
    python mcps/gmail_mcp.py 
    ```
    *(Runs on `http://localhost:8001`)*

2.  **Local Zerodha MCP Server (SSE Version):**
    *   Navigate to your local `aptro/zerodha-mcp` directory (the one modified to run as SSE).
    ```bash
    python main.py 
    ```
    *(Runs MCP SSE on `http://localhost:8002`, Auth FastAPI on `http://localhost:5000`)*

3.  **CLAP RAG Agent Server:**
    *   Navigate to `clap_a2a_integration/`.
    ```bash
    python run_clap_a2a_server.py 
    ```
    *(Runs on `http://localhost:9999`)*

4.  **Agno Financial Brief Agent Server:**
    *   Navigate to `ADK/adk-voice-agent/agno_finance_agent/` (adjust path if different).
    ```bash
    python run_financial_a2a_server.py 
    ```
    *(Runs on `http://localhost:10000`)*

5.  **Jordon ADK Voice Agent (Main Application):**
    *   Navigate to `ADK/adk-voice-agent/app/`.
    ```bash
    uvicorn main:app --reload
    ```
    *(Runs on `http://localhost:8000`)*

Access Jordon by opening `http://localhost:8000` in your browser.

## How to Use / Example Interactions

*   **Voice or Text Input:** Interact via the web UI.
*   **Calendar:** "What's on my calendar for tomorrow?", "Create an event for a meeting on Friday at 2 PM called Project Sync."
*   **Gmail:** "Send an email to example@example.com with subject Hello and body Just checking in.", "What are my latest emails?"
*   **Zerodha:**
    *   "I want to login to my Zerodha account." (Follow browser prompts)
    *   "What are my Zerodha holdings?"
    *   "Show my open positions in Zerodha."
*   **Financial Brief:** "What's the market brief for today?", "Give me a financial report focusing on AI stocks."
*   **CLAP RAG:** "What is a P/E ratio?", "Explain market capitalization."

## Project Structure (Key Directories)

*   `ADK/adk-voice-agent/app/`: Contains the main Jordon ADK agent, its FastAPI server, and static UI files.
    *   `jarvis/`: Jordon's agent logic and tools.
*   `mcps/`: Contains the standalone MCP server implementations (Gmail, local Zerodha).
*   `clap_a2a_integration/`: Contains the CLAP RAG agent and its FastAPI server.
*   `ADK/adk-voice-agent/agno_finance_agent/`: Contains the Agno Financial Brief agent team and its FastAPI server.

## Future Enhancements / Considerations

*   Integrate a live Alpha Vantage SSE MCP if a stable public one is found or a local one is built.
*   More sophisticated error handling and response generation in Jordon.
*   Expand the financial glossary for the CLAP RAG agent.
*   Refine prompts for all LLMs for better accuracy and conciseness.
*   Containerize services for easier deployment (e.g., using Docker).
