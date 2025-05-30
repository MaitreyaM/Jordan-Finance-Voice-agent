import asyncio
import os
from typing import Dict, Any, Optional
import requests 
from datetime import datetime
from dotenv import load_dotenv

from agno.agent import Agent as AgnoAgent, Message as AgnoMessage, RunResponse as AgnoRunResponse
from agno.models.groq import Groq 
from agno.tools.yfinance import YFinanceTools

def fetch_coingecko_btc_price_sync() -> Optional[Dict[str, Any]]:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {'ids': 'bitcoin', 'vs_currencies': 'usd,inr'}
    try:
        response = requests.get(url, params=params, timeout=15) 
        response.raise_for_status()
        data = response.json().get('bitcoin', {})
        print(f"[FinancialReportAgent - CoinGecko] Fetched: {data}")
        return data
    except Exception as e:
        print(f"[FinancialReportAgent - CoinGecko] Error fetching BTC price: {e}")
        return None

class FinancialReportAgentLogic: 
    _agno_agent_instance: AgnoAgent 

    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=dotenv_path)
        print("FinancialReportAgentLogic: .env loaded.")

        llm_model_id = os.getenv("AGNO_FINANCIAL_LLM_MODEL", "llama3-70b-8192") 
        print(f"FinancialReportAgentLogic: Initializing Agno model: Groq with id='{llm_model_id}'")
        try:
            agno_llm = Groq(id=llm_model_id) 
        except Exception as e:
            print(f"CRITICAL: Failed to initialize Agno Groq LLM: {e}")
            raise

        yfinance_tools = YFinanceTools(
            company_news=True, stock_price=True, company_info=True, analyst_recommendations=True
        )
        print("FinancialReportAgentLogic: YFinanceTools initialized.")

        self._agno_agent_instance = AgnoAgent( 
            model=agno_llm,
            tools=[yfinance_tools],
            markdown=True, 
            instructions=[
                "You are a financial analyst. Your goal is to create a concise 'Morning Market Brief'.",
                "First, you will be provided with the current Bitcoin price from an external source (CoinGecko).",
                "Then, use your YFinance tools to gather relevant general market news and specific stock information for major tech companies (e.g., Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Nvidia (NVDA), TSMC (TSM), Samsung (005930.KS)). Focus on recent news and earnings.",
                "Synthesize all this information (Bitcoin price, YFinance market news, specific stock news/earnings) into a brief, insightful report.",
                "Highlight any significant market movements, trends, and notable earnings surprises.",
                "Structure the report clearly. Start with a Bitcoin update, then general market sentiment, then specific company highlights.",
                "Keep the language professional and to the point. Ensure the final output is a single block of text for the report."
            ]
        )
        print("FinancialReportAgentLogic: Agno Agent initialized.")

    async def generate_report(self, user_query_context: Optional[str] = None) -> str:
        print("FinancialReportAgentLogic: Generating report...")
        btc_data = await asyncio.to_thread(fetch_coingecko_btc_price_sync)
        btc_report_line = "Bitcoin Price: Data unavailable from CoinGecko."
        if btc_data:
            usd_price = btc_data.get('usd', 'N/A')
            inr_price = btc_data.get('inr', 'N/A')
            btc_report_line = f"Current Bitcoin Price: ${usd_price} (approx. ₹{inr_price})."
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        input_content = f"Date: {current_date}\n{btc_report_line}\n\n"
        if user_query_context and user_query_context.lower() != "generate standard morning brief.":
            input_content += f"User focus for today's report: {user_query_context}\n\n"
        input_content += "Please generate the Morning Market Brief using your YFinance tools for additional market news and analysis, focusing on major tech stocks and any earnings surprises."
        
        print(f"FinancialReportAgentLogic: Input content for Agno agent (first 200 chars): {input_content[:200]}...")
        agno_message = AgnoMessage(role="user", content=input_content)
        
        try:
            response: AgnoRunResponse = await self._agno_agent_instance.arun(agno_message) # Use updated attribute
            report_text = response.content if response and response.content else "No textual report generated by Agno agent."
            print(f"FinancialReportAgentLogic: Agno agent raw response content (first 200 chars): {report_text[:200]}...")
            return report_text.strip()
        except Exception as e:
            print(f"FinancialReportAgentLogic: Error running Agno agent: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during Agno agent execution: {str(e)}"

class AgnoFinancialA2AExecutorFastAPI: 
    _financial_report_logic: FinancialReportAgentLogic 

    def __init__(self):
        print("AgnoFinancialA2AExecutorFastAPI: Initializing...")
        self._financial_report_logic = FinancialReportAgentLogic()
        print("AgnoFinancialA2AExecutorFastAPI: FinancialReportAgentLogic instantiated.")

    async def generate_brief_directly(self, user_input: str) -> str:
        """
        Called by the FastAPI endpoint. Directly invokes the Agno agent logic.
        """
        print(f"AgnoFinancialA2AExecutorFastAPI: Received direct call. User input: '{user_input}'")
        try:
            report_str = await self._financial_report_logic.generate_report(user_query_context=user_input)
            print(f"AgnoFinancialA2AExecutorFastAPI: Report generated: '{report_str[:200]}...'")
            return report_str
        except Exception as e:
            print(f"AgnoFinancialA2AExecutorFastAPI: Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing your financial brief request: {str(e)}"

    async def close_resources(self):
        print("AgnoFinancialA2AExecutorFastAPI: Closing resources...")
        if hasattr(self._financial_report_logic._agno_agent_instance.model, 'close') and \
           asyncio.iscoroutinefunction(self._financial_report_logic._agno_agent_instance.model.close):
            try:
                await self._financial_report_logic._agno_agent_instance.model.close()
                print("AgnoFinancialA2AExecutorFastAPI: Agno LLM model client closed.")
            except Exception as e:
                print(f"AgnoFinancialA2AExecutorFastAPI: Error closing Agno LLM model client: {e}")
        print("AgnoFinancialA2AExecutorFastAPI: Resource cleanup finished.")