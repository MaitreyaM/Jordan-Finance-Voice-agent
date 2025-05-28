import asyncio
import os
import shutil
import time 
from dotenv import load_dotenv

# RAG
from clap import Agent 
from clap.vector_stores.chroma_store import ChromaStore
from clap.utils.rag_utils import (
    load_pdf_file,
    chunk_text_by_fixed_size
)

from clap.llm_services.google_openai_compat_service import GoogleOpenAICompatService 
from clap.llm_services.groq_service import GroqService
#from clap.llm_services.ollama_service import OllamaService


try:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    DEFAULT_EF = SentenceTransformerEmbeddingFunction() 
except ImportError:
    print("ERROR: sentence-transformers library not found. ChromaDB default EF will be used, or RAG may fail.")
    print("Please install with: pip install sentence-transformers 'clap-agents[chromadb]'")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    DEFAULT_EF = DefaultEmbeddingFunction()

class ClapAgentA2AExecutorFastAPIStyle:
    _rag_agent: Agent 
    _llm_service_for_rag: GoogleOpenAICompatService 
    _vector_store: ChromaStore
    _initialized: bool = False

    PDF_FILENAME = "holbox.pdf" 
    PDF_PATH = os.path.join(os.path.dirname(__file__), PDF_FILENAME)
    CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "./interactive_rag_chroma_db")
    COLLECTION_NAME = "ml_book_interactive_rag_a2a" # Unique name for this A2A server's DB
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    RAG_LLM_MODEL = "gemini-2.5-flash-preview-04-17" # Model for the RAG agent

    def __init__(self):
        # Basic init. Actual RAG setup will be in an async method.
        print("ClapAgentA2AExecutorFastAPIStyle: Synchronous __init__ called.")
        # .env loading for API keys is now handled by the server's lifespan method
        # before this executor is fully set up.

    async def setup_rag_agent(self):
        """
        Asynchronously initializes the RAG system (vector store, LLM, RAG agent).
        This should be called once when the A2A server starts.
        """
        if self._initialized:
            print("ClapAgentA2AExecutorFastAPIStyle: RAG agent already initialized.")
            return

        print(f"ClapAgentA2AExecutorFastAPIStyle: Starting RAG agent setup...")
        print(f"  PDF Path: {self.PDF_PATH}")
        print(f"  ChromaDB Path: {self.CHROMA_DB_PATH}")

        db_exists_and_has_data = False
        if os.path.exists(self.CHROMA_DB_PATH) and os.listdir(self.CHROMA_DB_PATH):
            print(f"  Found existing non-empty DB at {self.CHROMA_DB_PATH}. Attempting to use.")
            try:
                self._vector_store = ChromaStore(
                    path=self.CHROMA_DB_PATH,
                    collection_name=self.COLLECTION_NAME,
                    embedding_function=DEFAULT_EF
                )
                # A proper check would be collection.count() > 0 if API allows async count
                # For now, non-empty directory implies data.
                print(f"  Connected to existing ChromaDB collection '{self.COLLECTION_NAME}'. Skipping ingestion.")
                db_exists_and_has_data = True
            except Exception as e:
                print(f"  Error connecting to existing ChromaDB, will re-ingest: {e}")
                db_exists_and_has_data = False


        if not db_exists_and_has_data:
            if os.path.exists(self.CHROMA_DB_PATH): # Exists but was empty or failed to connect
                shutil.rmtree(self.CHROMA_DB_PATH)
            
            if not os.path.exists(self.PDF_PATH):
                print(f"  ERROR: PDF file not found at '{self.PDF_PATH}'. Cannot build RAG agent.")
                self._initialized = False # Mark as not successfully initialized
                return

            print(f"  Loading PDF from: {self.PDF_PATH}")
            pdf_content = load_pdf_file(self.PDF_PATH)
            if not pdf_content:
                print("  Failed to load content from PDF. Cannot build RAG agent.")
                self._initialized = False
                return

            print(f"  Chunking PDF content (Size: {self.CHUNK_SIZE}, Overlap: {self.CHUNK_OVERLAP})...")
            chunks = chunk_text_by_fixed_size(pdf_content, self.CHUNK_SIZE, self.CHUNK_OVERLAP)
            print(f"  Generated {len(chunks)} chunks.")

            if not chunks:
                print("  No chunks generated from PDF. Cannot build RAG agent.")
                self._initialized = False
                return

            self._vector_store = ChromaStore(
                path=self.CHROMA_DB_PATH,
                collection_name=self.COLLECTION_NAME,
                embedding_function=DEFAULT_EF
            )
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": os.path.basename(self.PDF_PATH), "chunk_index": i} for i in range(len(chunks))]

            print("  Adding chunks to vector store (this may take some time)...")
            ingestion_start_time = time.time()
            await self._vector_store.add_documents(documents=chunks, ids=ids, metadatas=metadatas)
            ingestion_time = time.time() - ingestion_start_time
            print(f"  Ingestion complete. Took {ingestion_time:.2f} seconds.")

        print(f"  Initializing LLM Service for RAG: {GoogleOpenAICompatService.__name__} with model {self.RAG_LLM_MODEL}")
        try:
            self._llm_service_for_rag = GoogleOpenAICompatService() 
        except Exception as e:
            print(f"  ERROR: Failed to initialize LLM Service for RAG: {e}")
            self._initialized = False
            return

        self._rag_agent = Agent( 
            name="A2A_RAG_Expert_CLAP_Agent",
            backstory="I am an expert assistant with access to detailed information about Holbox AI from a provided document. I will answer your questions based on this document.",
            task_description="Answer user queries based on the document.", # Generic
            task_expected_output="A concise and accurate answer derived solely from the retrieved document.",
            llm_service=self._llm_service_for_rag,
            model=self.RAG_LLM_MODEL,
            vector_store=self._vector_store
        )
        print("  CLAP RAG Agent initialized.")
        self._initialized = True
        print(f"ClapAgentA2AExecutorFastAPIStyle: RAG agent setup complete.")


    async def execute(self, user_input: str) -> str:
        if not self._initialized or not hasattr(self, '_rag_agent'):
            print("ClapAgentA2AExecutorFastAPIStyle: RAG Agent not initialized properly. Cannot execute.")
            return "Error: The RAG knowledge agent is not ready. Please try again later."

        print(f"ClapAgentA2AExecutorFastAPIStyle: Received RAG query: '{user_input}'")
        try:
            self._rag_agent.task_description = user_input
            
            clap_response_dict = await self._rag_agent.run() 
            clap_response_str = clap_response_dict.get("output", "RAG agent did not produce an 'output' string.")
            
            print(f"ClapAgentA2AExecutorFastAPIStyle: CLAP RAG agent responded: '{clap_response_str[:200]}...'")
            return clap_response_str
        except Exception as e:
            print(f"ClapAgentA2AExecutorFastAPIStyle: Error during CLAP RAG agent execution: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing your RAG query via CLAP agent: {str(e)}"
        finally:
            print(f"ClapAgentA2AExecutorFastAPIStyle: RAG 'execute' method finished for input: '{user_input}'.")

    async def close_resources(self):
        print("ClapAgentA2AExecutorFastAPIStyle: Closing resources...")
        if hasattr(self, '_llm_service_for_rag') and \
           hasattr(self._llm_service_for_rag, 'close') and \
           asyncio.iscoroutinefunction(self._llm_service_for_rag.close):
            try:
                await self._llm_service_for_rag.close()
                print("ClapAgentA2AExecutorFastAPIStyle: CLAP RAG LLM Service client closed.")
            except Exception as e:
                print(f"ClapAgentA2AExecutorFastAPIStyle: Error closing CLAP RAG LLM service: {e}")
        if hasattr(self, '_vector_store') and \
            hasattr(self._vector_store, 'close') and \
            asyncio.iscoroutinefunction(self._vector_store.close):
            try:
                await self._vector_store.close() 
                print("ClapAgentA2AExecutorFastAPIStyle: Vector store closed.")
            except Exception as e:
                print(f"ClapAgentA2AExecutorFastAPIStyle: Error closing vector store: {e}")

        print("ClapAgentA2AExecutorFastAPIStyle: Resource cleanup finished.")