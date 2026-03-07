import json
import os
import torch
from typing import Any, Callable, Dict, List, Optional, Sequence

import tiktoken
from langchain.chains import LLMChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.tools.retriever import create_retriever_tool
from pydantic import Field
from transformers import AutoModel, AutoTokenizer
from utils.retrieval_functions import hybrid_retrieve, list_sources as list_sources_fn, open_connection


retriever_prompt_template = """Given the query:\n{query}\n\n
Rate the relevance of the following document to the query on a scale from 1 to 10:\n{document}\n\n
Only output the score as an integer.
"""

batch_retriever_prompt_template = """Given the query:\n{query}\n\n
Rate the relevance of the following documents to the query on a scale from 1 to 10:\n{documents}\n\n
Only output the scores as a list of integers.
"""

SYSTEM_PROMPT_TEMPLATE = (
    "You are a knowledgeable assistant specializing in lore from {source_label}. "
    "You have access to the following tools:\n"
    "- retrieval: args: {{\"query\": <string>}} - Searches the documents for relevant information.\n"
    "- hybrid_retrieval: args: {{\"query\": <string>}} - Combines semantic and lexical search.\n"
    "- reranking_hybrid_retrieval: args: {{\"query\": <string>}} - Uses hybrid retrieval with LLM reranking for improved relevance.\n"
    "- list_sources: args: {{}} - Shows the available document sources.\n"
    "Always start by using one of the retrieval tools to gather relevant context before answering. "
    "Use list_sources if you are unsure which source to query. "
    "Call tools by responding only with a JSON block in the format {{\"tool\": \"tool_name\", \"args\": {{...}}}}. "
    "Otherwise, answer directly and cite relevant sources when possible."
)


system_message = (
            "You are a knowledgeable assistant specializing in J.R.R. Tolkien's 'The Silmarillion', the epic history of Middle-earth that chronicles the First Age of the world. You answer questions based on the provided context from the book, offering accurate, detailed, and engaging information about its rich lore, including characters, events, histories, genealogies, and themes.\n"
            "You have access to the following tools:\n"
            "- retrieval: args: {\"query\": <string>} - Searches The Silmarillion documents for relevant information.\n"
            "- hybrid_retrieval: args: {\"query\": <string>} - Searches The Silmarillion documents using hybrid retrieval combining semantic and lexical search.\n"
            "- reranking_hybrid_retrieval: args: {\"query\": <string>} - Searches The Silmarillion documents using hybrid retrieval with LLM reranking for improved relevance.\n"
            "Always first use one of the retrieval tools to retrieve relevant information from The Silmarillion.\n"
            "Be pro-active in using these tools to find the information needed to answer the user's question.\n"
            "Provide responses that are immersive and capture the spirit of Tolkien's narrative style, while remaining faithful to the source material.\n"
            "If the answer is in the documents, provide it with references to the relevant parts of The Silmarillion.\n"
            "If the answer is not in the documents, state that The Silmarillion does not contain the answer.\n"
            "To use a tool, respond ONLY with a JSON block in this format:\n"
            '{"tool": "tool_name", "args": {"arg1": "value1", ...}}.\n'
            "Do not include any explanation or extra text outside the JSON block when calling a tool.\n"
            "Otherwise, answer the user's question directly and clearly."
        )

class SimpleTool:
    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, args: Optional[Dict[str, Any]] = None) -> Any:  # pragma: no cover - simple wrapper
        return self.func(**(args or {}))


class LLMReranker(BaseDocumentCompressor):
    """LLM Reranker that uses LLMChain to rerank documents.
    It passes each document to the LLM and expects the LLM to return a score for each document.
    The documents are then sorted by score and the top_k documents are returned.
    """
    llm_chain: object = Field(LLMChain, description="LLM chain to rerank documents")
    document_variable_name: str = "document"
    top_k: int = 5  # Number of top documents to return

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        *,
        callbacks: Optional[list] = None,
    ) -> List[Document]:
        
        scored_docs = []
        for doc in documents:
            inputs = {
                "query": query,
                self.document_variable_name: doc.page_content,
            }
            output = self.llm_chain.invoke(inputs)
            try:
                score = int(output.strip())
            except Exception:
                score = 0
            scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:self.top_k]]  # Return only top_k documents    

class LLMRerankerBatched(BaseDocumentCompressor):
    """ LLM Reranker that uses LLMChain to rerank documents in batches.
    It passes the documents to the LLM in a single call and expects the LLM to return a list of scores.
    """
    llm_chain: object = Field(LLMChain, description="LLM chain to rerank documents in batches")
    document_variable_name: str = "documents"
    top_k: int = 5  # Number of top documents to return

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        *,
        callbacks: Optional[list] = None,
    ) -> List[Document]:
        
        scored_docs = []
        
        inputs = {
            "query": query,
            self.document_variable_name: [doc.page_content for doc in documents],
        }
        output = self.llm_chain.invoke(inputs)
        try:
            scores = [int(score.strip()) for score in output.split(",")]
        except Exception:
            scores = [0] * len(documents)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.top_k]]
    
class RerankingRetriever:
    """
    Encapsulates a retriever with LLM-based reranking using ContextualCompressionRetriever.
    """
    def __init__(
        self,
        vector_store,
        reranker_model_name: str = "gpt-4o-mini",
        rerank_top_k: int = 10,
        retriever_k: int = 20,
        source_filter: Optional[str] = None,
    ):
        # Set up the base retriever
        search_kwargs: Dict[str, Any] = {"k": retriever_k}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}
        self.base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        self.source_filter = source_filter

        self.rerank_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=batch_retriever_prompt_template,
        )

        # Set up the reranker LLM
        self.llm_reranker = ChatOpenAI(
            model_name=reranker_model_name,
            temperature=0.0
        )

        # Set up the reranker compressor
        self.reranker_batched = LLMRerankerBatched(
            llm_chain=self.rerank_prompt | self.llm_reranker,
            top_k=rerank_top_k
        )

        # Compose the contextual compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.base_retriever,
            base_compressor=self.reranker_batched,
        )

    def as_tool(self, name="retrieval", description="Searches the project documents for relevant information."):
        """
        Returns a retriever tool for use in tool-augmented chat.
        """
        return create_retriever_tool(
            self.retriever,
            name=name,
            description=description
        )


class HybridRetriever(BaseRetriever):
    """
    A retriever that uses hybrid search combining semantic and lexical retrieval with RRF.
    """
    dsn: str = ""
    profile: str = ""
    embedder: Any = None
    top_k: int = 8
    candidate_k: int = 100
    rrf_k: int = 60
    source_filter: Optional[str] = None

    def __init__(
        self,
        dsn: str,
        profile: str,
        embedder: Any,
        top_k: int = 8,
        candidate_k: int = 100,
        rrf_k: int = 60,
        source_filter: Optional[str] = None,
    ):
        super().__init__()
        self.dsn = dsn
        self.profile = profile
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.source_filter = source_filter

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        
        # Open connection
        conn = open_connection(self.dsn)
        try:
            # Perform hybrid retrieval
            rows = hybrid_retrieve(
                conn,
                profile=self.profile,
                query_text=query,
                query_embedding=query_embedding,
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                rrf_k=self.rrf_k,
                source_filter=self.source_filter,
            )
            # Convert to Document objects
            documents = [
                Document(
                    page_content=row.content,
                    metadata={"source": row.source, "id": row.id, "rrf_score": row.rrf_score}
                )
                for row in rows
            ]
            return documents
        finally:
            conn.close()

    def as_tool(self, name="hybrid_retrieval", description="Searches the project documents using hybrid retrieval (semantic + lexical)."):
        """
        Returns a retriever tool for use in tool-augmented chat.
        """
        return create_retriever_tool(
            self,
            name=name,
            description=description
        )


class RerankingHybridRetriever:
    """
    Encapsulates a hybrid retriever with LLM-based reranking.
    """
    def __init__(
        self,
        dsn: str,
        profile: str,
        embedder: Any,
        reranker_model_name: str = "gpt-4o-mini",
        rerank_top_k: int = 10,
        retriever_top_k: int = 20,
        candidate_k: int = 100,
        rrf_k: int = 60,
        source_filter: Optional[str] = None,
    ):
        # Set up the base hybrid retriever
        self.base_retriever = HybridRetriever(
            dsn=dsn,
            profile=profile,
            embedder=embedder,
            top_k=retriever_top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            source_filter=source_filter,
        )

        self.rerank_prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template=batch_retriever_prompt_template,
        )

        # Set up the reranker LLM
        self.llm_reranker = ChatOpenAI(
            model_name=reranker_model_name,
            temperature=0.0
        )

        # Set up the reranker compressor
        self.reranker_batched = LLMRerankerBatched(
            llm_chain=self.rerank_prompt | self.llm_reranker,
            top_k=rerank_top_k
        )

        # Compose the contextual compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.base_retriever,
            base_compressor=self.reranker_batched,
        )

    def as_tool(self, name="reranked_hybrid_retrieval", description="Searches the project documents using hybrid retrieval with LLM reranking."):
        """
        Returns a retriever tool for use in tool-augmented chat.
        """
        return create_retriever_tool(
            self.retriever,
            name=name,
            description=description
        )


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        if model_name == "text-embedding-ada-002":
            # no need to import unless we're actually using the OpenAI embedding model
            from langchain_openai import OpenAIEmbeddings
            self.embedding_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.is_openai = True
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.is_openai = False

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the input text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the input text."""
        if self.is_openai:
            enc = tiktoken.encoding_for_model("text-embedding-ada-002")
            return len(enc.encode(text))
        else:
            return self._count_tokens(text)
    
class WrappedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    
class VectorStore:
    def __init__(self, collection_name: str, connection_string: str, embeddings: Embedder, async_mode: bool = False):
        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection= connection_string,
            async_mode= async_mode,
        )

    def as_retriever(self, search_kwargs: dict = None):
        if search_kwargs is None:
            search_kwargs = {"k": 20}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    

class ChatManagerWithTools:
    def __init__(self, config=None, debug=False):
        self.config = config or {}
        self.debug = debug
        self._setup_env()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_llm()
        self._setup_memory()
        self.selected_source: Optional[str] = None
        self.retrieve_k: int = 10
        self.rerank_top_k: int = 5
        self.max_memory_tokens = int(os.getenv("MAX_MEMORY_TOKENS", "8000"))
        self._refresh_tools_for_source()
        self._setup_prompt()

    def _setup_env(self):
        self.PGHOST = os.getenv("PGHOST", "localhost")
        self.PGPORT = os.getenv("PGPORT")
        self.PGUSER = os.getenv("PGUSER")
        self.PGPASSWORD = os.getenv("PGPASSWORD")
        self.PGDATABASE = os.getenv("PGDATABASE")
        self.connection_string = f"postgresql+psycopg://{self.PGUSER}:{self.PGPASSWORD}@{self.PGHOST}:{self.PGPORT}/{self.PGDATABASE}"
        self.hybrid_dsn = self.connection_string.replace("postgresql+psycopg://", "postgresql://")
        self.collection_name = "project_documents"
        self.profile = os.getenv("PROFILE", "md")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.rerank_model = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")

    def _setup_embeddings(self):
        model_name = {
            "mini": os.getenv("DEPLOY_MINI", "sentence-transformers/all-MiniLM-L6-v2"),
            "sm": os.getenv("DEPLOY_SMALL", "BAAI/bge-large-en-v1.5"),
            "md": os.getenv("DEPLOY_MEDIUM", "text-embedding-ada-002"),
        }[self.profile]
        self.embedding_model = Embedder(model_name=model_name)

    def _setup_vector_store(self):
        self.vector_store = VectorStore(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embeddings=self.embedding_model,
        )

    def _setup_llm(self):
        self.llm = ChatOpenAI(
            model=self.chat_model,
            temperature=0.2,
            streaming=True,
        )

    def _setup_memory(self):
        self.message_history = InMemoryChatMessageHistory()

    def _refresh_tools_for_source(self):
        source_filter = self.selected_source
        self.reranking_retriever = RerankingRetriever(
            vector_store=self.vector_store,
            reranker_model_name=self.rerank_model,
            rerank_top_k=self.rerank_top_k,
            retriever_k=self.retrieve_k,
            source_filter=source_filter,
        )
        self.hybrid_retriever = HybridRetriever(
            dsn=self.hybrid_dsn,
            profile=self.profile,
            embedder=self.embedding_model,
            top_k=10,
            source_filter=source_filter,
        )
        self.reranking_hybrid_retriever = RerankingHybridRetriever(
            dsn=self.hybrid_dsn,
            profile=self.profile,
            embedder=self.embedding_model,
            reranker_model_name=self.rerank_model,
            rerank_top_k=self.rerank_top_k,
            retriever_top_k=self.retrieve_k,
            source_filter=source_filter,
        )
        self.tools = {
            "retrieval": self.reranking_retriever.as_tool(
                name="retrieval",
                description="Searches the documents for relevant information."
            ),
            "hybrid_retrieval": self.hybrid_retriever.as_tool(
                name="hybrid_retrieval",
                description="Searches the documents using hybrid retrieval combining semantic and lexical search."
            ),
            "reranking_hybrid_retrieval": self.reranking_hybrid_retriever.as_tool(
                name="reranking_hybrid_retrieval",
                description="Searches the documents using hybrid retrieval with LLM reranking for improved relevance."
            ),
            "list_sources": SimpleTool(
                name="list_sources",
                description="Lists the available sources for the current profile.",
                func=lambda: self._invoke_list_sources(),
            ),
        }

    def _invoke_list_sources(self) -> str:
        sources = self.list_available_sources()
        if not sources:
            return "No sources available for the current profile."
        lines = ["Available sources:"] + [f"- {source}" for source in sources]
        return "\n".join(lines)

    def list_available_sources(self) -> List[str]:
        conn = None
        try:
            conn = open_connection(self.hybrid_dsn)
            return list_sources_fn(conn, profile=self.profile)
        except Exception:
            return []
        finally:
            if conn:
                conn.close()

    def set_source(self, source: Optional[str]) -> None:
        normalized = source or None
        if normalized == self.selected_source:
            return
        self.selected_source = normalized
        self._refresh_tools_for_source()
        self._setup_prompt()

    def set_retrieval_params(self, retrieve_k: int, rerank_top_k: int) -> None:
        max_retrieve = 30
        max_rerank = 15
        bounded_retrieve = min(max(1, retrieve_k), max_retrieve)
        bounded_rerank = min(max(1, min(rerank_top_k, bounded_retrieve)), max_rerank)
        if bounded_retrieve == self.retrieve_k and bounded_rerank == self.rerank_top_k:
            return
        self.retrieve_k = bounded_retrieve
        self.rerank_top_k = bounded_rerank
        self._refresh_tools_for_source()

    def _setup_prompt(self):
        source_label = self.selected_source or "the available knowledge base"
        self.system_message = SYSTEM_PROMPT_TEMPLATE.format(source_label=source_label)

    def _count_memory_tokens(self) -> int:
        return sum(self.embedding_model.count_tokens(msg.content) for msg in self.message_history.messages)

    def _enforce_memory_limit(self, max_tokens: Optional[int] = None):
        messages = self.message_history.messages
        limit = max_tokens or self.max_memory_tokens
        while self._count_memory_tokens() > limit and messages:
            # Remove the oldest message (after system message)
            messages.pop(0)

    def _enforce_memory_message_limit(self, max_messages: int = 20):
        """
        Keeps only the most recent `max_messages` in memory.
        """
        messages = self.message_history.messages
        if len(messages) > max_messages:
            # Remove oldest messages, keep only the last `max_messages`
            del messages[:len(messages) - max_messages]

    # All the milk and honey is here
    async def stream_response(self, user_message: str, debug: bool = False):
        self.debug = debug  # Update debug flag
        # Build chat history
        history_msgs = self.message_history.messages
        system_msg = SystemMessage(content=self.system_message)
        user_msg = HumanMessage(content=user_message)
        messages = [system_msg] + history_msgs + [user_msg]

        # Stream LLM response
        assistant_reply = ""
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                assistant_reply += chunk.content
 
        # Check if LLM wants to use a tool (by outputting a JSON block)
        tool_call = self._extract_tool_call(assistant_reply)
        if tool_call:
            if self.debug:
                print(f"[DEBUG] Tool call detected: {tool_call}")
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})
            tool_func = self.tools.get(tool_name)
            if tool_func:
                if self.debug:
                    print(f"[DEBUG] Invoking tool: {tool_name} with args: {args}")
                tool_result = tool_func.invoke(args)
                if self.debug:
                    print(f"[DEBUG] Tool result: {tool_result[:200]}...")  # Truncate for readability
                tool_context_msg = HumanMessage(
                    content=f"The result of your tool call `{tool_name}` is:\n{tool_result}\n"
                            "Please use this information to answer the user's question."
                )
                messages.append(AIMessage(content=assistant_reply))
                messages.append(tool_context_msg)
                final_reply = ""
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        final_reply += chunk.content
                        yield chunk.content  # Only yield the final answer
                assistant_reply = final_reply
        else:
            # Only yield if no tool call was made
            yield assistant_reply

        # Update memory
        self.message_history.add_user_message(user_message)
        self.message_history.add_ai_message(assistant_reply)
        # self._enforce_memory_limit(max_tokens=4000)
        self._enforce_memory_message_limit(max_messages=20)

    def _extract_tool_call(self, text: str) -> Dict[str, Any] | None:
        # Look for a JSON block in the LLM output
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_block = text[start:end]
            return json.loads(json_block)
        except Exception:
            return None