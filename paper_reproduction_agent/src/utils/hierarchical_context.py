"""
Hierarchical Context Manager with Multi-Factor Relevance Scoring

Implements 2025 best practices for context engineering:
- Three-tier storage: Hot (memory) -> Warm (vector store) -> Cold (summaries)
- Token-based budget management
- Multi-factor relevance scoring (semantic + recency + importance + authority)

Reference: Google ADK approach (December 2025), Krishnan (2025)
"""

import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Single context entry with metadata for relevance scoring."""

    id: str
    content: str
    source: str  # "paper_analyzer", "discovery", "reproduction", etc.
    entry_type: str  # "result", "error", "decision", "observation", "debug"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    tokens: int = 0
    embedding: Optional[List[float]] = None
    pending_embedding: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "tokens": self.tokens,
        }


class HierarchicalContextManager:
    """
    Three-tier context storage with semantic retrieval.

    Storage Tiers:
    - Hot: Recent entries in memory (OrderedDict for LRU-like behavior)
    - Warm: Vector store for semantic retrieval (ChromaDB)
    - Cold: Summarized historical context (LLM-compressed)

    Relevance Scoring Formula:
        score = 0.4 * semantic_similarity
              + 0.3 * recency_decay
              + 0.2 * importance
              + 0.1 * source_authority
    """

    # Relevance scoring weights (from 2025 best practices)
    WEIGHT_SEMANTIC = 0.4
    WEIGHT_RECENCY = 0.3
    WEIGHT_IMPORTANCE = 0.2
    WEIGHT_SOURCE_AUTHORITY = 0.1

    # Source authority scores - paper facts highest, debug lowest
    SOURCE_AUTHORITY = {
        "paper_analyzer": 1.0,  # Paper facts are highest authority
        "user": 1.0,  # User input is authoritative
        "reproduction": 0.9,  # Execution results are important
        "discovery": 0.8,  # Repository discovery
        "environment_setup": 0.7,
        "system": 0.5,
        "debug": 0.3,
    }

    # Type importance multipliers
    TYPE_IMPORTANCE = {
        "result": 1.0,
        "error": 0.9,
        "decision": 0.8,
        "observation": 0.6,
        "debug": 0.3,
    }

    def __init__(
        self,
        model_name: str = "gpt-4",
        hot_capacity: int = 30,
        max_tokens: int = 50000,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: Optional[str] = None,
        embedder: Optional[Any] = None,
    ):
        """
        Initialize hierarchical context manager.

        Args:
            model_name: Model name for tokenizer selection (gpt-4, claude, etc.)
            hot_capacity: Maximum entries in hot storage before compaction
            max_tokens: Maximum tokens to return from retrieval
            embedding_model: Sentence transformer model for embeddings (used if embedder not provided)
            collection_name: ChromaDB collection name (auto-generated if None)
            embedder: External embedder instance (e.g., GeminiEmbedder). If provided,
                      bypasses local SentenceTransformer loading for faster startup.
        """
        self.model_name = model_name
        self.hot_capacity = hot_capacity
        self.max_tokens = max_tokens
        self.embedding_model_name = embedding_model

        # Initialize tokenizer
        self._init_tokenizer()

        # Use provided embedder or lazy-load local model later
        self._embedder = embedder
        self._embedder_provided = embedder is not None

        # Hot storage: OrderedDict for LRU-like behavior
        self.hot_context: OrderedDict[str, ContextEntry] = OrderedDict()

        # Warm storage: ChromaDB for vector retrieval (lazy init)
        self._chroma_client = None
        self._collection = None
        self._collection_name = collection_name or f"context_{int(time.time())}"

        # Cold storage: Summaries
        self.cold_summaries: List[str] = []

        # Track total tokens in hot storage
        self.hot_tokens = 0

        logger.info(
            f"HierarchicalContextManager initialized: "
            f"hot_capacity={hot_capacity}, max_tokens={max_tokens}"
        )

    def _init_tokenizer(self):
        """Initialize tiktoken tokenizer."""
        try:
            import tiktoken

            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fallback to cl100k_base (GPT-4/ChatGPT encoding)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not available, using character-based estimation")
            self.tokenizer = None

    @property
    def embedder(self):
        """Return embedder (provided externally or lazy-loaded local model)."""
        # If embedder was provided externally, use it
        if self._embedder_provided:
            return self._embedder

        # Otherwise, lazy-load local SentenceTransformer
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, semantic search disabled"
                )
                self._embedder = False  # Mark as unavailable
        return self._embedder if self._embedder else None

    @property
    def collection(self):
        """Lazy load ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb

                self._chroma_client = chromadb.Client()
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self._collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"ChromaDB collection initialized: {self._collection_name}")
            except ImportError:
                logger.warning("chromadb not available, warm storage disabled")
                self._collection = False  # Mark as unavailable
        return self._collection if self._collection else None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens for text using tiktoken.

        Falls back to character-based estimation if tiktoken unavailable.
        """
        if not text:
            return 0

        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: ~4 characters per token
            return len(text) // 4

    def generate_id(self, content: str, source: str) -> str:
        """Generate unique ID for entry based on content hash."""
        hash_input = f"{content[:100]}{source}{time.time()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def add(
        self,
        content: str,
        source: str = "unknown",
        entry_type: str = "observation",
        importance: Optional[float] = None,
        lazy: bool = False,
    ) -> str:
        """
        Add new context entry to storage.

        Adds to both hot storage (fast access) and warm storage (semantic search).
        Automatically compacts hot storage when capacity exceeded.

        Args:
            content: The context content to store
            source: Source identifier (paper_analyzer, reproduction, etc.)
            entry_type: Type of entry (result, error, decision, observation, debug)
            importance: Override importance score (0.0 to 1.0)
            lazy: If True, skip embedding generation (useful for rapid initialization)

        Returns:
            Entry ID
        """
        if not content or not content.strip():
            logger.warning("Attempted to add empty content, skipping")
            return ""

        # Calculate importance if not provided
        if importance is None:
            importance = self.TYPE_IMPORTANCE.get(entry_type, 0.5)

        # Generate entry
        entry_id = self.generate_id(content, source)
        tokens = self.count_tokens(content)

        # Generate embedding if available AND NOT LAZY
        embedding = None
        if not lazy and self.embedder:
            try:
                embedding = self.embedder.encode(content).tolist()
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        entry = ContextEntry(
            id=entry_id,
            content=content,
            source=source,
            entry_type=entry_type,
            importance=importance,
            tokens=tokens,
            embedding=embedding,
            pending_embedding=lazy,
        )

        # Add to hot storage
        self.hot_context[entry_id] = entry
        self.hot_tokens += tokens

        # Add to warm storage (vector store) immediately ONLY if not lazy and embedding exists
        if not lazy and self.collection and embedding:
            try:
                self.collection.add(
                    ids=[entry_id],
                    embeddings=[embedding],
                    metadatas=[entry.to_dict()],
                    documents=[content],
                )
            except Exception as e:
                logger.warning(f"Failed to add to vector store: {e}")

        # Compact if needed
        if len(self.hot_context) > self.hot_capacity:
            self._compact_hot_to_warm()

        logger.debug(
            f"Added context entry: id={entry_id}, source={source}, "
            f"type={entry_type}, tokens={tokens}, lazy={lazy}"
        )

        return entry_id

    def flush_embeddings(self):
        """
        Process any pending embeddings in hot storage.
        
        This forces the embedding model to load and processes all entries
        marked as pending_embedding. Call this before starting phases
        that require semantic retrieval.
        """
        pending_entries = [
            e for e in self.hot_context.values() if e.pending_embedding
        ]
        
        if not pending_entries:
            return

        logger.info(f"Flushing embeddings for {len(pending_entries)} pending entries...")
        
        # Ensure model is loaded
        if not self.embedder:
            logger.warning("Embedder unavailable, skipping flush")
            return

        count = 0
        for entry in pending_entries:
            try:
                # Generate embedding
                entry.embedding = self.embedder.encode(entry.content).tolist()
                entry.pending_embedding = False
                
                # Update vector store
                if self.collection:
                    self.collection.add(
                        ids=[entry.id],
                        embeddings=[entry.embedding],
                        metadatas=[entry.to_dict()],
                        documents=[entry.content],
                    )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to generate embedding during flush: {e}")

        logger.info(f"Successfully generated embeddings for {count} entries")

    def _compact_hot_to_warm(self):
        """
        Move oldest entries from hot to warm storage.

        Keeps the most recent half of hot_capacity entries.
        Entries remain in warm storage for semantic retrieval.
        """
        entries_to_remove = list(self.hot_context.keys())[: -self.hot_capacity // 2]

        removed_tokens = 0
        for entry_id in entries_to_remove:
            entry = self.hot_context.pop(entry_id)
            removed_tokens += entry.tokens

        self.hot_tokens -= removed_tokens

        logger.info(
            f"Compacted hot storage: removed {len(entries_to_remove)} entries, "
            f"freed {removed_tokens} tokens"
        )

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot_product = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_relevance(
        self,
        entry: ContextEntry,
        query_embedding: Optional[List[float]],
        semantic_score: float = 0.0,
    ) -> float:
        """
        Calculate multi-factor relevance score.

        Formula: 0.4 * semantic + 0.3 * recency + 0.2 * importance + 0.1 * authority

        Args:
            entry: Context entry to score
            query_embedding: Query embedding for semantic scoring
            semantic_score: Pre-computed semantic similarity (0.0 to 1.0)

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Recency score: decay over time
        # 1.0 for now, 0.5 after 1 hour, 0.1 minimum
        age_seconds = time.time() - entry.timestamp
        recency = max(0.1, 1.0 - (age_seconds / 7200))  # 2-hour decay to 0.5

        # Source authority
        authority = self.SOURCE_AUTHORITY.get(entry.source, 0.5)

        # If we have embedding but no pre-computed score, calculate it
        if semantic_score == 0.0 and query_embedding and entry.embedding:
            semantic_score = max(
                0.0, self._cosine_similarity(entry.embedding, query_embedding)
            )

        # Combined score
        score = (
            self.WEIGHT_SEMANTIC * semantic_score
            + self.WEIGHT_RECENCY * recency
            + self.WEIGHT_IMPORTANCE * entry.importance
            + self.WEIGHT_SOURCE_AUTHORITY * authority
        )

        return score

    def retrieve(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        include_cold: bool = True,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using multi-factor scoring.

        Searches both hot storage (recent) and warm storage (semantic).
        Returns entries sorted by relevance, respecting token budget.

        Args:
            query: Query string for retrieval
            max_tokens: Maximum tokens to return (defaults to half of max_tokens)
            include_cold: Whether to include cold summaries
            min_relevance: Minimum relevance score to include (0.0 to 1.0)

        Returns:
            List of context entries with relevance scores
        """
        if max_tokens is None:
            max_tokens = self.max_tokens // 2  # Reserve half for response

        # Generate query embedding
        query_embedding = None
        if self.embedder:
            try:
                query_embedding = self.embedder.encode(query).tolist()
            except Exception as e:
                logger.warning(f"Failed to encode query: {e}")

        scored_entries = []
        seen_ids = set()

        # Score hot entries (in-memory, fast)
        for entry in self.hot_context.values():
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)

            semantic_score = 0.0
            if query_embedding and entry.embedding:
                semantic_score = max(
                    0.0, self._cosine_similarity(entry.embedding, query_embedding)
                )

            relevance = self._calculate_relevance(
                entry, query_embedding, semantic_score
            )

            if relevance >= min_relevance:
                scored_entries.append(
                    {
                        "entry": entry,
                        "relevance": relevance,
                        "semantic_score": semantic_score,
                        "storage": "hot",
                    }
                )

        # Query warm storage (vector search)
        if self.collection and query_embedding:
            try:
                warm_count = self.collection.count()
                if warm_count > 0:
                    warm_results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(50, warm_count),
                        include=["metadatas", "documents", "distances"],
                    )

                    # Process warm results
                    if warm_results["ids"] and warm_results["ids"][0]:
                        for i, entry_id in enumerate(warm_results["ids"][0]):
                            if entry_id in seen_ids:
                                continue
                            seen_ids.add(entry_id)

                            metadata = warm_results["metadatas"][0][i]
                            content = warm_results["documents"][0][i]
                            distance = warm_results["distances"][0][i]

                            # Reconstruct entry
                            entry = ContextEntry(
                                id=entry_id,
                                content=content,
                                source=metadata.get("source", "unknown"),
                                entry_type=metadata.get("entry_type", "observation"),
                                timestamp=metadata.get("timestamp", 0),
                                importance=metadata.get("importance", 0.5),
                                tokens=metadata.get(
                                    "tokens", self.count_tokens(content)
                                ),
                            )

                            # ChromaDB returns distance, convert to similarity
                            semantic_score = max(0.0, 1 - distance)
                            relevance = self._calculate_relevance(
                                entry, query_embedding, semantic_score
                            )

                            if relevance >= min_relevance:
                                scored_entries.append(
                                    {
                                        "entry": entry,
                                        "relevance": relevance,
                                        "semantic_score": semantic_score,
                                        "storage": "warm",
                                    }
                                )
            except Exception as e:
                logger.warning(f"Warm storage query failed: {e}")

        # Sort by relevance
        scored_entries.sort(key=lambda x: x["relevance"], reverse=True)

        # Select entries within token budget
        selected = []
        total_tokens = 0

        # Add cold summaries first if requested
        if include_cold and self.cold_summaries:
            # Get last 3 summaries
            summaries = self.cold_summaries[-3:]
            summary_text = "\n---\n".join(summaries)
            summary_tokens = self.count_tokens(summary_text)

            if summary_tokens < max_tokens:
                selected.append(
                    {
                        "content": f"[Historical Summary]\n{summary_text}",
                        "relevance": 0.5,  # Moderate relevance
                        "source": "cold_summary",
                        "type": "summary",
                        "storage": "cold",
                        "tokens": summary_tokens,
                    }
                )
                total_tokens += summary_tokens

        # Add scored entries within budget
        for item in scored_entries:
            entry = item["entry"]

            if total_tokens + entry.tokens > max_tokens:
                continue

            selected.append(
                {
                    "content": entry.content,
                    "relevance": item["relevance"],
                    "semantic_score": item.get("semantic_score", 0.0),
                    "source": entry.source,
                    "type": entry.entry_type,
                    "storage": item["storage"],
                    "tokens": entry.tokens,
                }
            )
            total_tokens += entry.tokens

        logger.debug(
            f"Retrieved {len(selected)} entries ({total_tokens} tokens) "
            f"for query: {query[:50]}..."
        )

        return selected

    def compile_context(
        self, query: str, system_prompt: str = "", max_tokens: Optional[int] = None
    ) -> str:
        """
        Compile final context string for LLM consumption.

        Retrieves relevant context and formats it for inclusion in prompts.
        Respects token budget including system prompt overhead.

        Args:
            query: Query/task description for relevance scoring
            system_prompt: System prompt to account for in budget
            max_tokens: Total token budget (defaults to self.max_tokens)

        Returns:
            Formatted context string
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Reserve tokens for system prompt and response
        system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
        response_reserve = max_tokens // 4  # Reserve 25% for response
        available = max_tokens - system_tokens - response_reserve

        if available <= 0:
            logger.warning("No token budget available for context")
            return ""

        # Retrieve relevant context
        relevant = self.retrieve(query, max_tokens=available)

        if not relevant:
            return ""

        # Format context sections
        sections = []
        for item in relevant:
            source_label = item.get("source", "context")
            entry_type = item.get("type", "")
            content = item["content"]

            # Format header
            header = f"[{source_label}"
            if entry_type and entry_type != "observation":
                header += f"/{entry_type}"
            header += "]"

            sections.append(f"{header}\n{content}")

        compiled = "\n\n---\n\n".join(sections)

        logger.info(
            f"Compiled context: {len(relevant)} entries, "
            f"{self.count_tokens(compiled)} tokens"
        )

        return compiled

    def create_cold_summary(
        self, summarize_fn: Callable[[str], str], max_entries: int = 20
    ) -> str:
        """
        Create summary of warm storage and add to cold storage.

        Uses provided LLM summarization function to compress context.

        Args:
            summarize_fn: Function that takes text and returns summary
            max_entries: Maximum entries to include in summary

        Returns:
            Generated summary
        """
        if not self.collection:
            logger.warning("No warm storage available for summarization")
            return ""

        try:
            # Get entries from warm storage
            all_entries = self.collection.get(
                include=["documents", "metadatas"], limit=max_entries
            )

            if not all_entries["documents"]:
                return ""

            # Build content for summarization
            content_parts = []
            for i, doc in enumerate(all_entries["documents"]):
                meta = all_entries["metadatas"][i] if all_entries["metadatas"] else {}
                source = meta.get("source", "unknown")
                content_parts.append(f"[{source}] {doc}")

            content = "\n\n".join(content_parts)

            # Summarize
            summary = summarize_fn(content)

            if summary:
                self.cold_summaries.append(summary)
                logger.info(
                    f"Created cold summary: {self.count_tokens(summary)} tokens"
                )

            return summary

        except Exception as e:
            logger.error(f"Failed to create cold summary: {e}")
            return ""

    def clear_hot(self):
        """Clear hot storage only (warm and cold preserved)."""
        self.hot_context.clear()
        self.hot_tokens = 0
        logger.info("Hot storage cleared")

    def clear_all(self):
        """Clear all storage tiers."""
        self.hot_context.clear()
        self.hot_tokens = 0
        self.cold_summaries.clear()

        if self._collection:
            try:
                # Delete and recreate collection
                self._chroma_client.delete_collection(self._collection_name)
                self._collection = None
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB collection: {e}")

        logger.info("All context storage cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics.

        Returns:
            Dictionary with storage statistics
        """
        warm_count = 0
        if self.collection:
            try:
                warm_count = self.collection.count()
            except Exception:
                pass

        return {
            "hot_entries": len(self.hot_context),
            "hot_tokens": self.hot_tokens,
            "warm_entries": warm_count,
            "cold_summaries": len(self.cold_summaries),
            "max_tokens": self.max_tokens,
            "hot_capacity": self.hot_capacity,
            "embedder_available": self.embedder is not None,
            "vector_store_available": self.collection is not None,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"HierarchicalContextManager("
            f"hot={stats['hot_entries']}/{self.hot_capacity}, "
            f"warm={stats['warm_entries']}, "
            f"cold={stats['cold_summaries']}, "
            f"tokens={stats['hot_tokens']}/{self.max_tokens})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize manager state to dictionary for checkpoint saving.

        Note: ChromaDB (warm storage) is not directly serialized as it's ephemeral.
        Hot entries with embeddings will be re-added to warm storage on restore.

        Returns:
            Dictionary containing serializable state
        """
        hot_entries = []
        for entry_id, entry in self.hot_context.items():
            entry_dict = entry.to_dict()
            # Include embedding if available (for restoration to warm storage)
            if entry.embedding:
                entry_dict["embedding"] = entry.embedding
            hot_entries.append(entry_dict)

        return {
            "model_name": self.model_name,
            "hot_capacity": self.hot_capacity,
            "max_tokens": self.max_tokens,
            "embedding_model_name": self.embedding_model_name,
            "collection_name": self._collection_name,
            "hot_entries": hot_entries,
            "hot_tokens": self.hot_tokens,
            "cold_summaries": list(self.cold_summaries),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalContextManager":
        """
        Restore manager state from dictionary.

        Creates a new instance with restored hot and cold storage.
        Entries with embeddings are re-added to warm storage.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored HierarchicalContextManager instance
        """
        manager = cls(
            model_name=data.get("model_name", "gpt-4"),
            hot_capacity=data.get("hot_capacity", 30),
            max_tokens=data.get("max_tokens", 50000),
            embedding_model=data.get("embedding_model_name", "all-MiniLM-L6-v2"),
            collection_name=data.get("collection_name"),
        )

        # Restore cold summaries
        manager.cold_summaries = list(data.get("cold_summaries", []))

        # Restore hot entries
        for entry_data in data.get("hot_entries", []):
            entry = ContextEntry(
                id=entry_data["id"],
                content=entry_data["content"],
                source=entry_data["source"],
                entry_type=entry_data["entry_type"],
                timestamp=entry_data.get("timestamp", time.time()),
                importance=entry_data.get("importance", 0.5),
                tokens=entry_data.get("tokens", 0),
                embedding=entry_data.get("embedding"),
            )
            manager.hot_context[entry.id] = entry
            manager.hot_tokens += entry.tokens

            # Re-add to warm storage if embedding available
            if entry.embedding and manager.collection:
                try:
                    manager.collection.add(
                        ids=[entry.id],
                        embeddings=[entry.embedding],
                        metadatas=[entry.to_dict()],
                        documents=[entry.content],
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore entry to warm storage: {e}")

        logger.info(
            f"Restored HierarchicalContextManager: "
            f"hot={len(manager.hot_context)}, cold={len(manager.cold_summaries)}"
        )

        return manager
