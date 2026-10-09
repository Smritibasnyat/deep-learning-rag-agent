"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """Connect to the persistent ChromaDB collection."""
        import chromadb

        try:
            Path(self._settings.chroma_db_path).mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_db_path,
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "Collection '{}' ready with {} chunks",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not initialise ChromaDB."
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """Check whether this chunk is already stored."""
        result = self._collection.get(ids=[chunk_id])
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """Embed and store chunks, skipping existing duplicates."""
        result = IngestionResult()

        for chunk in chunks:
            if self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue

            embeddings = self._embeddings.embed_documents(
                [chunk.chunk_text]
            )

            self._collection.upsert(
                ids=[chunk.chunk_id],
                embeddings=embeddings,
                documents=[chunk.chunk_text],
                metadatas=[chunk.metadata.to_dict()],
            )

            result.ingested += 1

        logger.info(
            "Ingested {} chunks; skipped {} duplicates",
            result.ingested,
            result.skipped,
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using cosine similarity."""
        count = self._collection.count()
        limit = self._settings.retrieval_k if k is None else k

        if count == 0 or limit <= 0 or not query_text.strip():
            return []

        filters = []
        if topic_filter:
            filters.append({"topic": topic_filter})
        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        return sorted(
            retrieved,
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        # TODO: implement
        # Query all metadata from the collection
        # Group by metadata["source"] and count chunks per source
        # Return sorted list of dicts
        raise NotImplementedError

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        # TODO: implement
        # self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        # Reconstruct DocumentChunk objects from results
        raise NotImplementedError

    def get_collection_stats(self) -> dict:
        """Return corpus counts and metadata summaries."""
        result = self._collection.get(include=["metadatas"])
        metadata = result["metadatas"] or []

        return {
            "total_chunks": self._collection.count(),
            "topics": sorted(
                {item["topic"] for item in metadata}
            ),
            "sources": sorted(
                {item["source"] for item in metadata}
            ),
            "bonus_topics_present": any(
                item.get("is_bonus") == "true"
                for item in metadata
            ),
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        # TODO: implement
        # self._collection.delete(where={"source": source})
        raise NotImplementedError
