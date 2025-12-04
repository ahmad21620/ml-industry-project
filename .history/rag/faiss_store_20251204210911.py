from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DOCS_DIR, FAISS_INDEX_DIR, embeddings


@dataclass
class RetrievedChunk:
    content: str
    score: float
    source_file: str
    section_heading: str
    chunk_id: int
    start_index: Optional[int]


class RAGAgent:
    """
    RAG component:
    - builds a FAISS index from local docs
    - performs semantic retrieval
    """

    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        index_dir: Path = FAISS_INDEX_DIR,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._vectorstore: Optional[FAISS] = None

    # --------- PUBLIC API ---------

    def build_or_load_index(self) -> None:
        """
        Load an existing FAISS index if present,
        otherwise build from the docs in docs_dir and persist it.
        """
        if (self.index_dir / "index.faiss").exists():
            self._load_index()
        else:
            self._build_index()
            self._save_index()

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve top-k relevant chunks for the query with metadata.
        """
        if self._vectorstore is None:
            raise RuntimeError(
                "Vectorstore not initialized. Call build_or_load_index() first."
            )

        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=k)

        results: List[RetrievedChunk] = []
        for doc, score in docs_and_scores:
            meta = doc.metadata or {}
            source_file = str(meta.get("source_file", meta.get("source", "unknown")))
            section_heading = meta.get("section_heading", "").strip()
            chunk_id = int(meta.get("chunk_id", -1))
            start_index = meta.get("start_index")

            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    score=float(score),
                    source_file=source_file,
                    section_heading=section_heading,
                    chunk_id=chunk_id,
                    start_index=start_index,
                )
            )

        return results

    # --------- INTERNAL: INDEXING PIPELINE ---------

    def _load_raw_documents(self) -> List[Document]:
        """
        Load all files from docs_dir.
        You can specialize this later (PDF, HTML, etc.).
        """
        # Basic example: treat everything as text-like using TextLoader
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.*",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
        )
        documents = loader.load()
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, adding metadata:
        - source_file (file path)
        - chunk_id
        - section_heading (first line of chunk)
        - start_index (character offset in original doc)
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        # First split using the splitter
        split_docs = splitter.split_documents(documents)

        # Enrich metadata
        enriched_docs: List[Document] = []
        for i, doc in enumerate(split_docs):
            meta = dict(doc.metadata) if doc.metadata else {}

            # Normalize source field
            raw_source = meta.get("source", "")
            meta["source_file"] = raw_source

            # Start index from splitter
            start_index = meta.get("start_index")

            # Heuristic section heading: first non-empty line of the chunk
            first_line = ""
            for line in doc.page_content.splitlines():
                stripped = line.strip()
                if stripped:
                    first_line = stripped[:120]
                    break

            meta["section_heading"] = first_line
            meta["chunk_id"] = i
            meta["start_index"] = start_index

            enriched_docs.append(
                Document(page_content=doc.page_content, metadata=meta)
            )

        return enriched_docs

    def _build_index(self) -> None:
        """
        Load, split, and index documents in FAISS (CPU).
        """
        documents = self._load_raw_documents()
        if not documents:
            raise RuntimeError(f"No documents found in {self.docs_dir}")

        chunks = self._split_documents(documents)

        self._vectorstore = FAISS.from_documents(chunks, embeddings)

    def _save_index(self) -> None:
        assert self._vectorstore is not None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_dir))

    def _load_index(self) -> None:
        self._vectorstore = FAISS.load_local(
            str(self.index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
