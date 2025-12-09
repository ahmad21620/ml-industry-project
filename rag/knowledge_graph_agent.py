from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph

from config import (
    DOCS_DIR,
    embeddings,
    llm,  # imported for consistency with other agents, not used yet
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


@dataclass
class KGRetrievedChunk:
    """
    Retrieved chunk from the Neo4j knowledge graph.
    Mirrors the style of `RetrievedChunk` from RAGAgent.
    """
    id: str
    text: str
    source: str
    index: int
    semantic_score: Optional[float] = None
    enhanced_score: Optional[float] = None
    context_preview: str = ""


class KnowledgeGraphAgent:
    """
    Knowledge Graph component:
    - indexes document chunks as :Chunk nodes in Neo4j (with vector embeddings)
    - performs graph-aware semantic retrieval over Chunk nodes
    """

    def __init__(
        self,
        docs_dir: Path = DOCS_DIR,
        use_gpu: bool = False,  # kept for signature symmetry; device handled in config.embeddings
    ) -> None:
        # Paths
        self.docs_dir = docs_dir

        # Neo4j connection (from config, like other globals)
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )

        # Use global embeddings from config (same as RAGAgent)
        self.embeddings = embeddings

        # Infer embedding dimension once
        test_embedding = self.embeddings.embed_query("test")
        self.embedding_dim: int = len(test_embedding)

        # Text splitter, same logic as original StreamlinedChunkGraph
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
        )

    # --------- PUBLIC API ---------

    def initialize_schema(self) -> None:
        """
        Public method to clear existing constraints and recreate indices.
        Call this once before indexing if needed.
        """
        self._clear_existing_constraints()
        self._create_indices()

    def index_pdf(self, pdf_path: str | Path) -> List[Dict]:
        """
        Index a single PDF document into Neo4j as :Chunk nodes.

        - If pdf_path is relative, it is resolved against DOCS_DIR.
        - Logic preserved from original index_pdf (chunking, embeddings, NEXT links).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_absolute():
            pdf_path = self.docs_dir / pdf_path

        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return []

        print(f"📖 Indexing PDF: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))

            # Extract text
            text_parts: List[str] = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception:
                    continue

            if not text_parts:
                print("❌ No extractable text found")
                return []

            full_text = "\n".join(text_parts)

            # Create chunks
            chunks = self.splitter.split_text(full_text)
            chunks = [
                chunk.strip()
                for chunk in chunks
                if chunk.strip() and len(chunk.strip()) > 50
            ]

            if not chunks:
                print("❌ No valid chunks after filtering")
                return []

            # Generate embeddings
            embeddings_list = self.embeddings.embed_documents(chunks)

            # Store in Neo4j
            chunk_data: List[Dict] = []
            source_name = pdf_path.name

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                chunk_id = f"{source_name}_{i}"

                self.graph.query(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.text = $text,
                        c.embedding = $embedding,
                        c.source = $source,
                        c.chunk_index = $index,
                        c.created_at = datetime()
                    """,
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "embedding": embedding.tolist()
                        if hasattr(embedding, "tolist")
                        else list(embedding),
                        "source": source_name,
                        "index": i,
                    },
                )

                chunk_data.append(
                    {
                        "id": chunk_id,
                        "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                        "source": source_name,
                    }
                )

                # Create sequential relationships
                if i > 0:
                    prev_id = f"{source_name}_{i - 1}"
                    self.graph.query(
                        """
                        MATCH (c1:Chunk {id: $prev_id})
                        MATCH (c2:Chunk {id: $current_id})
                        MERGE (c1)-[:NEXT]->(c2)
                        """,
                        {"prev_id": prev_id, "current_id": chunk_id},
                    )

            print(f"✅ Successfully indexed {len(chunk_data)} chunks")
            return chunk_data

        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            import traceback

            traceback.print_exc()
            return []
    
    def is_graph_empty(self) -> bool:
        result = self.graph.query("MATCH (c:Chunk) RETURN count(c) AS count")
        return result[0]["count"] == 0

    def retrieve(self, query: str, k: int = 7) -> List[KGRetrievedChunk]:
        """
        Retrieve top-k graph-aware relevant chunks (semantic + local context).
        This is the KG analogue of RAGAgent.retrieve.
        """
        results = self._graph_aware_search(query, k)
        return results

    def get_context_for_llm(self, query: str, k: int = 7) -> str:
        """
        Get a formatted context string for the LLM.
        This keeps your original formatting logic.
        """
        search_results = self._graph_aware_search(query, k)

        if not search_results:
            return "No relevant information found in the AWS billing documentation."

        context_parts: List[str] = []
        context_parts.append("=== AWS BILLING DOCUMENTATION CONTEXT (KG) ===")
        context_parts.append(f"Query: {query}")
        context_parts.append(f"Found {len(search_results)} relevant chunks\n")

        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"[CHUNK {i} - Score: {result.enhanced_score or 0:.3f}]"
            )
            context_parts.append(f"Source: {result.source}")
            context_parts.append(f"Text: {result.text}")

            if result.context_preview:
                context_parts.append(f"Related context: {result.context_preview}")

            context_parts.append("")  # Empty line between chunks

        return "\n".join(context_parts)

    def create_semantic_relationships(
        self,
        similarity_threshold: float = 0.7,
    ) -> None:
        """
        Create semantic relationships between similar chunks.
        Logic preserved from your original create_semantic_relationships.
        """
        try:
            result = self.graph.query(
                """
                MATCH (c1:Chunk)
                WHERE c1.embedding IS NOT NULL
                WITH c1
                MATCH (c2:Chunk)
                WHERE c2.embedding IS NOT NULL 
                  AND id(c1) < id(c2)
                  AND c1.source = c2.source
                WITH c1, c2, gds.similarity.cosine(c1.embedding, c2.embedding) AS similarity
                WHERE similarity >= $threshold
                MERGE (c1)-[r:SEMANTIC_SIMILAR]->(c2)
                SET r.similarity = similarity,
                    r.created_at = datetime()
                RETURN count(r) as relationships_created
                """,
                {"threshold": similarity_threshold},
            )

            if result:
                print(
                    f"✅ Created {result[0]['relationships_created']} semantic relationships"
                )

        except Exception as e:
            print(f"❌ Error creating semantic relationships: {e}")

    # --------- INTERNAL HELPERS ---------

    def _clear_existing_constraints(self) -> None:
        """
        Remove existing constraints on :Chunk if they exist.
        Same logic as your original clear_existing_constraints, but internal.
        """
        try:
            constraints = self.graph.query(
                """
                SHOW CONSTRAINTS
                YIELD name, labelsOrTypes, properties
                WHERE labelsOrTypes = ['Chunk']
                RETURN name, properties
                """
            )

            for constraint in constraints:
                name = constraint.get("name")
                if name:
                    print(f"   Dropping constraint: {name}")
                    self.graph.query(f"DROP CONSTRAINT {name} IF EXISTS")

            print("✅ Existing constraints cleared")
        except Exception as e:
            print(f"⚠️  Error clearing constraints: {e}")

    def _create_indices(self) -> None:
        """
        Create all necessary indices.
        Logic preserved from your original _create_indices.
        """
        try:
            # Vector index for semantic search
            self.graph.query(
                f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON c.embedding
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {self.embedding_dim},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """
            )

            # Index for sequential traversal
            self.graph.query(
                """
                CREATE INDEX chunk_source_index IF NOT EXISTS 
                FOR (c:Chunk) ON (c.source, c.chunk_index)
                """
            )

            print("✅ All indices created")
        except Exception as e:
            print(f"⚠️  Index creation note: {e}")

    def _graph_aware_search(self, query: str, k: int = 7) -> List[KGRetrievedChunk]:
        """
        PRIMARY SEARCH METHOD (graph-aware).
        Logic is the same as your original graph_aware_search,
        but mapped into KGRetrievedChunk dataclasses.
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)

            cypher = """
            // Find semantically similar chunks
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, $embedding) as semantic_score
            
            // Get sequential context (previous and next chunks)
            OPTIONAL MATCH (c)-[:NEXT*0..2]-(context:Chunk)
            WHERE context.source = c.source
            
            WITH c, semantic_score, 
                 collect(DISTINCT context.text) as context_texts,
                 count(DISTINCT context) as context_count
            
            // Calculate enhanced score with context bonus
            WITH c, semantic_score, context_texts, context_count,
                 semantic_score * (1.0 + 0.15 * context_count) as enhanced_score
            
            RETURN c.id as id,
                   c.text as text,
                   c.source as source,
                   c.chunk_index as index,
                   semantic_score,
                   enhanced_score,
                   context_texts[0..3] as context_preview,
                   context_count
            ORDER BY enhanced_score DESC
            LIMIT $k
            """

            raw_results = self.graph.query(
                cypher,
                {
                    "embedding": query_embedding,
                    "k": k,
                },
            )

            results: List[KGRetrievedChunk] = []
            for row in raw_results:
                previews = row.get("context_preview", []) or []
                preview_str = " ... ".join(previews) if previews else ""

                results.append(
                    KGRetrievedChunk(
                        id=row.get("id", ""),
                        text=row.get("text", ""),
                        source=row.get("source", "Unknown"),
                        index=row.get("index", -1),
                        semantic_score=row.get("semantic_score"),
                        enhanced_score=row.get("enhanced_score"),
                        context_preview=preview_str,
                    )
                )

            print(f"🔍 Graph-aware search found {len(results)} results")
            return results

        except Exception as e:
            print(f"❌ Graph-aware search failed: {e}")
            return self._fallback_search(query, k)

    def _fallback_search(self, query: str, k: int = 7) -> List[KGRetrievedChunk]:
        """
        Simple fallback search using only cosine similarity.
        Logic preserved, but returns KGRetrievedChunk.
        """
        query_embedding = self.embeddings.embed_query(query)

        raw_results = self.graph.query(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, gds.similarity.cosine(c.embedding, $embedding) as similarity
            RETURN c.id as id, c.text as text, c.source as source,
                   c.chunk_index as index, similarity
            ORDER BY similarity DESC
            LIMIT $k
            """,
            {"embedding": query_embedding, "k": k},
        )

        results: List[KGRetrievedChunk] = []
        for row in raw_results:
            results.append(
                KGRetrievedChunk(
                    id=row.get("id", ""),
                    text=row.get("text", ""),
                    source=row.get("source", "Unknown"),
                    index=row.get("index", -1),
                    semantic_score=row.get("similarity"),
                    enhanced_score=row.get("similarity"),
                    context_preview="",
                )
            )

        return results
