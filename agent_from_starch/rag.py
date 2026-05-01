import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Optional, Any, List, Dict, Tuple

class RAG:
    def __init__(self, name: Optional[str] = "rag"):
        self.chroma_client = chromadb.Client()
        self.local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chroma_client.create_collection(name=name,embedding_function=self.local_ef)

    def add(self, documents:Optional[List[any]] = None, embeddings: Optional[List[any]] = None , metadatas: Optional[List[any]]= None):
        
        if documents is None and embeddings is None:
            raise ValueError("must provide either documents, embeddings, or both")

        base = documents if documents is not None else embeddings
        n = len(base)

        def check(name, value):
            if value is not None and len(value) != n:
                raise ValueError(f"{name} must have length {n}")

        check("documents", documents)
        check("embeddings", embeddings)
        check("metadatas", metadatas)

        id = [f"id_{i}" for i in range(1, n+1)]

        self.collection.add(
            ids = id,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print("done")

    def update(self, id: Optional[List[any]] = None, documents:Optional[List[any]] = None, embeddings: Optional[List[any]] = None , metadatas: Optional[List[any]]= None):
        try:
            self.collection.update(
                ids=id,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"Update failed: {e}")
        
    def delete(self, id: Optional[List[Dict[Any,Any]]] = None, where: Optional[Tuple[List[Dict[Any, Any]], str]] = None):
        try:
            if id:
                self.collection.delete(
                    ids=id[0]
                )
            if where:
                n = len(where[0])
                if n > 1:
                    operation = f"${where[1]}"
                    self.collection.delete(
                        where={
                            operation:where[1]
                        }
                    )
                else:
                    self.collection.delete(
                        where=where[1][0]
                    )
        except Exception as e:
            print(f"Delete failed: {e}")
            
    def get_or_create_collection(self, name: str):
        try:
            self.collection = self.chroma_client.get_collection(
                name=name,
                embedding_function=self.local_ef
            )
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=name,
                embedding_function=self.local_ef
            )
