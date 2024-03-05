from qdrant_client import QdrantClient
from typing import Union
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llm_spotify.config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_DISTANCE, QDRANT_VECTOR_SIZE
from qdrant_client.models import PointStruct
from qdrant_client.http import models
from llm_spotify.embeddings import extract_embeddings
import numpy as np

def build_client():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("Connected to Qdrant server.")
    
    return client

def build_collection(client: QdrantClient, collection_name: str = QDRANT_COLLECTION_NAME):
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print (collection_info)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=QDRANT_VECTOR_SIZE, distance=QDRANT_VECTOR_DISTANCE),
        )
        print (f"succesfully create collection {collection_name}")

def upsert_data(client:QdrantClient, data: list[dict], collection_name: str):
    try:
        for item in data:
            client.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=item['id'], payload=item['payload'], vector=item['vector'])]
            )
        return
    except Exception as e:
        print(f"ERROR: {e}")

def insert_embeddings(client: QdrantClient, collection_name: str = QDRANT_COLLECTION_NAME):
    doc_embeddings = extract_embeddings()
    batch_data = {"points": [{"id": idx+1,
                "vector": review["embeddings"][0],
                "payload": {"review_id": review["id"], 
                            "text": review["text"],
                            "rating": review["rating"]}     
            } for idx, review in enumerate(doc_embeddings)]}
    
    upsert_data(client=client, data=batch_data["points"], collection_name=QDRANT_COLLECTION_NAME)
    
    return

def semantic_search(query: str, client: QdrantClient, 
                    embedding_model: Union[OpenAIEmbeddings, HuggingFaceEmbeddings], 
                    collection_name: str,
                    top_k:int=20,
                    ):
    
    vector = np.array(embedding_model.embed_query(query))
    
    semantic_context = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k)
    
    return semantic_context