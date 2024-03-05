from ragatouille import RAGPretrainedModel
from llm_spotify.config import RERANK_MODEL

RERANKER = RAGPretrainedModel.from_pretrained(RERANK_MODEL)