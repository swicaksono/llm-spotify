from transformers import Pipeline
from ragatouille import RAGPretrainedModel
from typing import Tuple, List, Dict, Optional
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from llm_spotify.rerank import RERANKER
from llm_spotify.indexing import semantic_search, build_client
from llm_spotify.prompt import PROMP_IN_CHAT_FORMAT
from llm_spotify.reader import MODEL_READER_TOKENIZER, READER_LLM

RAG_PROMPT_TEMPLATE = MODEL_READER_TOKENIZER.apply_chat_template(
    PROMP_IN_CHAT_FORMAT, tokenize=False, add_generation_prompt=True
)

QDRANT_CLIENT = build_client()
EMBEDDING_MODEL = HuggingFaceEmbeddings(
                        model_name="jinaai/jina-embeddings-v2-base-en",
                        model_kwargs={"device": "cuda"},
                        encode_kwargs={"device": "cuda", "batch_size": 100}
)

def answer_with_rag(
    question: str,
    llm: Pipeline = READER_LLM,
    reranker: Optional[RAGPretrainedModel] = RERANKER,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 15,
) -> Tuple[str, List[Dict]]:

    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = semantic_search(query=question, client=QDRANT_CLIENT, embedding_model=EMBEDDING_MODEL, top_k=num_retrieved_docs)
    relevant_docs = [doc.payload['text'] for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs