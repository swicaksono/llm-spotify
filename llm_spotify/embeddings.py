from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import ray

from llm_spotify.data_extraction import extract_data
from llm_spotify.config import MODEL_EMBEDDING, MAX_NUM_PENDING_TASKS, \
                                MAX_CONCURRENCY, MAX_GPUS, SAMPLING_DATA

ray.init()
load_dotenv()

@ray.remote
class EmbedModels:
    def __init__(self, model_name=MODEL_EMBEDDING):
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"device": "cuda", 
                             "batch_size": 100})

    def get_embedding(self, batch):
        embeddings = self.embedding_model.embed_documents([batch["review_text"]])
        return {"id": batch["review_id"], "text": batch["review_text"], "rating": batch["review_rating"], "embeddings": 
embeddings}

def load_sample_data():
    sampled_data = extract_data(sample=SAMPLING_DATA)

    return sampled_data

def extract_embeddings():
    data = load_sample_data()
    embeddingmodel = EmbedModels.options(num_gpus=MAX_GPUS, max_concurrency=MAX_CONCURRENCY).remote()
    
    result_refs = []
    doc_embeddings = []
    for _, row in data.iterrows():
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            result = ray.get(ready_refs)
            doc_embeddings.extend(result)

        result_refs.append(embeddingmodel.get_embedding.remote(row))

    last_result = ray.get(result_refs)
    doc_embeddings.extend(last_result)

    return doc_embeddings