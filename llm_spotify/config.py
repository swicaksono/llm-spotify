import os
from dotenv import load_dotenv

load_dotenv()

DATAPATH = os.getenv('DATAPATH')
SAMPLING_DATA = True if str(os.getenv('SAMPLING_DATA')) == "true" else False
MODEL_EMBEDDING = os.getenv('MODEL_EMBEDDING')
READER_MODEL = os.getenv('READER_MODEL')
MODEL_QUANTIZED = True if str(os.getenv('MODEL_QUANTIZED')) == "true" else False
RERANK_MODEL = os.getenv('RERANK_MODEL')
QA_GENERATION_MODEL = os.getenv('QA_GENERATION_MODEL')
EVALUTOR_MODEL = os.getenv('EVALUTOR_MODEL')
EVALUTOR_NAME = os.getenv('EVALUTOR_NAME')

MAX_NUM_PENDING_TASKS = int(os.getenv('MAX_NUM_PENDING_TASKS'))
MAX_GPUS = int(str(os.getenv('MAX_GPUS')))
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY'))

QDRANT_VECTOR_SIZE = os.getenv('QDRANT_VECTOR_SIZE')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME')
QDRANT_VECTOR_DISTANCE = os.getenv('QDRANT_VECTOR_DISTANCE')

