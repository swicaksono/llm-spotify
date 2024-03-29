from transformers import pipeline
import torch
from llm_spotify.config import READER_MODEL, MODEL_QUANTIZED
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

## load tokenizer
MODEL_READER_TOKENIZER = AutoTokenizer.from_pretrained(READER_MODEL)

## load model reader
if MODEL_QUANTIZED:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    MODEL_READER = AutoModelForCausalLM.from_pretrained(READER_MODEL, quantization_config=bnb_config)
else:
    MODEL_READER = AutoModelForCausalLM.from_pretrained(READER_MODEL)

READER_LLM = pipeline(
    model=MODEL_READER,
    tokenizer=MODEL_READER_TOKENIZER,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)