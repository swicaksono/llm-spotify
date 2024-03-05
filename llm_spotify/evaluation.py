import os
import pandas as pd
import datasets
from huggingface_hub import InferenceClient
from langchain.chat_models import ChatOpenAI
from ragatouille import RAGPretrainedModel
from transformers import pipeline
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Optional
from langchain.schema import SystemMessage
from llm_spotify.config import QA_GENERATION_MODEL, EVALUTOR_MODEL, EVALUTOR_NAME
from llm_spotify.prompt import QA_GENERATION_PROMPT, QUESTION_GROUNDNESS_CRITIQUE_PROMPT, \
      QUESTION_RELEVANCE_CRITIQUE_PROMPT, QUESTION_STANDALONE_CRITIQUE_PROMPT, EVALUATION_PROMPT
import json
import random
from tqdm import tqdm

EVALUATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

llm_client = InferenceClient(
    model=QA_GENERATION_MODEL,
    timeout=120,
)

EVAL_CHAT_MODEL = ChatOpenAI(model=EVALUTOR_MODEL, temperature=0)



def call_llm(inference_client: InferenceClient, prompt: str):
    """agents for question generation"""
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

def generate_synthetic_question(relevant_document: list[str], number_generations: int=15):
    """generate synthetic questions based on context given"""

    print(f"Generating {number_generations} QA couples...")
    
    outputs = []
    for sampled_context in tqdm(random.sample(relevant_document, number_generations)):
        # Generate QA couple
        output_QA_couple = call_llm(llm_client, QA_GENERATION_PROMPT.format(context=sampled_context))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context,
                    "question": question,
                    "answer": answer,
                }
            )
        except Exception as e:
            print (e)
            continue
    
    return outputs

def generate_critique_qa(synthentic_questions: list[dict]):
    """agent critique for the QA data"""
    for output in tqdm(synthentic_questions):
        evaluations = {
            "groundedness": call_llm(
                llm_client,
                QUESTION_GROUNDNESS_CRITIQUE_PROMPT.format(context=output["context"], question=output["question"]),
            ),
            "relevance": call_llm(
                llm_client,
                QUESTION_RELEVANCE_CRITIQUE_PROMPT.format(question=output["question"]),
            ),
            "standalone": call_llm(
                llm_client,
                QUESTION_STANDALONE_CRITIQUE_PROMPT.format(question=output["question"]),
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue
    
    return synthentic_questions

def generate_data_evaluation(contexts: list[str]):
    synthetic_questions = generate_synthetic_question(relevant_document=contexts)
    outputs = generate_critique_qa(synthentic_questions=synthetic_questions)
    generated_questions = pd.DataFrame.from_dict(outputs)
    
    # get the best relevance qa couples for data evaluation
    generated_questions = generated_questions.loc[
        (generated_questions["groundedness_score"] >= 4)
        & (generated_questions["relevance_score"] >= 4)
        & (generated_questions["standalone_score"] >= 4)
    ]

    eval_dataset = datasets.Dataset.from_pandas(generated_questions, split="train", preserve_index=False)
    return eval_dataset

def run_rag_tests(
    eval_dataset: datasets.Dataset,
    output_file: str,
    rag: function,
    llm: Optional[pipeline] = None,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = rag(question, llm, reranker=reranker)
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["context"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)
    
def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)

def evaluate_model(eval_dataset: datasets.Dataset, llm: pipeline):
    if not os.path.exists("./output"):
        os.mkdir("./output")

    for embeddings in ["jina-embeddings-v2-base-en"]:  # add other embeddings as needed
        for rerank in [True, False]:
            settings_name = f"embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}_reader-model"
            output_file_name = f"./output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=llm,
                output_file=output_file_name,
                reranker=rerank,
                verbose=False,
                test_settings=settings_name,
            )

            print("Running evaluation...")
            evaluate_answers(
                output_file_name,
                EVAL_CHAT_MODEL,
                EVALUTOR_NAME,
                EVALUATION_PROMPT_TEMPLATE,
            )