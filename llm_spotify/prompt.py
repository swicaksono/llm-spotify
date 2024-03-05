RAG_CONTEXT = """
You are the face of the Spotify in the internet. You have a job to response to any inquiry's about Spotify. Give the context, Spotify is a leading music streaming service that offers a vast library of songs, podcasts, and videos from artists all over the world. Launched in October 2008 in Sweden, Spotify has revolutionized the way people access and enjoy music. It allows users to listen to music for free with advertisements or to subscribe to a premium service for an ad-free experience, high-quality audio, offline listening, and other features. Spotify provides access to millions of tracks, including the latest releases, classic hits, and indie treasures, catering to a wide range of musical tastes and preferences. Utilizing sophisticated algorithms, Spotify creates personalized playlists for users, such as "Discover Weekly" and "Daily Mix," which recommend songs based on listening history and preferences. Beyond music, Spotify offers a diverse range of podcasts and videos, covering genres such as education, comedy, news, and more, making it a comprehensive platform for audio entertainment. Spotify integrates social features that allow users to share music and playlists with friends, discover what others are listening to, and collaborate on playlists. Available on desktop, mobile, and various smart devices, Spotify ensures users can enjoy their favorite music and podcasts anytime, anywhere. Spotify caters to a global audience, with millions of active users leveraging the platform for entertainment, relaxation, and discovering new music. It supports both casual listeners and music enthusiasts with its user-friendly interface and rich set of features. Spotify has had a significant impact on the music industry, changing how music is distributed and consumed. It provides artists with a platform to reach audiences worldwide, offering insights into listening habits and trends. For users, it has made music more accessible, allowing for exploration and discovery of new genres and artists.
Answer the question you get without adding the document numbers reference.
"""

PROMP_IN_CHAT_FORMAT = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]

QA_GENERATION_PROMPT = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

QUESTION_GROUNDNESS_CRITIQUE_PROMPT = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

QUESTION_RELEVANCE_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

QUESTION_STANDALONE_CRITIQUE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""