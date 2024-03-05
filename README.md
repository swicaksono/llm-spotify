# LLM-Spotify

This document outlines the workflow for developing an internal Q&A tool capable of extracting meaningful information from the Google Store reviews dataset for the Spotify music streaming application. The tool leverages a Language Model (LM) and Retrieval-Augmented Generation (RAG) approach, utilizing a variety of tools and models.

## Tools Used:

1. LangChain
2. Hugging Face Models:
   - Embedding Model: `jinaai/jina-embeddings-v2-base-en`
   - Reader Model: `HuggingFaceH4/zephyr-7b-beta`
   - Re-ranking Model: `colbert-ir/colbertv2.0`
   - QA Generation Model: `mistralai/Mixtral-8x7B-Instruct-v0.1`
3. OpenAI
4. Qdrant

## Workflow Overview:
![RAG Workflow](/assets/rag-workflow.png)

### 1. Data Processing

In addressing the challenge of handling a vast dataset of 3 million Google Store reviews for Spotify, a strategic approach was essential for ensuring computational efficiency while retaining data diversity.

#### Steps:

- **Character Length Segmentation**: Reviews were categorized based on their character count, segmented into intervals of 10 characters. This segmentation aimed to ensure a systematic and diverse sampling process.
- **Strategic Sampling**: For each character length bucket, 1000 reviews were randomly sampled. If a bucket had fewer than 1000 reviews, all were included to ensure comprehensive representation.
- **Sampled Dataset Construction**: The sampled reviews across all buckets were aggregated to form a representative dataset, mirroring the original dataset's diversity.
- **Review and Adjustment**: The distribution of the sampled dataset was reviewed and adjusted as necessary to accurately reflect the original dataset's diversity, ensuring the development of a robust model.

### 2. Embedding

Using the `jinaai/jina-embeddings-v2-base-en` model, review texts were converted into dense vector representations (768-dimensional vectors), creating a knowledge base where each vector represents a piece of information from the reviews.

### 3. Indexing

The generated embeddings were stored in a Qdrant cloud server, facilitating efficient retrieval for the Q&A tool.

### 4. RAG Workflow

The RAG workflow involved several models and steps to generate answers:

- **Reader Model Pipeline**: Utilized `HuggingFaceH4/zephyr-7b-beta` to retrieve relevant documents based on the question input.
- **Re-ranking**: To refine the quality of retrieved documents, initially, a larger number of documents were retrieved and then re-ranked using `colbert-ir/colbertv2.0`, a cross-encoder model, ensuring only the top_k documents were selected for answer generation.
- **QA Synthetic Generation Dataset**: Employed `mistralai/Mixtral-8x7B-Instruct-v0.1` to generate a synthetic dataset of questions and associated contexts.

### 5. Evaluation: Evaluating RAG Performance

Evaluating the RAG system's performance is critical due to the significant impact of various components on overall effectiveness. Our evaluation pipeline includes:

- **Evaluation Dataset**: A set of question-answer couples (QA couples) is required for testing the system's accuracy.
- **Evaluator**: An evaluation mechanism to measure the system's accuracy on the provided dataset.

To enhance the evaluation process, I am using `mistralai/Mixtral-8x7B-Instruct-v0.1` for creating agents capable of question generation. Additionally, critique agents are built to perform a quality check before validating these questions based on three criteria: Groundedness, Relevance, and Standalone. The benchmarking efforts aim to optimize the RAG system by adjusting factors such as embedding size, model selection, re-ranking efficiency, and data size.

## Example Questions and Generated Answers

Below are the captured images of the generated responses based on example questions. These visuals help illustrate the effectiveness and precision of our RAG system in addressing user queries.

#### Question 1: 
![Generated Response Example 1](/assets/sample_generated_response1.png)
"What are the specific features or aspects that users appreciate the most in our application?"


**Generated Answer 1:**
Based on the provided context, some of the features that users appreciate the most in the Spotify application include:

1. Ability to listen to a wide variety of songs from different genres (Document 3)
2. Convenience of being able to listen to music at any time (Document 1)
3. User-friendly interface (Documents 0 and 2)
4. Option to clear recently listened to (Document 2)
5. Ability to remove a song from a playlist (Document 2)
6. Reduction in advertisements (Document 3)
7. Stability and reliability of the app (Documents 4, 6, and 8)
8. Premium features such as unlimited skips and no ads (Document 5)
9. Display of currently playing song (Document 9)

#### Question 2: 
![Generated Response Example 2](/assets/sample_generated_response2.png)
"In comparison to our application, which music streaming platform are users most likely to compare ours with?"

**Generated Answer 2:** Based on the context provided, users are most likely to compare Spotify with other music streaming platforms such as Apple Music (Document 0), Pandora, SoundCloud, Deezer, Tidal, and Eslap (Document 14). However, the majority of complaints and negative feedback in the extracts seem to be directed towards Spotify, indicating that users may be more inclined to compare our application with Spotify specifically. Therefore, it's safe to say that Spotify is the primary music streaming platform that users are most likely to compare ours with.

#### Question 3: 
![Generated Response Example 3](/assets/sample_generated_response3.png)
"What are the primary reasons users express dissatisfaction with Spotify?"

**Generated Answer 3:** Based on the extracts provided, the primary reasons users express dissatisfaction with Spotify are:

1. Abundance of ads: Many users complain about the excessive number of ads, with some stating that they are experiencing more ads than actual music. Some users find the frequency of ads to be intolerable, with one user suggesting that it's almost forcing them to buy premium.

2. Connection issues: Several users report connectivity problems, with Spotify insisting that there is no internet connection despite having an active connection.

3. UI/UX issues: Some users criticize the desktop app's UI, describing it as "horrendous." Others suggest improvements to the app's visuals, such as adding more videos and movie-like content.

4. Inability to control playback: Multiple users complain about the lack of control over playback, particularly the inability to skip ahead to their desired song in a playlist or return to a specific point in a track.

5. Premium features: While some users acknowledge the app's strengths, others feel that premium features are lacking, making the app unusable without a subscription. These users criticize the app for charging for basic functions and suggest that the number of ads given to non-premium users is excessive.

Overall, these issues seem to stem from a combination of technical difficulties, UI/UX shortcomings, and pricing concerns.

#### Question 4: 
![Generated Response Example 4](/assets/sample_generated_response4.png)
"Can you identify emerging trends or patterns in recent user reviews that may impact our product strategy?"

**Generated Answer 4:** Yes, based on the recent user reviews provided, some emerging trends and patterns that may impact the product strategy include:

1. Increasing complaints about bugs and glitches, particularly with regards to playback and syncing issues. Users are reporting frequent crashes, failure to recognize locally stored music, and inconsistencies between device and web versions of the app. This suggests a need for improved stability and consistency across platforms.

2. Frustration with changes to the user interface and feature set. Some users are expressing dissatisfaction with the removal of certain functions, such as the heart icon for liking songs, and the introduction of paid features that were previously available for free. This highlights the importance of balancing innovation with user preferences and needs.

3. Criticism of intrusive advertising and promotional content. Several users have complained about unwanted pop-ups and sponsored content, as well as the forced display of artist images and promotional messages. This suggests a need for more options for users to customize their experience and control the types of content they see.

4. Concerns about pricing and value for money. Some users are questioning the cost of premium subscriptions, particularly in light of the increasing number of free alternatives and competing services. This underscores the importance of offering competitive pricing and clear benefits to justify the cost of premium membership.

5. Demands for better customer support and communication. Many users are struggling to resolve technical issues and account problems, and are frustrated by the lack of responsiveness and resources available through official channels. This highlights the need for improved customer service and communication strategies to address user concerns and build trust.

Overall, these trends suggest a need for greater attention to user feedback and preferences, as well as a focus on improving stability, customization, and value for money. By addressing these concerns, the company can improve user satisfaction and loyalty, and stay ahead of competitors in a crowded market.
