import pandas as pd
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llm_spotify.config import DATAPATH

load_dotenv()

def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATAPATH)
    data['review_id'] = data['review_id'].astype("string")
    data['review_text'] = data['review_text'].astype("string")

    data.dropna(subset=['review_text', 'review_id', 'review_rating'], inplace=True)
    
    return data

def preprocess(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    def subpreprocess(texts):
        """Clean and preprocess a single review text."""
        # Lowercasing
        texts = texts.lower()

        # Remove HTML tags
        texts = BeautifulSoup(texts, "html.parser").get_text()
        
        # Remove punctuation/special characters - keep only words and spaces
        texts = re.sub(r'[^a-z\s]', '', texts)
        return texts
    
    df[column_name] = df[column_name].apply(subpreprocess)
    
    return df

def sampling(data: pd.DataFrame) -> pd.DataFrame:
    sampled_reviews = []
    
    # define the specific ranges and sample sizes
    special_start_range = 50
    special_end_range = 300
    special_sample_size = 3000  # larger sample size for the 50 to 300 character range
    default_sample_size = 1000  # default sample size for lengths beyond 300 characters
    max_length = data['review_rating_length'].max()

    # sample from 50 to 300 characters in increments of 10, with 3000 samples each
    for start in range(special_start_range, special_end_range + 1, 10):
        end = start + 10
        segment = data[(data['review_rating_length'] > start) & (data['review_rating_length'] <= end)]
        if len(segment) >= special_sample_size:
            samples = segment.sample(n=special_sample_size, random_state=42)
        else:
            samples = segment
        sampled_reviews.append(samples)

    # continue sampling beyond 300 characters, up to the maximum length, with 1000 samples each
    for start in range(special_end_range + 10, max_length + 1, 10):
        end = start + 10
        segment = data[(data['review_rating_length'] > start) & (data['review_rating_length'] <= end)]
        if len(segment) >= default_sample_size:
            samples = segment.sample(n=default_sample_size, random_state=42)
        else:
            samples = segment
        sampled_reviews.append(samples)

    # combine all sampled segments into a single DataFrame
    sampled_df = pd.concat(sampled_reviews)

    return sampled_df

def extract_data(sample: bool = True):
    data = load_data()
    data = preprocess(data, column_name='review_text')
    if sample:
        return sampling(data=data)
    else:
        return data
