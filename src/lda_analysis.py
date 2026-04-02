from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import numpy as np

def run_lda_model(df, n_topics=5, no_top_words=10):
    """
    Fits an LDA model on the 'raw_paragraph' column of a dataframe.
    Returns a dictionary of topics and their most representative words.
    """
    # 1. Setup French Stopwords
    try:
        stop_words = stopwords.words('french')
    except:
        nltk.download('stopwords')
        stop_words = stopwords.words('french')
    
    # Add common political/OCR noise words to ignore
    custom_noise = ['plus', 'cette', 'fait', 'tout', 'tous', 'si', 'être', 'faire', 'comme', 'dans', 'pour']
    stop_words.extend(custom_noise)

    # 2. Vectorization: Convert text into a matrix of word counts
    # We ignore words that appear in > 90% or < 5 paragraphs to reduce noise
    vectorizer = CountVectorizer(
        max_df=0.9, 
        min_df=5, 
        stop_words=stop_words
    )
    
    # Handle potential empty dataframes
    if df.empty:
        return {"Error": "Empty dataframe provided"}

    dtm = vectorizer.fit_transform(df['raw_paragraph'])

    # 3. LDA Model Training
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        learning_method='online'
    )
    lda.fit(dtm)
    
    # 4. Extract Top Words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics_summary = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        # Sort words by importance and take the top N
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_summary[f"Topic {topic_idx + 1}"] = " | ".join(top_words)
        
    return topics_summary

# File: src/lda_analysis.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import pandas as pd

def run_seeded_lda(df, n_topics=5, no_top_words=10, seed_words=None):
    """
    Fits an LDA model by oversampling rows containing seed words 
    to force the emergence of a specific theme.
    """
    # 1. Setup Stopwords
    try:
        stop_words = stopwords.words('french')
    except:
        nltk.download('stopwords')
        stop_words = stopwords.words('french')
    
    custom_noise = ['plus', 'cette', 'fait', 'tout', 'tous', 'si', 'être', 'faire', 'comme', 'dans', 'pour', 'aux']
    stop_words.extend(custom_noise)

    # 2. SEEDING STRATEGY: Oversampling
    # If seed words are provided, we duplicate paragraphs containing them to 'force' the topic
    df_boosted = df.copy()
    if seed_words:
        # Pattern to find any of the seed words (case insensitive)
        pattern = '|'.join(seed_words)
        mask = df_boosted['raw_paragraph'].str.contains(pattern, case=False, na=False)
        df_seeds = df_boosted[mask]
        
        # We duplicate these paragraphs 10 times to make them statistically significant
        df_boosted = pd.concat([df_boosted] + [df_seeds] * 10, ignore_index=True)

    # 3. Vectorization
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words=stop_words)
    dtm = vectorizer.fit_transform(df_boosted['raw_paragraph'])
    feature_names = vectorizer.get_feature_names_out()

    # 4. LDA Training (Standard symmetric prior to avoid InvalidParameterError)
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        learning_method='online'
    )
    lda.fit(dtm)
    
    # 5. Extract Topics
    topics_summary = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-no_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        
        # Identify if this topic is the one influenced by our seeds
        is_seed_topic = any(word in top_words for word in (seed_words if seed_words else []))
        label = f"Topic {topic_idx + 1} (Thematic)" if is_seed_topic else f"Topic {topic_idx + 1}"
        
        topics_summary[label] = " | ".join(top_words)
        
    return topics_summary