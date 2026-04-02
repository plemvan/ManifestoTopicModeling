# File: src/bertopic_analysis.py
from bertopic import BERTopic
import pandas as pd

def robust_cleaning(value):
    """Converts list-like strings or lists into clean sentences."""
    if isinstance(value, list):
        return " ".join(value)
    elif isinstance(value, str):
        return value.replace("[", "").replace("]", "").replace("'", "").replace(",", "")
    return ""

def run_winning_zeroshot(df_para, target_themes):
    """
    Fixed version with automatic column detection.
    """
    # --- FIX: Column Detection ---
    if 'mots_propres' in df_para.columns:
        col_to_use = 'mots_propres'
    elif 'raw_paragraph' in df_para.columns:
        col_to_use = 'raw_paragraph'
    else:
        raise KeyError(f"Neither 'mots_propres' nor 'raw_paragraph' found in DataFrame. Available columns: {df_para.columns.tolist()}")

    print(f"Using column: '{col_to_use}' for analysis.")
    
    # 1. Prepare docs
    docs = df_para[col_to_use].apply(robust_cleaning).tolist()

    # 2. Configure Model
    topic_model = BERTopic(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        zeroshot_topic_list=target_themes,
        zeroshot_min_similarity=0.80, 
        language="french",
        verbose=True
    )

    # 3. Fit and Transform
    topics, _ = topic_model.fit_transform(docs)
    
    # 4. Process Results
    df_result = df_para.copy()
    df_result['topic_bertopic'] = topics

    # Create the percentage weight per document
    # Using 'filename' as the index for the aggregation
    theme_counts = df_result.groupby(['filename', 'topic_bertopic']).size().unstack(fill_value=0)
    theme_percentages = theme_counts.div(theme_counts.sum(axis=1), axis=0).reset_index()

    return topic_model, df_result, theme_percentages