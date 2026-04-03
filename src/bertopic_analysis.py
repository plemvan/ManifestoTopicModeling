# File: src/bertopic_analysis.py
import pandas as pd
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Téléchargement silencieux de la ponctuation NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def robust_cleaning(value):
    """Converts list-like strings or lists into clean sentences."""
    if isinstance(value, list):
        return " ".join(value)
    elif isinstance(value, str):
        return value.replace("[", "").replace("]", "").replace("'", "").replace(",", "")
    return ""

def run_bertopic(df_para, target_themes, threshold=0.4):
    """
    Fixed version with automatic column detection.
    Intègre discrètement l'approche "Scanner d'évocation" (Chunking + Roll-up)
    tout en gardant une compatibilité stricte avec la structure d'origine.
    """
    # --- FIX: Column Detection ---
    if 'mots_propres' in df_para.columns:
        col_to_use = 'mots_propres'
    elif 'raw_paragraph' in df_para.columns:
        col_to_use = 'raw_paragraph'
    else:
        raise KeyError(f"Neither 'mots_propres' nor 'raw_paragraph' found in DataFrame. Available columns: {df_para.columns.tolist()}")

    print(f"Using column: '{col_to_use}' for analysis (Hybrid Engine Active).")
    
    # 1. Prepare base DataFrame and tracking IDs
    df_result = df_para.copy()
    df_result['para_id'] = range(len(df_result))
    
    # 2. Chunking : Découpage en phrases
    sentences_data = []
    for _, row in df_result.iterrows():
        # Application du robust_cleaning à la volée avant découpage
        text = robust_cleaning(row[col_to_use])
        sents = nltk.sent_tokenize(text, language='french')
        
        for sent in sents:
            if len(sent) > 40: # Filtre anti-bruit pour les phrases courtes
                sentences_data.append({
                    'para_id': row['para_id'],
                    'sentence': sent
                })
                
    df_sentences = pd.DataFrame(sentences_data).dropna(subset=['sentence'])
    
    # 3. Configure Model (CamemBERT + Stop words)
    try:
        from spacy.lang.fr.stop_words import STOP_WORDS as fr_stops
        vectorizer_model = CountVectorizer(stop_words=list(fr_stops))
    except ImportError:
        vectorizer_model = CountVectorizer(stop_words=["le", "la", "les", "de", "des", "un", "une", "et", "ou", "pour", "dans", "qui", "que"])

    sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")

    topic_model = BERTopic(
        embedding_model=sentence_model,
        zeroshot_topic_list=target_themes,
        zeroshot_min_similarity=threshold, 
        vectorizer_model=vectorizer_model,
        language="french",
        verbose=False 
    )

    # 4. Fit and Transform on sentences
    docs = df_sentences['sentence'].tolist()
    topics, _ = topic_model.fit_transform(docs)
    df_sentences['topic_bertopic_sent'] = topics

    # 5. Roll-up to Paragraphs (Reconstruction)
    # Si une des phrases vaut 0, le paragraphe entier vaut 0. Sinon -1.
    df_sentences['is_target'] = (df_sentences['topic_bertopic_sent'] == 0).astype(int)
    para_rollup = df_sentences.groupby('para_id')['is_target'].max().reset_index()
    
    # Intégration du résultat binaire dans df_result
    df_result = pd.merge(df_result, para_rollup, on='para_id', how='left')
    df_result['is_target'] = df_result['is_target'].fillna(0)
    
    # On recrée la colonne 'topic_bertopic' attendue par la structure d'origine
    df_result['topic_bertopic'] = df_result['is_target'].apply(lambda x: 0 if x == 1 else -1)
    
    # Nettoyage des colonnes temporaires
    df_result = df_result.drop(columns=['para_id', 'is_target'])

    # 6. Process Results (Strictement identique à votre code d'origine)
    # Create the percentage weight per document
    # Using 'filename' as the index for the aggregation
    theme_counts = df_result.groupby(['filename', 'topic_bertopic']).size().unstack(fill_value=0)
    theme_percentages = theme_counts.div(theme_counts.sum(axis=1), axis=0).reset_index()

    return topic_model, df_result, theme_percentages