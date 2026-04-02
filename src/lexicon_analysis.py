# Fichier: src/lexicon_analysis.py
import pandas as pd
import re

def calculate_unemployment_ratio(df_para, keywords):
    """
    Détecte la présence de mots-clés dans chaque paragraphe et calcule
    le ratio d'attention portée au chômage par document et par département.
    """
    # 1. Création d'une expression régulière (regex) pour chercher les mots
    # Le \b permet de chercher des mots entiers (ou préfixes si on adapte)
    # L'utilisation du | agit comme un "OU" (chômage OU emploi OU...)
    pattern = r'\b(' + '|'.join(keywords) + r')\b'
    
    # 2. Détection (Renvoie True si un des mots est présent, False sinon)
    df_para['parle_de_chomage'] = df_para['raw_paragraph'].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
    
    # 3. Agrégation par Document (Profession de foi)
    # On calcule la moyenne de True/False, ce qui donne exactement le % de paragraphes
    doc_stats = df_para.groupby(['filename', 'department_code']).agg(
        total_paragraphes=('raw_paragraph', 'count'),
        paragraphes_chomage=('parle_de_chomage', 'sum'),
        ratio_chomage_doc=('parle_de_chomage', 'mean') # C'est notre fameux ratio (de 0 à 1)
    ).reset_index()
    
    # 4. Agrégation par Département
    # On fait la moyenne des ratios des candidats de ce département
    dept_stats = doc_stats.groupby('department_code').agg(
        nb_candidats=('filename', 'count'),
        ratio_moyen_dept=('ratio_chomage_doc', 'mean')
    ).reset_index()
    
    # On multiplie par 100 pour avoir un pourcentage plus lisible
    doc_stats['ratio_chomage_doc'] = doc_stats['ratio_chomage_doc'] * 100
    dept_stats['ratio_moyen_dept'] = dept_stats['ratio_moyen_dept'] * 100
    
    return doc_stats, dept_stats