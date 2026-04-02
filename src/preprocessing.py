import pandas as pd
from pathlib import Path

def load_and_split_corpus(folder_path):
    """
    Loads text files from a directory, extracts metadata,
    and directly splits the raw text into robust paragraphs.
    """
    documents = []
    path = Path(folder_path)
    
    # --- STEP 1: Load the full documents ---
    for file in path.glob("*.txt"):
        try:
            raw_text = file.read_text(encoding='utf-8', errors='ignore')
            
            # Extract metadata (Year and Department) from the filename
            parts = file.name.split('_')
            if len(parts) > 4:
                year = parts[2]
                dept = parts[4].lstrip('0') if parts[4].startswith('0') and len(parts[4]) > 1 else parts[4]
            else:
                year, dept = "Unknown", "Unknown"
                
            documents.append({
                "filename": file.name, 
                "year": year,
                "department_code": dept,
                "raw_text": raw_text
            })
        except Exception as e:
            pass # Silently ignore unreadable files
            
    df = pd.DataFrame(documents)
    
    # Safety check if the folder is empty
    if df.empty:
        print(f"⚠️ No files found in {folder_path}")
        return df 
        
    # --- STEP 2: Split into paragraphs ---
    # Split the text at every line break
    df['paragraph_list'] = df['raw_text'].str.split(r'\n+')
    
    # Explode the list to create one row per paragraph
    df_para = df.explode('paragraph_list').reset_index(drop=True)
    df_para = df_para.rename(columns={'paragraph_list': 'raw_paragraph'})
    
    # --- STEP 3: Clean and filter out OCR noise ---
    df_para['raw_paragraph'] = df_para['raw_paragraph'].astype(str).str.strip()
    
    # Drop OCR noise (lines that are too short to be meaningful sentences)
    df_para = df_para[df_para['raw_paragraph'].str.len() > 50].reset_index(drop=True)
    
    return df_para

def get_descriptive_stats(df_para, df_insee, corpus_year, insee_year):
    """
    Calculates basic statistics for a specific election year, 
    allowing a proxy year for the INSEE macroeconomic data.
    """
    # 1. Document Stats
    num_docs = df_para['filename'].nunique()
    num_paras = len(df_para)
    avg_para_per_doc = num_paras / num_docs if num_docs > 0 else 0
    
    # 2. Geographic Coverage
    docs_per_dept = df_para.groupby('department_code')['filename'].nunique().describe()
    
    # 3. INSEE Unemployment Stats using the proxy year
    cols = [f'T1_{insee_year}', f'T2_{insee_year}', f'T3_{insee_year}', f'T4_{insee_year}']
    
    # Safely calculate the mean if columns exist
    if all(col in df_insee.columns for col in cols):
        insee_avg_unemployment = df_insee[cols].mean().mean()
        insee_year_avg = df_insee[cols].mean(axis=1).describe()
    else:
        insee_avg_unemployment = float('nan')
        insee_year_avg = None

    stats = {
        "Election Year": corpus_year,
        "INSEE Data Year": insee_year,
        "Total Documents": num_docs,
        "Total Paragraphs": num_paras,
        "Depts Covered": f"{len(df_para['department_code'].unique())} departments",
        "Mean Unemployment (%)": round(insee_avg_unemployment, 2)
    }
    
    return stats, docs_per_dept, insee_year_avg