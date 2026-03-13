import pandas as pd
from pathlib import Path

# Path 
path = Path("data/raw/text_files/1993/legislatives")
documents = []

# To get all .txt files in the folder and read their content
for file in path.glob("*.txt"):
    try:
        texte = file.read_text(encoding='utf-8', errors='ignore')
        documents.append({
            "filename": file.name,
            "texte_brut": texte})
    except Exception as e:
        print(f"Error with {file.name} : {e}")


df = pd.DataFrame(documents)
print(f"{len(df)} documents in the dataframe")