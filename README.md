# Discourse vs. Reality: Correlating Political Manifestos with Economic Indicators via Topic Modeling

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Topic_Modeling-orange.svg)
![Format](https://img.shields.io/badge/Paper-NeurIPS-b31b1b.svg)


This repository contains the code, data preprocessing pipeline, and analysis for the scientific report on the **Archelec Corpus**, exploring the correlation between themes in French political manifestos and real-world economic indicators. 

This project is conducted as part of the NLP/Digital Humanities assessment and follows the NeurIPS conference paper format.

## Project Overview
The primary research question addressed in this project is: *Do the themes present in electoral manifestos reflect the actual economic situation of a department, or are they strictly dictated by national party lines?*

By leveraging Natural Language Processing (NLP) techniques, this project extracts dominant economic themes (e.g., unemployment, inflation) from historical political manifestos and compares their frequency against actual departmental economic indicators provided by the French National Institute of Statistics and Economic Studies (INSEE).

## Data Sources
1. **Archelec Corpus:** - Transcriptions (OCR): Extracted from the [Arkindex repository](https://gitlab.teklia.com/ckermorvant/arkindex_archelec).
   - Metadata: Downloaded from the Sciences Po Archelec explorer (contains `titulaire-soutien`, `departement-insee`, dates).
2. **INSEE Data:** Historical departmental economic indicators corresponding to the election years.

## Methodology
1. **Data Preprocessing:** Collaborative cleaning of OCR texts, merging with Archelec CSV metadata and INSEE data.
2. **NLP Pipeline:** Tokenization, stop-word removal, and lemmatization/stemming applied to French political texts.
3. **Topic Modeling:** Application of topic modeling algorithms (e.g., LDA / NMF / BERTopic) to extract latent themes.
4. **Statistical Analysis:** Computing correlations (Pearson/Spearman) between the extracted "Economic/Unemployment" topic frequencies and real-world INSEE metrics.

## Repository Structure
```text
├── data/
│   ├── raw/             # Raw OCR text files and original CSVs (Not tracked by Git)
│   ├── processed/       # Cleaned datasets ready for modeling
│   └── external/        # INSEE economic datasets
├── notebooks/           # Jupyter notebooks for data exploration and visualization
├── src/                 # Python scripts for the NLP pipeline and topic modeling
├── paper/               # LaTeX source files and the final PDF report (NeurIPS format)
├── .gitignore
├── requirements.txt     # Python dependencies
└── README.md
