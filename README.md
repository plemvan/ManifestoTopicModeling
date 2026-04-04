# Discourse vs. Reality: Correlating Political Manifestos with Economic Indicators via Topic Modeling

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Topic_Modeling-orange.svg)
![Format](https://img.shields.io/badge/Paper-NeurIPS-b31b1b.svg)

This repository contains the code, data preprocessing pipeline, and analysis for the scientific report on the **Archelec Corpus**, exploring the correlation between themes in French political manifestos and real-world economic indicators across three major legislative election years (1981, 1988, and 1993). 

## Project Overview
The primary research question addressed in this project is: *Do the themes present in electoral manifestos reflect the actual economic situation of a local department, or are they strictly dictated by national party lines?*

By leveraging Natural Language Processing (NLP) techniques, this project extracts the dominant theme of economic crisis and unemployment from historical political manifestos. I  then compare the intensity of this discourse against the actual departmental unemployment rates provided by the French National Institute of Statistics and Economic Studies (INSEE).

## Data Sources
1. **Archelec Corpus:** - Transcriptions (OCR): Extracted from the [Arkindex repository](https://gitlab.teklia.com/ckermorvant/arkindex_archelec).
   - Metadata: Downloaded from the Sciences Po Archelec explorer (contains `titulaire-soutien`, `departement-insee`, dates).
2. **INSEE Data:** Historical departmental unemployment rates (T1-T4 averages) corresponding to the targeted election years.

## Methodology
To accurately capture the macroeconomic reality within political discourse, I structured the text analysis as a methodological funnel, comparing Topic modeling models with robust rule-based metrics:

1. **Data Preprocessing:** Collaborative cleaning of OCR texts, handling nested lists and brackets, and merging with Archelec metadata and INSEE indicators.
2. **Baseline Exploration (Unsupervised LDA):** Initial modeling to understand the macro-structure of electoral manifestos, which highlighted the heavy presence of "electoral noise" and the semantic dilution of the economic theme.
3. **Semantic Contextualization (Zero-Shot BERTopic):** Utilizing "dangvantuan/sentence-camembert-large" embeddings to group paragraphs based on semantic proximity to a target phrase (*"chômage, emploi, précarité, licenciement et crise économique"*). This acted as the qualitative, context-aware proxy.
4. **Robust Quantification (Lexical Approach):** A targeted dictionary approach to calculate the exact ratio of paragraphs containing explicit economic keywords, providing a rigid, transparent metric to cross-reference with the AI outputs.
5. **Geographic & Statistical Analysis:** - **Spatial Distribution:** Generating choropleth maps (`geopandas`) to visually compare the INSEE "ground truth" with the NLP discursive hotspots.
   - **Exploratory Correlation:** Calculating Pearson's correlation coefficient ($r$) and plotting scatter plots to observe the relationship between discursive intensity and local reality over time. We approach these measurements with strict methodological prudence, treating the NLP outputs as exploratory proxies for broad trends rather than absolute statistical proofs, given the inherent noise of unsupervised classification.

## Repository Structure
```text
├── data/
│   ├── raw/             # Raw OCR text files and original CSVs (Not tracked by Git)
│   ├── processed/       # Cleaned datasets ready for modeling
│   └── external/        # INSEE economic datasets
├── notebooks/           # Jupyter notebooks for data exploration, ground-truthing, and visualization
├── src/                 # Python scripts for the NLP pipeline (BERTopic, Lexical, Zero-Shot NLI) and plotting
├── paper/               # LaTeX source files and the final PDF report (NeurIPS format)
├── .gitignore           # Excludes __pycache__, raw data, and environment variables
├── requirements.txt     # Python dependencies (pandas, seaborn, bertopic, geopandas, etc.)
└── README.md
