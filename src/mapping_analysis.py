import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

def plot_geographic_comparison(lexicon_results, final_results, df_insee, year_mapping, geojson_path):
    """
    Generates a 3x3 grid of choropleth maps comparing INSEE reality 
    with Lexical and BERTopic discursive proxies.
    """
    # 1. Load the geographic shapes of France
    france = gpd.read_file(geojson_path)
    # Ensure department codes are strings and padded (e.g., '01')
    france['code'] = france['code'].astype(str).str.zfill(2)

    # 2. Setup the figure (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle("Geographic Distribution: Economic Reality vs. Political Discourse", 
                 fontsize=20, fontweight='bold', y=0.95)

    for i, year in enumerate([1981, 1988, 1993]):
        insee_year = year_mapping[year]
        
        # INSEE
        cols = [f'T1_{insee_year}', f'T2_{insee_year}', f'T3_{insee_year}', f'T4_{insee_year}']
        df_insee_temp = df_insee.copy()
        df_insee_temp['Code_clean'] = df_insee_temp['Code'].astype(str).str.replace('.0', '', regex=False).str.zfill(2)
        df_insee_temp['taux_insee'] = df_insee_temp[cols].mean(axis=1)
        
        # Lexical
        df_lex = lexicon_results[year]['depts'].copy()
        
        # BERTopic
        df_bert_weights = final_results[year]['weights'].copy()
        df_bert_weights['score_chomage'] = df_bert_weights.get(0, 0.0) * 100
        dept_mapping = final_results[year]['df_para'][['filename', 'department_code']].drop_duplicates()
        df_bert_dept = df_bert_weights.merge(dept_mapping, on='filename')
        df_bert_dept_avg = df_bert_dept.groupby('department_code')['score_chomage'].mean().reset_index()

        map_insee = france.merge(df_insee_temp, left_on='code', right_on='Code_clean')
        map_lex = france.merge(df_lex, left_on='code', right_on='department_code')
        map_bert = france.merge(df_bert_dept_avg, left_on='code', right_on='department_code')

        
        # Row 1: INSEE (The Reality)
        map_insee.plot(column='taux_insee', ax=axes[0, i], legend=True, cmap='Reds',
                       legend_kwds={'label': "Unemployment Rate (%)", 'orientation': "horizontal", 'pad': 0.01})
        axes[0, i].set_title(f"INSEE Reality ({year})", fontsize=14, fontweight='bold')

        # Row 2: Lexical (The Proxy)
        map_lex.plot(column='ratio_moyen_dept', ax=axes[1, i], legend=True, cmap='Blues',
                     legend_kwds={'label': "Lexical Weight (%)", 'orientation': "horizontal", 'pad': 0.01})
        axes[1, i].set_title(f"Lexical Intensity ({year})", fontsize=14)

        # Row 3: BERTopic (The Proxy)
        map_bert.plot(column='score_chomage', ax=axes[2, i], legend=True, cmap='Oranges',
                      legend_kwds={'label': "BERTopic Weight (%)", 'orientation': "horizontal", 'pad': 0.01})
        axes[2, i].set_title(f"BERTopic Intensity ({year})", fontsize=14)

        for row in range(3):
            axes[row, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig