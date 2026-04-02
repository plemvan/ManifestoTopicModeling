import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

def plot_comparative_correlation(lexicon_results, final_results, df_insee, year_mapping):
    """
    Generates a 2x3 grid comparing the correlations of the lexical approach 
    and the BERTopic approach against the INSEE unemployment rates.
    """
    # Figure preparation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exploratory Analysis: INSEE Unemployment Rate vs. Discursive Intensity", 
                 fontsize=16, fontweight='bold', y=1.02)

    for i, year in enumerate([1981, 1988, 1993]):
        insee_year = year_mapping[year]
        
        cols = [f'T1_{insee_year}', f'T2_{insee_year}', f'T3_{insee_year}', f'T4_{insee_year}']
        df_insee_temp = df_insee.copy()
        df_insee_temp['Code_clean'] = df_insee_temp['Code'].astype(str).str.replace('.0', '', regex=False).str.zfill(2)
        
        if not all(col in df_insee_temp.columns for col in cols):
            axes[0, i].set_title(f"Missing INSEE Data ({year})")
            axes[1, i].set_title(f"Missing INSEE Data ({year})")
            continue 
            
        df_insee_temp['taux_insee'] = df_insee_temp[cols].mean(axis=1)
        
        df_lex = lexicon_results[year]['depts'].copy()
        
        df_bert_weights = final_results[year]['weights'].copy()
        df_bert_weights['score_chomage'] = df_bert_weights.get(0, 0.0) * 100
        
        # Retrieve department code for merging
        dept_mapping = final_results[year]['df_para'][['filename', 'department_code']].drop_duplicates()
        df_bert_dept = df_bert_weights.merge(dept_mapping, on='filename')
        df_bert_dept_avg = df_bert_dept.groupby('department_code')['score_chomage'].mean().reset_index()

        df_merged_lex = pd.merge(df_lex, df_insee_temp[['Code_clean', 'taux_insee']], 
                                 left_on='department_code', right_on='Code_clean').dropna(subset=['taux_insee', 'ratio_moyen_dept'])
        
        df_merged_bert = pd.merge(df_bert_dept_avg, df_insee_temp[['Code_clean', 'taux_insee']], 
                                  left_on='department_code', right_on='Code_clean').dropna(subset=['taux_insee', 'score_chomage'])

        r_lex, p_lex = pearsonr(df_merged_lex['taux_insee'], df_merged_lex['ratio_moyen_dept'])
        r_bert, p_bert = pearsonr(df_merged_bert['taux_insee'], df_merged_bert['score_chomage'])

        sns.regplot(x='taux_insee', y='ratio_moyen_dept', data=df_merged_lex, ax=axes[0, i], 
                    color='#1f77b4', scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linewidth': 2})
        axes[0, i].set_title(f"Lexical Approach ({year})", fontsize=13)
        axes[0, i].set_ylabel("Lexical Weight (%)" if i == 0 else "")
        axes[0, i].set_xlabel("")
        axes[0, i].grid(True, linestyle='--', alpha=0.4)
        axes[0, i].text(0.05, 0.95, f"r = {r_lex:.2f}\np = {p_lex:.3f}", 
                        transform=axes[0, i].transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

        sns.regplot(x='taux_insee', y='score_chomage', data=df_merged_bert, ax=axes[1, i], 
                    color='#ff7f0e', scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linewidth': 2})
        axes[1, i].set_title(f"BERTopic Zero-Shot ({year})", fontsize=13)
        axes[1, i].set_ylabel("BERTopic Weight (%)" if i == 0 else "")
        axes[1, i].set_xlabel("INSEE Unemployment Rate (%)")
        axes[1, i].grid(True, linestyle='--', alpha=0.4)
        axes[1, i].text(0.05, 0.95, f"r = {r_bert:.2f}\np = {p_bert:.3f}", 
                        transform=axes[1, i].transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)

    plt.tight_layout()
    return fig