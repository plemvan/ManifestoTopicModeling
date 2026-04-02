import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_unemployment_comparison(df_insee, year_mapping):
    """
    Downloads the map of France and displays the unemployment maps 
    side-by-side using a UNIFIED color scale for accurate comparison.
    """
    # 1. Load the map data
    url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    try:
        gdf_france = gpd.read_file(url_geojson)
    except Exception as e:
        print(f"Error downloading the map data: {e}")
        return
        
    gdf_france['code'] = gdf_france['code'].astype(str).str.zfill(2)

    # Find the global min and max unemployment rates across all years
    # This is crucial so the color scale is identical on all 3 maps
    all_values = []
    for insee_year in year_mapping.values():
        cols = [f'T1_{insee_year}', f'T2_{insee_year}', f'T3_{insee_year}', f'T4_{insee_year}']
        if all(col in df_insee.columns for col in cols):
            # Calculate means and add to our master list
            year_means = df_insee[cols].mean(axis=1).dropna().tolist()
            all_values.extend(year_means)
            
    if not all_values:
        print("Error: Could not find data to establish a color scale.")
        return
        
    global_min = min(all_values)
    global_max = max(all_values)

    fig, axes = plt.subplots(1, len(year_mapping), figsize=(18, 6))

    for ax, (election_year, insee_year) in zip(axes, year_mapping.items()):
        
        cols_year = [f'T1_{insee_year}', f'T2_{insee_year}', f'T3_{insee_year}', f'T4_{insee_year}']
        
        if not all(col in df_insee.columns for col in cols_year):
            ax.set_title(f"Missing data for {insee_year}")
            ax.axis('off')
            continue

        df_temp = df_insee.copy()
        df_temp['mean_unemployment'] = df_temp[cols_year].mean(axis=1)
        df_temp['code_departement'] = df_temp['Code'].astype(str).str.replace('.0', '', regex=False).str.zfill(2)

        # Merge data
        gdf_merged = gdf_france.merge(df_temp, left_on='code', right_on='code_departement', how='left')
        gdf_metro = gdf_merged[~gdf_merged['code'].str.startswith('97')]

        # Draw map USING FIXED VMIN AND VMAX
        gdf_metro.plot(
            column='mean_unemployment', 
            ax=ax, 
            cmap='OrRd',              
            vmin=global_min,          # Fix the bottom of the color scale
            vmax=global_max,          # Fix the top of the color scale
            linewidth=0.5, 
            edgecolor='0.3',          
            legend=False,             # Turn off individual legends
            missing_kwds={"color": "lightgrey"}
        )

        ax.set_title(f"Election {election_year}\n(INSEE Data: {insee_year})", fontsize=14, fontweight='bold')
        ax.axis('off')

    # We create a mapping object based on our global min/max
    sm = ScalarMappable(cmap='OrRd', norm=Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([]) # Empty array required by Matplotlib
    
    # Place it at the bottom, centered across the axes
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', shrink=0.5, pad=0.05)
    cbar.set_label('Mean Unemployment Rate (%)', fontsize=12, fontweight='bold')

    plt.show()