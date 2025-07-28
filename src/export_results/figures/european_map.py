import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
import requests
import zipfile
import os
from set_paths import create_path_dict
from export_results.figures.color_map import JET_COLOR_MAP

def download_europe_shapefile(data_path):
    """Download European countries shapefile if not already present."""
    shapefile_path = os.path.join(data_path, 'ne_50m_admin_0_countries.shp')
    
    if not os.path.exists(shapefile_path):
        # Check if zip file exists in the data directory
        zip_path = os.path.join(data_path, 'ne_50m_admin_0_countries.zip')
        
        if os.path.exists(zip_path):
            print(f"Found zip file at: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                print("Extraction complete!")
                return True
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file.")
                return False
        
        # If no zip file found, try to download
        print("Zip file not found locally. Attempting to download...")
        url = "https://www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs(data_path, exist_ok=True)
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            
            os.remove(zip_path)
            print("Download and extraction complete!")
            return True
            
        except (requests.RequestException, zipfile.BadZipFile) as e:
            print(f"Download failed: {e}")
            print(f"\nPlease manually:")
            print(f"1. Download 'ne_50m_admin_0_countries.zip' from:")
            print(f"   https://www.naturalearthdata.com/downloads/50m-cultural-vectors/")
            print(f"2. Place it in: {data_path}")
            return False
    
    return True

def color_europe_map(country_colors, paths_dict, figsize=(12, 8), title="European Map"):
    """
    Color a map of Europe based on a dictionary of countries and colors.
    
    Parameters:
    country_colors (dict): Dictionary with country names/codes as keys and colors as values
                          Countries can be full names (e.g., 'Germany', 'France') or 
                          ISO codes (e.g., 'DE', 'FR', 'DEU', 'FRA')
                          Colors can be hex codes, color names, or RGB tuples
    paths_dict (dict): Dictionary containing paths, must include 'open_data' key
    figsize (tuple): Figure size for the plot
    title (str): Title for the map
    
    Returns:
    matplotlib figure object
    """
    
    data_path = paths_dict['open_data'] + 'map_data/'
    
    # Download shapefile if needed
    if not download_europe_shapefile(data_path):
        print("Could not download shapefile. Please download manually and try again.")
        return None
    
    # Load world countries shapefile
    shapefile_path = os.path.join(data_path, 'ne_50m_admin_0_countries.shp')
    world = gpd.read_file(shapefile_path)
    
    # Filter for European countries (typical European map bounds)
    # Longitude: -10 to 35 (Portugal/Ireland to western Russia)
    # Latitude: 35 to 72 (small bit of North Africa to northern Scandinavia)
    europe = world.cx[-10:35, 35:72]
    
    # Create a copy to avoid modifying original data
    europe = europe.copy()
    
    # Add a color column, default to white
    europe['color'] = 'white'
    
    # Map colors based on the input dictionary
    for country, color in country_colors.items():
        # Convert RGB tuples to matplotlib-compatible format
        if isinstance(color, tuple) and len(color) == 3:
            color = color  # Keep as tuple, matplotlib handles it
        
        # Try to match by different country identifiers
        mask = (
            (europe['NAME'].str.upper() == str(country).upper()) |
            (europe['NAME_EN'].str.upper() == str(country).upper()) |
            (europe['ISO_A2'].str.upper() == str(country).upper()) |
            (europe['ISO_A3'].str.upper() == str(country).upper()) |
            (europe['ADM0_A3'].str.upper() == str(country).upper())
        )
        
        if mask.any():
            # Use .iloc to avoid the pandas assignment issue
            indices = europe.index[mask]
            for idx in indices:
                europe.at[idx, 'color'] = color
        else:
            # Debug: print available country codes for troubleshooting
            print(f"Warning: Country '{country}' not found in the dataset")
            if str(country).upper() in ['FR', 'ESP', 'DE', 'IT']:  # Only for common codes
                print(f"Available ISO_A2 codes around this region: {sorted(europe['ISO_A2'].dropna().unique()[:20])}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot the map
    europe.plot(ax=ax, color=europe['color'], edgecolor='black', linewidth=0.5)
    
    # Set the map bounds to focus on Europe
    ax.set_xlim(-10, 35)  # Longitude bounds
    ax.set_ylim(35, 72)   # Latitude bounds
    
    # Set title and remove axes
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove the box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

def color_europe_map_ongoing_ret_age_increases(paths_dict, figsize=(12, 8)):
    """
    Color a map of Europe with default countries showing ongoing retirement age reforms as of 2024.
    Countries are colored by their target retirement age: <65, [65,66), [66,67), 67+
    
    Parameters:
    paths_dict (dict): Dictionary containing paths, must include 'open_data' key
    figsize (tuple): Figure size for the plot
    
    Returns:
    matplotlib figure object
    """
    
    countries_by_target_age = {
        'Less than 65': ['Slovakia', 'Sweden', 'France'], 
        '[65, 66)': ['Estonia', 'Latvia', 'Lithuania', 'Finland', 'Czechia', 'Russia', 'Croatia', 'Austria', 'Switzerland'],
        '[66, 67)': ['Portugal'], 
        '67 or more': ['Denmark', 'Germany', 'Netherlands', 'Belgium', 'Spain', 'United Kingdom'] 
    }

    
    # Use different colors from JET_COLOR_MAP for each category
    colors = {
        'Less than 65': JET_COLOR_MAP[3],    
        '[65, 66)': JET_COLOR_MAP[3],        
        '[66, 67)': JET_COLOR_MAP[3],       
        '67 or more': JET_COLOR_MAP[3]       
    }
    
    # Create country colors dictionary
    country_colors = {}
    for category, countries in countries_by_target_age.items():
        for country in countries:
            country_colors[country] = colors[category]
    
    # Create the map
    fig = color_europe_map(
        country_colors=country_colors,
        paths_dict=paths_dict,
        figsize=figsize,
        title="Countries with Ongoing Reforms of the Retirement Age as of 2024"
    )
    
    if fig is None:
        return None
    
    ## Add legend
    #ax = fig.get_axes()[0]
    #
    ## Create legend elements
    #from matplotlib.patches import Patch
    #legend_elements = []
    #for category, color in colors.items():
    #    if countries_by_target_age[category]:  # Only add to legend if category has countries
    #        legend_elements.append(Patch(facecolor=color, label=f'Target retirement age: {category}'))
    
    #Add legend to top-left corner
    #ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
    #          fancybox=True, shadow=True, fontsize=10)
    
    return fig

# Example usage
if __name__ == "__main__":
    # Create paths dictionary
    path_dict = create_path_dict(define_user=False)
    
    # Create and display the map using defaults
    fig = color_europe_map_ongoing_ret_age_increases(path_dict)
    if fig:
        plt.show()
    
    # fig.savefig('europe_map.png', dpi=300, bbox_inches='tight')