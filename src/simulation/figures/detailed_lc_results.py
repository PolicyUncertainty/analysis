import matplotlib.pyplot as plt
import pandas as pd
import os
from set_styles import get_figsize, set_colors

def plot_detailed_lifecycle_results(df_results_path, path_dict, specs, subfolder=None, show=False, save=True):
    """
    Plot detailed lifecycle results by demographic groups.
    
    Parameters:
    -----------
    df_results_path : str
        Path to CSV file with multi-index DataFrame from calc_life_cycle_detailed
    path_dict : dict
        Path dictionary for saving plots
    specs : dict
        Model specifications with labels
    subfolder : str, optional
        Subfolder within simulation_plots to save plots
    show : bool
        Whether to display plots
    save : bool
        Whether to save plots
    """
    
    # Load the detailed results from CSV
    df_results = pd.read_csv(df_results_path, index_col=[0, 1, 2])

    # Set up plot save directory
    if subfolder:
        plot_dir = path_dict["simulation_plots"] + f"{subfolder}/"
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = path_dict["simulation_plots"]
    
    colors, _ = set_colors()
    
    # Define outcome variables and their display names
    outcomes = {
        'choice_0_rate': 'Retirement Rate',
        'choice_1_rate': 'Unemployment Rate', 
        'choice_2_rate': 'Part-time Work Rate',
        'choice_3_rate': 'Full-time Work Rate',
        'savings_rate': 'Savings Rate',
        'avg_wealth': 'Average Wealth'
    }
    
    # Define group types to plot (exclude aggregate)
    group_types = ['sex', 'education', 'informed', 'health', 'partner_state']
    
    # Get group labels from specs
    group_labels = {
        'sex': specs.get('sex_labels', ['Male', 'Female']),
        'education': specs.get('education_labels', ['Low Education', 'High Education']),
        'informed': ['Uninformed', 'Informed'],
        'health': ['Good Health', 'Bad Health', 'Disabled', 'Dead'],
        'partner_state': ['Single', 'Partnered']
    }
    
    for group_type in group_types:
        # Skip if group not in data
        if group_type not in df_results.index.get_level_values('group_type'):
            continue
            
        # Create figure with subplots for each outcome
        fig, axes = plt.subplots(2, 3, figsize=get_figsize(3, 2))
        axes = axes.flatten()
        
        group_data = df_results.loc[group_type]
        group_values = group_data.index.get_level_values('group_value').unique()
        
        for i, (outcome_var, outcome_name) in enumerate(outcomes.items()):
            ax = axes[i]
            
            # Plot line for each group value
            for j, group_val in enumerate(sorted(group_values)):
                if group_val in group_data.index.get_level_values('group_value'):
                    data = group_data.loc[group_val][outcome_var]
                    
                    # Convert group_val to int for indexing (CSV loading makes it a string)
                    try:
                        group_val_int = int(float(group_val))
                        if group_val_int < len(group_labels[group_type]):
                            label = group_labels[group_type][group_val_int]
                        else:
                            label = f'{group_type}={group_val}'
                    except (ValueError, TypeError):
                        label = f'{group_type}={group_val}'
                    
                    ax.plot(data.index, data.values, 
                           color=colors[j % len(colors)], 
                           label=label, 
                           linewidth=2)
            
            ax.set_xlabel('Age')
            ax.set_ylabel(outcome_name)
            ax.set_title(outcome_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Lifecycle Profiles by {group_type.title()}', fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = f'lifecycle_profiles_by_{group_type}'
            fig.savefig(plot_dir + f'{filename}.pdf', bbox_inches='tight')
            fig.savefig(plot_dir + f'{filename}.png', bbox_inches='tight', dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close(fig)