import pandas as pd
import numpy as np

from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs
from process_data.structural_sample_scripts.create_structural_est_sample import create_structural_est_sample


def create_dataset_description_table_latex(path_dict, specs, save=False):
    """
    Create a LaTeX table describing the structural estimation sample dataset.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and specifications
    specs : dict
        Specifications dictionary

    Returns
    -------
    str
        LaTeX table code for dataset description
    """

    # Load the structural estimation sample
    df = create_structural_est_sample(path_dict, specs, load_data=True)

    # Define groups: sex (0=men, 1=women) x education (0=low, 1=high)
    groups = {
        'men_low': df[(df['sex'] == 0) & (df['education'] == 0)],
        'men_high': df[(df['sex'] == 0) & (df['education'] == 1)],
        'women_low': df[(df['sex'] == 1) & (df['education'] == 0)],
        'women_high': df[(df['sex'] == 1) & (df['education'] == 1)]
    }

    # Calculate statistics for each group
    stats = {}
    for group_name, group_df in groups.items():
        stats[group_name] = {}

        # Basic counts
        stats[group_name]['unique_households'] = group_df['pseudo_hid'].nunique()
        stats[group_name]['unique_individuals'] = group_df['pseudo_pid'].nunique()
        stats[group_name]['observations'] = len(group_df)

        # Employment shares
        choice_counts = group_df['choice'].value_counts()
        total_obs = len(group_df)
        stats[group_name]['share_retired'] = choice_counts.get(0, 0) / total_obs
        stats[group_name]['share_unemployed'] = choice_counts.get(1, 0) / total_obs
        stats[group_name]['share_part_time'] = choice_counts.get(2, 0) / total_obs
        stats[group_name]['share_full_time'] = choice_counts.get(3, 0) / total_obs

        # Other shares
        stats[group_name]['share_good_health'] = (group_df['health'] == 0).mean()
        stats[group_name]['share_single'] = (group_df['partner_state'] == 0).mean()

        # Averages
        stats[group_name]['avg_experience'] = group_df['experience'].mean()
        stats[group_name]['avg_wealth'] = group_df['wealth'].mean() / 1000  # Convert to thousands of euros

    # Calculate totals
    stats['total'] = {}
    stats['total']['unique_households'] = df['pseudo_hid'].nunique()
    stats['total']['unique_individuals'] = df['pseudo_pid'].nunique()
    stats['total']['observations'] = len(df)

    choice_counts_total = df['choice'].value_counts()
    total_obs_all = len(df)
    stats['total']['share_retired'] = choice_counts_total.get(0, 0) / total_obs_all
    stats['total']['share_unemployed'] = choice_counts_total.get(1, 0) / total_obs_all
    stats['total']['share_part_time'] = choice_counts_total.get(2, 0) / total_obs_all
    stats['total']['share_full_time'] = choice_counts_total.get(3, 0) / total_obs_all

    stats['total']['share_good_health'] = (df['health'] == 0).mean()
    stats['total']['share_single'] = (df['partner_state'] == 0).mean()
    stats['total']['avg_experience'] = df['experience'].mean()
    stats['total']['avg_wealth'] = df['wealth'].mean() / 1000  # Convert to thousands of euros

    # Generate LaTeX table
    latex_table = (
        r"\begin{tabular}{l|cc|cc|c}" + "\n" +
        r"\toprule" + "\n" +
        r" & \multicolumn{2}{c|}{Men} & \multicolumn{2}{c|}{Women} & \\" + "\n" +
        r" & High Educ. & Low Educ. & High Educ. & Low Educ. & Total \\" + "\n" +
        r"\midrule" + "\n"
    )

    # Add rows
    latex_table += f"Unique Households & {stats['men_high']['unique_households']:,} & {stats['men_low']['unique_households']:,} & {stats['women_high']['unique_households']:,} & {stats['women_low']['unique_households']:,} & {stats['total']['unique_households']:,} \\\\\n"
    latex_table += f"Unique Individuals & {stats['men_high']['unique_individuals']:,} & {stats['men_low']['unique_individuals']:,} & {stats['women_high']['unique_individuals']:,} & {stats['women_low']['unique_individuals']:,} & {stats['total']['unique_individuals']:,} \\\\\n"
    latex_table += f"Observations & {stats['men_high']['observations']:,} & {stats['men_low']['observations']:,} & {stats['women_high']['observations']:,} & {stats['women_low']['observations']:,} & {stats['total']['observations']:,} \\\\\n"
    latex_table += r"\midrule" + "\n"

    # Employment shares
    latex_table += f"Share Full-time & {stats['men_high']['share_full_time']:.3f} & {stats['men_low']['share_full_time']:.3f} & {stats['women_high']['share_full_time']:.3f} & {stats['women_low']['share_full_time']:.3f} & {stats['total']['share_full_time']:.3f} \\\\\n"
    latex_table += f"Share Part-time & {stats['men_high']['share_part_time']:.3f} & {stats['men_low']['share_part_time']:.3f} & {stats['women_high']['share_part_time']:.3f} & {stats['women_low']['share_part_time']:.3f} & {stats['total']['share_part_time']:.3f} \\\\\n"
    latex_table += f"Share Unemployed & {stats['men_high']['share_unemployed']:.3f} & {stats['men_low']['share_unemployed']:.3f} & {stats['women_high']['share_unemployed']:.3f} & {stats['women_low']['share_unemployed']:.3f} & {stats['total']['share_unemployed']:.3f} \\\\\n"
    latex_table += f"Share Retired & {stats['men_high']['share_retired']:.3f} & {stats['men_low']['share_retired']:.3f} & {stats['women_high']['share_retired']:.3f} & {stats['women_low']['share_retired']:.3f} & {stats['total']['share_retired']:.3f} \\\\\n"
    latex_table += r"\midrule" + "\n"

    # Other characteristics
    latex_table += f"Share Good Health & {stats['men_high']['share_good_health']:.3f} & {stats['men_low']['share_good_health']:.3f} & {stats['women_high']['share_good_health']:.3f} & {stats['women_low']['share_good_health']:.3f} & {stats['total']['share_good_health']:.3f} \\\\\n"
    latex_table += f"Share Single & {stats['men_high']['share_single']:.3f} & {stats['men_low']['share_single']:.3f} & {stats['women_high']['share_single']:.3f} & {stats['women_low']['share_single']:.3f} & {stats['total']['share_single']:.3f} \\\\\n"
    latex_table += r"\midrule" + "\n"

    # Averages
    latex_table += f"Average Work Experience & {stats['men_high']['avg_experience']:.1f} & {stats['men_low']['avg_experience']:.1f} & {stats['women_high']['avg_experience']:.1f} & {stats['women_low']['avg_experience']:.1f} & {stats['total']['avg_experience']:.1f} \\\\\n"
    latex_table += f"Average Wealth (1000 EUR) & {stats['men_high']['avg_wealth']:.1f} & {stats['men_low']['avg_wealth']:.1f} & {stats['women_high']['avg_wealth']:.1f} & {stats['women_low']['avg_wealth']:.1f} & {stats['total']['avg_wealth']:.1f} \\\\\n"

    latex_table += r"\bottomrule" + "\n" + r"\end{tabular}"

    if save:
        output_path = path_dict["data_tables"] + "structural_estimation_sample_description.tex"
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"Table saved to: {output_path}")

    return latex_table


if __name__ == "__main__":
    # Example usage
    path_dict = create_path_dict(define_user=True)
    specs = read_and_derive_specs(path_dict["specs"])
    latex_table = create_dataset_description_table_latex(path_dict, specs)
    print(latex_table)