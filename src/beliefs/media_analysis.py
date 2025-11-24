import datetime
import time

import mediacloud.api

from set_paths import create_path_dict

path_dict = create_path_dict()

import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict

path_dict = create_path_dict()

from set_styles import set_colors, set_plot_defaults

set_plot_defaults()

# =========
# PARAMETERS
# =========
MEDIACLOUD_API_KEY = "208d82c26a2871958b7b737bab35b7cc23e8f59d"

# Initialize Media Cloud API client
mc = mediacloud.api.SearchApi(MEDIACLOUD_API_KEY)

# Media Cloud collection ID for German State and Local media
GERMAN_STATE_LOCAL_COLLECTION = 38379816

# Using German search terms
GROUPS = {
    "SRA": [
        "Rentenzugangsalter OR Renteneintrittsalter OR (Rente AND (Eintrittsalter OR Zugangsalter))",
    ],
    "ERP": [
        "Rente AND (abschlag OR abschlagsfrei) AND 3,6",
    ],
}

START_DATE = "2014-01-01"
END_DATE = "2025-01-01"

ROLLING_COUNT = 7


# ==========
# HELPER FUNCTION
# ==========
def mediacloud_count_over_time(mc_client, query):
    """
    Query Media Cloud for story counts over time using collection ID

    Parameters:
    -----------
    mc_client : mediacloud.api.SearchApi
        Initialized Media Cloud client
    query : str
        Search query (supports boolean operators)

    Returns:
    --------
    pd.DataFrame with columns: date, count
    """

    # Transform start and end date to datetime objects and then to date
    start_date = datetime.datetime.strptime(START_DATE, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(END_DATE, "%Y-%m-%d").date()

    # Get count split by day
    results = mc_client.story_count_over_time(
        query,
        start_date=start_date,
        end_date=end_date,
        collection_ids=[GERMAN_STATE_LOCAL_COLLECTION],
    )
    results_df = pd.DataFrame(results)

    # Use the existing structure
    df = results_df[["date", "ratio"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"ratio": query})
    return df


# ==========
# FETCH AND COMBINE DATA
# ==========
def fetch_all_data():
    """Main function to fetch all data from Media Cloud"""

    group_results = {}

    print("\n🔍 Starting Media Cloud queries...")
    print(f"📅 Date range: {START_DATE} to {END_DATE}")
    print(f"📰 Collection ID: {GERMAN_STATE_LOCAL_COLLECTION}\n")

    for group, keywords in GROUPS.items():
        dfs = []

        for kw in keywords:
            df_kw = mediacloud_count_over_time(mc, kw)

            if df_kw is not None:
                dfs.append(df_kw)
                time.sleep(1)

        # Merge all keywords in the group
        merged = dfs[0]
        for df_kw in dfs[1:]:
            merged = pd.merge(merged, df_kw, on="date", how="outer")

        merged = merged.fillna(0)
        # Sum all keyword columns (all except 'date')
        merged[group] = merged.drop(columns=["date"]).sum(axis=1)
        group_results[group] = merged[["date", group]]

        # One print statement per group
        total = merged[group].sum()
        print(f"✅ {group}: {total:.0f} total articles")

    first_key = list(GROUPS.keys())[0]
    # Combine all groups into one dataframe
    combined = group_results[first_key]
    for group_name, group_df in list(group_results.items())[1:]:
        combined = pd.merge(combined, group_df, on="date", how="outer")

    combined = combined.sort_values("date").fillna(0)

    # Save to CSV
    output_file = path_dict["open_data"] + "mediacloud_raw_data.csv"
    combined.to_csv(output_file, index=False)

    print(f"\n💾 Saved to: {output_file}")

    return combined


def plot_media_coverage(data):
    """
    Plot media coverage data with rolling averages

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'date' column and one or more data columns
    """

    # Make a copy to avoid modifying original data
    df = data.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date")

    # Create plot
    fig, ax = plt.subplots()
    color_map, _ = set_colors()

    colors = {
        "ERP": color_map[0],
        "SRA": color_map[1],
    }

    # Plot each column's rolling average
    for idx, group in enumerate(GROUPS.keys()):
        rolling_counts = df[group].rolling(ROLLING_COUNT, center=True).mean()

        ax.plot(
            df["date"],
            rolling_counts,
            label=group,
            color=colors[group],
        )

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Share of Articles (Seven-day MA)")
    ax.legend()
    fig.tight_layout()

    # Save plot
    output_path = path_dict["beliefs_plots"] + "media_comparison.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"✅ Plot saved to: {output_path}")


# ==========
# MAIN EXECUTION
# ==========
if __name__ == "__main__":
    # Ask if only the plot should be run
    only_plot = input("Use existing data? (y/n): ") == "y"
    if only_plot:
        # Load existing data
        input_file = path_dict["open_data"] + "mediacloud_raw_data.csv"
        group_results = pd.read_csv(input_file, parse_dates=["date"])
        print(f"\n📂 Loaded data from: {input_file}")
    else:
        # Fetch new data
        group_results = fetch_all_data()

    # Create plot
    plot_media_coverage(
        group_results,
    )
