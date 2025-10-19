from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests

# =========
# PARAMETERS
# =========
MAJOR_GERMAN_OUTLETS = [
    "spiegel.de",
    "zeit.de",
    "faz.net",
    "sueddeutsche.de",
    "welt.de",
    "tagesschau.de",
    "handelsblatt.com",
    "bild.de",
    "focus.de",
    "stern.de",
    "tagesspiegel.de",
    "taz.de",
    "n-tv.de",
    "fr.de",
    "rp-online.de",
    "merkur.de",
    "berliner-zeitung.de",
]

GROUPS = {
    "Renteneintrittsalter_Debatte": [
        "Pension AND Age",
        "Retirement AND Age",
    ],
    "Abschlag_3_6": ["Pension AND Deduction", "Retirement AND Deduction"],
}

START_DATE = "2020-01-01"
END_DATE = "2025-10-17"


# ==========
# HELPER FUNCTION WITH DOMAIN FILTERING - SPLIT QUERIES
# ==========
def gdelt_query(keyword, start_date=START_DATE, end_date=END_DATE, chunk_size=8):
    """Query GDELT with German domain restriction, splitting into chunks to avoid URL length limits"""

    start_dt = start_date.replace("-", "") + "000000"
    end_dt = end_date.replace("-", "") + "235959"

    all_dfs = []

    # Split domains into chunks to avoid query length limits
    for i in range(0, len(MAJOR_GERMAN_OUTLETS), chunk_size):
        domain_chunk = MAJOR_GERMAN_OUTLETS[i : i + chunk_size]
        domain_filter = " OR ".join([f"domain:{d}" for d in domain_chunk])
        full_query = f"{keyword} ({domain_filter})"

        base_url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={full_query.replace(' ', '+')}"
            f"&mode=TimelineVolRAW"
            f"&format=csv"
            f"&startdatetime={start_dt}"
            f"&enddatetime={end_dt}"
        )

        print(
            f"    Chunk {i // chunk_size + 1}/{(len(MAJOR_GERMAN_OUTLETS) - 1) // chunk_size + 1}...",
            end=" ",
        )

        try:
            response = requests.get(base_url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df = df[df["Series"] == "Article Count"].copy()

            if not df.empty:
                all_dfs.append(df)
                print(f"✅ {len(df)} rows")
            else:
                print("⚠️ empty")

        except Exception as e:
            print(f"❌ Error: {e}")

    if not all_dfs:
        print(f"  ⚠️ No results for '{keyword}'")
        return None

    # Combine all chunks
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Group by date and sum values (in case same date appears in multiple chunks)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"])
    combined_df = combined_df.groupby("Date", as_index=False)["Value"].sum()

    combined_df.rename(columns={"Date": "datetime", "Value": keyword}, inplace=True)

    print(
        f"  ✅ Total: {len(combined_df)} days (range: {combined_df[keyword].min():.0f}-{combined_df[keyword].max():.0f})"
    )

    return combined_df


# ==========
# FETCH AND COMBINE DATA
# ==========
group_results = {}

for group, keywords in GROUPS.items():
    print(f"\n📊 Group: {group}")
    dfs = []

    for kw in keywords:
        df_kw = gdelt_query(kw)
        if df_kw is not None:
            dfs.append(df_kw)

    if not dfs:
        print(f"  ⚠️ No data for any keywords in this group")
        continue

    print(f"  ✅ Successfully fetched {len(dfs)}/{len(keywords)} keywords")

    # Merge all keywords in the group
    merged = dfs[0]
    for df_kw in dfs[1:]:
        merged = pd.merge(merged, df_kw, on="datetime", how="outer")

    merged = merged.fillna(0)
    merged[group] = merged.drop(columns=["datetime"]).sum(axis=1)
    group_results[group] = merged[["datetime", group]]

# ==========
# CONTINUE WITH PLOTTING IF WE HAVE DATA
# ==========
if len(group_results) >= 2:
    combined = (
        pd.merge(
            group_results["Renteneintrittsalter_Debatte"],
            group_results["Abschlag_3_6"],
            on="datetime",
            how="outer",
        )
        .sort_values("datetime")
        .fillna(0)
    )

    # Apply 7-day rolling average
    combined["Renteneintrittsalter_Debatte_7Tage"] = (
        combined["Renteneintrittsalter_Debatte"].rolling(7, center=True).mean()
    )
    combined["Abschlag_3_6_7Tage"] = (
        combined["Abschlag_3_6"].rolling(7, center=True).mean()
    )

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        combined["datetime"],
        combined["Renteneintrittsalter_Debatte_7Tage"],
        label="Debatte über Renteneintrittsalter",
        linewidth=2,
    )
    plt.plot(
        combined["datetime"],
        combined["Abschlag_3_6_7Tage"],
        label="Diskussion über Rentenabschlag",
        linewidth=2,
    )

    plt.title(
        "Medienberichterstattung in Deutschland: Renteneintrittsalter vs. Rentenabschlag"
    )
    plt.xlabel("Datum")
    plt.ylabel("Artikel aus deutschen Medien (7-Tage-Durchschnitt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(
        f"\n✅ Analysis complete! Date range: {combined['datetime'].min()} to {combined['datetime'].max()}"
    )
else:
    print("\n❌ Not enough data to create comparison plot")
