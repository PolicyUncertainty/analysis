import pandas as pd


def add_wealth(data, path_dict, options):
    """This function adds wealth data to the merged data."""
    data = data.reset_index()

    # Gather wealth data
    data = gather_wealth_data(path_dict["soep_c38"], data, options)

    # Create household consumption equivalence scale
    data = create_hh_cons_equivalence_data(data)

    # Deflate wealth
    data = deflate_wealth(data, path_dict)

    data.set_index(["pid", "syear"], inplace=True)

    return data


def gather_wealth_data(soep_c38_path, merged_data, options):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)

    # for each household, create a row for each year between min and max syear
    min_max_syear = wealth_data.groupby("hid")["syear"].agg(["min", "max"])
    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(row["min"], row["max"] + 1)})
            for hid, row in min_max_syear.iterrows()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )

    # Set 'hid' and 'syear' as the index
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full.sort_index(inplace=True)

    # Interpolate the missing values for each household
    wealth_data_full["w011ha"] = wealth_data_full.groupby("hid")["w011ha"].transform(
        lambda group: group.interpolate(method="linear")
    )

    # rename to "wealth" and change unit to 1000s of euros
    wealth_data_full.rename(columns={"w011ha": "wealth"}, inplace=True)
    wealth_data_full["wealth"] = wealth_data_full["wealth"] / options["wealth_unit"]

    merged_data = merged_data.merge(wealth_data_full, on=["hid", "syear"], how="left")

    merged_data = merged_data[(merged_data["wealth"].notna())]

    merged_data.loc[merged_data["wealth"] < 0, "wealth"] = 0

    print(str(len(merged_data)) + " left after dropping people with missing wealth.")

    return merged_data


def create_hh_cons_equivalence_data(merged_data):
    """This function creates the household consumption equivalence scale following the
    OECD-modified equivalence scale."""
    # partner (>0 means has partner)
    merged_data["has_partner"] = 0
    merged_data.loc[merged_data["pgpartz"] >= 1, "has_partner"] = 1

    # number of children (<0 means 0 or "no info", which is treated as 0)
    merged_data["n_children"] = merged_data["hlc0043"]
    merged_data.loc[merged_data["n_children"] < 0, "n_children"] = 0

    # consumption equivalence scale
    merged_data["cons_equiv"] = (
        1 + 0.5 * merged_data["has_partner"] + 0.3 * merged_data["n_children"]
    )

    # drop missing values
    merged_data = merged_data[merged_data["cons_equiv"].notna()]

    print(
        str(len(merged_data))
        + " left after dropping people with missing consumption equivalence scale."
    )
    return merged_data


def deflate_wealth(merged_data, path_dict):
    """This function deflates the wealth variable using the consumer price index."""
    cpi_data = pd.read_csv(
        path_dict["intermediate_data"] + "cpi_base_2010.csv", index_col=0
    )
    merged_data = merged_data.merge(cpi_data, left_on="syear", right_index=True)
    merged_data["wealth"] = merged_data["wealth"] / merged_data["cpi"]
    return merged_data
