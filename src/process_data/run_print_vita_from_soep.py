import pandas as pd
import numpy as np
from set_paths import create_path_dict



def print_time_invariant_data_from_ppath(soep_path, pid):
    ppath_data = pd.read_stata(
        f"{soep_path}/ppath.dta",
        columns=[
            "pid",
            "rv_id",
            "sex",
            "gebjahr",
            "germborn",
            "birthregion_ew"
        ],
        convert_categoricals=False,
    )
    ppath_data_ind = ppath_data[ppath_data["pid"] == pid]
    ppath_data_ind = ppath_data_ind.reset_index(drop=True)
    if ppath_data_ind.shape[0] == 0:
        print(f"\n  pid {pid} not found in ppath.")
        return 
    else:
        print(f"\n Gathering time-invariant data for pid: {pid}...")
    
    rv_id = ppath_data_ind["rv_id"].values[0] # <0 missing
    rv_id_map = {-2: "missing"}
    print(f"rv_id: {rv_id_map.get(rv_id, rv_id)}")

    sex = ppath_data_ind["sex"].values[0] # 1 man, 2 woman, <0 other
    sex_map = {1: "man", 2: "woman"}
    print(f"sex: {sex_map.get(sex, 'other')}")

    birth_year = ppath_data_ind["gebjahr"].values[0] # <0 missing
    birth_year_map = {-1: "missing"}
    print(f"birth_year: {birth_year_map.get(birth_year, birth_year)}")

    german_born = ppath_data_ind["germborn"].values[0] # 1 = yes, -2 = no
    west_german_born = ppath_data_ind["birthregion_ew"].values[0] # 21 = yes, 22 = no, <0 missing
    if german_born == 1:
        birth_place_string = "Germany"
        if west_german_born == 21:
            birth_place_string += " (West)"
        elif west_german_born == 22:
            birth_place_string += " (East)"
    else:
        birth_place_string = "Abroad"
    print(f"birth_place: {birth_place_string}")

    return rv_id, ppath_data_ind

def print_spell_data_from_artkalen(soep_path, pid):
    spell_data = pd.read_stata(
        f"{soep_path}/artkalen.dta",
        columns=[
            "pid",
            "spelltyp",
            "begin",
            "end",
            "zensor",
        ],
        convert_categoricals=True,
    )
    spell_data_ind = spell_data[spell_data["pid"] == pid]
    spell_data_ind = spell_data_ind.reset_index(drop=True)
    spell_data_ind = spell_data_ind.drop(columns=["pid"])
    if spell_data_ind.shape[0] == 0:
        print(f"\n  pid {pid} not found in artkalen.")
        return 
    else:
        print(f"\n Gathering spell data for pid: {pid} from artkalen...")

    try: 
        from tabulate import tabulate
        print(tabulate(spell_data_ind, headers="keys", tablefmt="psql", showindex=False))
    except: 
        print("\"tabulate\" module not installed. Printing unformatted DataFrame instead.") 
        print(spell_data_ind)
    return     
    
def print_survey_data_from_pgen_and_pequiv(soep_path, pid):
    pgen_data = pd.read_stata(
        f"{soep_path}/pgen.dta",
        columns=[
            "pid",
            "syear",
            "pgemplst",
            "pgstib",
            "pgpartz",
            "pglabgro",
        ],
        convert_categoricals=True,
    )
    pgen_data_ind = pgen_data[pgen_data["pid"] == pid]
    pequiv_data = pd.read_stata(
        f"{soep_path}/pequiv.dta",
        columns=[
            "pid",
            "syear",
            "igrv1",
        ],
        convert_categoricals=True,
    )
    pequiv_data_ind = pequiv_data[pequiv_data["pid"] == pid]
    merged_data = pd.merge(pgen_data_ind, pequiv_data_ind, on=["pid", "syear"], how="outer")
    if merged_data.shape[0] == 0:
        print(f"\n  pid {pid} not found in pgen or pequiv.")
        return
    else:
        print(f"\n Gathering survey data for pid: {pid} from pgen and pequiv...")
    merged_data = merged_data.reset_index(drop=True)
    merged_data.rename(columns={"pgemplst": "employment_state_(pgen)", "pgstib": "job_type_(pgen)"}, inplace=True)
    merged_data["has_partner"] = merged_data.loc[:, "pgpartz"] != 0
    merged_data["labor_income_(pgen)"] = merged_data["pglabgro"]
    merged_data["state_pension_income_(pequiv)"] = merged_data["igrv1"]
    columns = [
        "syear",
        "employment_state_(pgen)",
        "job_type_(pgen)",
        "has_partner",
        "labor_income_(pgen)",
        "state_pension_income_(pequiv)",
    ]
    merged_data = merged_data[columns]
    merged_data = merged_data.reset_index(drop=True)
    try: 
        from tabulate import tabulate
        print(tabulate(merged_data, headers="keys", tablefmt="psql", showindex=False))
    except: 
        print("\"tabulate\" module not installed. Printing unformatted DataFrame instead.") 
        print(merged_data)
    return merged_data

def print_admin_data_from_soep_rv(soep_rv_path, rv_id):
    rv_data = pd.read_stata(
        f"{soep_rv_path}/vskt/SUF.SOEP-RV.VSKT.2020.var.1-0.dta",
        convert_categoricals=True,
    )
    rv_data_ind = rv_data[rv_data["rv_id"] == rv_id]
    print(f"\n Gathering administrative data for rv_id: {rv_id} from SOEP-RV...")
    print("Note: For the codebook of the STATUS variables refer the the SOEP-RV documentation.")
    print("Note: Status 4 and 5 (marginal employment) omitted.")
    columns = ["JAHR", "MONAT", "STATUS_1", "STATUS_1_EGPT", "STATUS_2", "STATUS_2_EGPT", "STATUS_3", "STATUS_3_EGPT"]
    rv_data_ind = rv_data_ind[columns]
    rv_data_ind = rv_data_ind.reset_index(drop=True)
    try: 
        from tabulate import tabulate
        print(tabulate(rv_data_ind, headers="keys", tablefmt="psql", showindex=False))
    except: 
        print("\"tabulate\" module not installed. Printing unformatted DataFrame instead.") 
        print(rv_data_ind)
    return rv_data_ind


def print_vita_from_soep(paths=False, pid=False):
    """This function prints the vita of a given pid in the SOEP data. It prints the time-invariant data from ppath,
    the spell data from artkalen, and the survey data from pgen and pequiv. If the pid has a valid rv_id, it also
    prints the administrative data from the SOEP-RV."""
    
    if paths is False:
        paths = create_path_dict(define_user=True)
    if pid is False:
        pid = int(input("Enter pid: "))

    invariant_df = print_time_invariant_data_from_ppath(soep_path=paths["soep_c38"], pid=pid)
    spell_df = print_spell_data_from_artkalen(soep_path=paths["soep_c38"], pid=pid)
    survey_df = print_survey_data_from_pgen_and_pequiv(soep_path=paths["soep_c38"], pid=pid)
    
    # Check if rv_id is valid and load and print administrative data if it is
    if invariant_df is not None:
        rv_id = invariant_df[1]["rv_id"].values[0]
        if rv_id > 0:
            string_in = input(f"\n  rv_id {rv_id} is valid. Print out public pension panel data (vskt) for pid {pid} (y/n)?\n")
            if string_in == "y":
                soep_rv_path = paths["soep_rv"]
                rv_data = print_admin_data_from_soep_rv(soep_rv_path=soep_rv_path, rv_id=rv_id)
    return invariant_df, spell_df, survey_df

# example pid = 14901
print_vita_from_soep()