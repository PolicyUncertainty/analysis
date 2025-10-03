def generate_working_hours(data):
    """This function creates a working hours variable from pgvebzeit in soep-pgen.

    This means working hours = contractual hours per week. The function drops
    observations where working hours are missing.

    """
    data = data.rename(columns={"pgvebzeit": "working_hours"})
    data = data[data["working_hours"] >= 0]
    print(str(len(data)) + " left after dropping people with missing working hours.")
    return data
