def generate_aux_options(options):
    options["min_ret_age"] = options["min_SRA"] - options["ret_years_before_SRA"]
    return options
