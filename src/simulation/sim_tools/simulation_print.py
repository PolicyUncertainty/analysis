def start_simulation_print(model_name = None, sra_30=None, sra_63=None, uncertainty=None, misinformation=None, load_model=None, load_solution=None, load_df=None):
    print(f"Start simulation for model {model_name} with \n"
          f"SRA at 30: {sra_30}, SRA at 63: {sra_63}, \n"
          f"Uncertainty: {uncertainty}, Misinformation: {misinformation}, \n")
    print("Simulation settings:\n")
    if load_model:
        print("1. Loading existing model configuration.\n")
    else:
        print("1. Creating new model configuration.\n")
    if load_solution:
        print("2. Loading existing solution.\n")
    else:
        print("2. Creating new solution. This will take some time.\n")
    if load_df:
        print("3. Loading existing Result DataFrame.\n")
    else:
        print("3. Creating new Result DataFrame.\n")