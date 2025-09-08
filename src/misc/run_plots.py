from set_paths import create_path_dict
from set_styles import set_plot_defaults

# paths, create directories
path_dict = create_path_dict(define_user=True)
show_plots = False
save_plots = True

# Set plot defaults
set_plot_defaults()

# Pension rates plot
from misc.plots.pension_rates import plot_pension_rates
plot_pension_rates(path_dict, show=show_plots, save=save_plots)

# 2007 reform plot
from misc.plots.reform_2007_plot import plot_SRA_2007_reform
plot_SRA_2007_reform(path_dict, show=show_plots, save=save_plots)

print("Misc plotting completed.")