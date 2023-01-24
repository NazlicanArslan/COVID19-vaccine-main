import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from InputOutputTools import SimReplication_IO_list_of_arrays_var_names
from pathlib import Path
from Plot_Manager import Plot

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"
real_data_file_names = {}

tier_colors = ["green", "blue", "yellow", "orange", "red"]
surge_colors = ['moccasin', 'pink']
plot_var_names = ["ICU_history", "ToIY_history", "ToIHT_history", "IH_history"]


def plot_from_file(simulation_filename, policy_filename, instance, real_history_end_date, equivalent_thresholds):
    start_date = instance.start_date

    with open(simulation_filename) as file:
        data = json.load(file)

    with open(policy_filename) as file:
        policy_data = json.load(file)

    for var in plot_var_names:
        print(var)
        if var in SimReplication_IO_list_of_arrays_var_names:
            y_data = data[var]
        else:
            print('The data is not outputted')
            pass
        if var == "ICU_history":
            real_data = instance.real_ICU_history
            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, var)
            plot.vertical_plot(policy_data["tier_history"], tier_colors)
        elif var == "ToIY_history":
            # ToDo: Fix the data reading part here:
            filename = 'austin_real_case.csv'
            real_data = pd.read_csv(
                str(instance.path_to_data / filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
            )["admits"]

            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, var)
            plot.vertical_plot(policy_data["surge_history"], surge_colors)
        elif var == "ToIHT_history":
            real_data = np.array(instance.real_ToIHT_history)
            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, var, 'k')
            plot.changing_horizontal_plot(policy_data["surge_history"],
                                          ["non_surge", "surge"],
                                          equivalent_thresholds,
                                          tier_colors)

            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, f"{var}_sum", 'k')
            plot.changing_horizontal_plot(policy_data["surge_history"],
                                          ["non_surge", "surge"],
                                          policy_data["hosp_adm_thresholds"],
                                          tier_colors)

        elif var == "D_history":
            real_data = real_data = [
                        ai - bi for (ai, bi) in zip(instance.real_IYD_history, instance.real_ICUD_history)
                    ]
            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, var, 'b')
            plot.vertical_plot(policy_data["tier_history"], tier_colors)
        elif var == "IH_history":
            real_data = [
                ai - bi for (ai, bi) in zip(instance.real_IH_history, instance.real_ICU_history)
            ]
            plot = Plot(instance, policy_data, real_history_end_date, real_data, y_data, f"{var}_average", 'k')
            plot.changing_horizontal_plot(policy_data["surge_history"],
                                          ["non_surge", "surge"],
                                          policy_data["staffed_bed_thresholds"],
                                          tier_colors)


