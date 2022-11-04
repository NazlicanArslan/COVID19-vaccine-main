import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from InputOutputTools import SimReplication_IO_list_of_arrays_var_names
from pathlib import Path
from scipy.signal import find_peaks

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"


def plot_from_file(simulation_filename, instance):
    with open(simulation_filename) as file:
        data = json.load(file)

    for var in SimReplication_IO_list_of_arrays_var_names:
        y_data = data[var]
        print(var)
        print(len(y_data))
        plt.clf()
        fig, ax = plt.subplots()
        start_date = instance.start_date
        num_days = len(y_data)
        x_axis = [start_date + datetime.timedelta(days=x) for x in range(num_days)]
        if var == "ICU_history":
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days], color='maroon', zorder=100, s=15)
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1), linewidth=2, zorder=50)
            ax.set_ylabel('COVID-19 ICU Patients')
            fig.autofmt_xdate()
            plt.savefig(path_to_plot / "ICU_history_a.png")
        elif var == "IH_history":
            real_hosp = [
                ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)
            ]
            ax.scatter(x_axis, real_hosp[0:num_days], color='maroon', zorder=100, s=15)
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            ax.set_ylabel('COVID-19 Hospitalizations')
            fig.autofmt_xdate()
            plt.savefig(path_to_plot / "IH_history_a.png")
        elif var == "ToIHT_history":
            # plot the 7-day moving average.
            moving_avg_len = instance.config["moving_avg_len"]
            ToIHT_moving = [np.sum(y_data, axis=(1, 2))[i: min(i + moving_avg_len, num_days)].mean() for i in range(num_days)]
            instance.real_hosp_ad = np.array(instance.real_hosp_ad)
            real_hosp_moving = [instance.real_hosp_ad[0:num_days][i: min(i + moving_avg_len, num_days)].mean() for i in range(num_days)]

            ax.scatter(x_axis, real_hosp_moving, color='maroon', zorder=100, s=15)
            ax.plot(x_axis, ToIHT_moving)
            ax.set_ylabel('COVID-19 Hospital Admissions\n(Seven-day Average)')
            fig.autofmt_xdate()
            plt.savefig(path_to_plot / "admission_history_a.png")
        elif var == "D_history":
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]), color='maroon', zorder=100, s=15)
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(path_to_plot / "death_history_a.png")

        # plt.show()



