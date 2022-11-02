import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from InputOutputTools import SimReplication_IO_list_of_arrays_var_names
from scipy.signal import find_peaks


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
            ax.scatter(x_axis, instance.real_hosp_icu[0:num_days])
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "ICU_history_a.png")
        elif var == "IH_history":
            real_hosp = [
                ai - bi for (ai, bi) in zip(instance.real_hosp, instance.real_hosp_icu)
            ]
            ax.scatter(x_axis, real_hosp[0:num_days])
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "IH_history_a.png")
        elif var == "ToIHT_history":
            ax.scatter(x_axis, instance.real_hosp_ad[0:num_days])
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "admission_history_a.png")
        elif var == "D_history":
            ax.scatter(x_axis, np.cumsum(instance.real_death_total[0:num_days]))
            ax.plot(x_axis, np.sum(np.sum(y_data, axis=1), axis=1))
            fig.autofmt_xdate()
            plt.savefig(instance.path_to_data / "death_history_a.png")

        # plt.show()


