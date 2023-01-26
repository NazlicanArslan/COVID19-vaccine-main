from pathlib import Path
import pandas as pd
import numpy as np
from Plot_Manager import Plot
from Report_Manager import Report
from InputOutputTools import import_stoch_reps_for_reporting

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"
real_data_file_names = {}

tier_colors = ["green", "blue", "yellow", "orange", "red"]
surge_colors = ['moccasin', 'pink']


def plot_from_file(seeds, num_reps, instance, real_history_end_date, equivalent_thresholds):
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, real_history_end_date, instance)
    for key, val in sim_outputs.items():
        print(key)
        if hasattr(instance, f"real_{key}"):
            real_data = getattr(instance, f"real_{key}")
        else:
            real_data = None

        if key == "ICU_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.dali_plot(policy_outputs["tier_history"], tier_colors)

        elif key == "ToIHT_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key, color=('k', 'silver'))
            plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                          ["non_surge", "surge"],
                                          equivalent_thresholds,
                                          tier_colors)

            plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", color=('k', 'silver'))
            plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                          ["non_surge", "surge"],
                                          policy_outputs["hosp_adm_thresholds"][0],
                                          tier_colors)
        elif key == "IH_history":
            real_data = [
                ai - bi for (ai, bi) in zip(instance.real_IH_history, instance.real_ICU_history)
            ]
            plot = Plot(instance,  real_history_end_date, real_data, val, f"{key}_average", color=('k', 'silver'))
            plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                          ["non_surge", "surge"],
                                          policy_outputs["staffed_bed_thresholds"][0],
                                          tier_colors)

            plot = Plot(instance, real_history_end_date, real_data, val, f"{key}", color=('k', 'silver'))
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)

        elif key == "ToIY_history":
            # ToDo: Fix the data reading part here:
            filename = 'austin_real_case.csv'
            real_data = pd.read_csv(
                str(instance.path_to_data / filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
            )["admits"]

            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["surge_history"], surge_colors)

        elif key == "D_history":
            real_data = np.cumsum(np.array([ai + bi for (ai, bi) in zip(instance.real_ToIYD_history, instance.real_ToICUD_history)]))
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        elif key == "ToIYD_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        elif key == "ToICUD_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        elif key == "ToRS_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)

            val = [np.cumsum(np.array(v), axis=0) for v in val]
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        elif key == "ToSS_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)

            val = [np.cumsum(np.array(v), axis=0) for v in val]
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)

        elif key == "S_history":
            real_data = None
            plot = Plot(instance, real_history_end_date, real_data, val, key)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)

def report_from_file(seeds, num_reps, instance, stats_start_date, stats_end_date):
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, instance)
    report = Report(instance, sim_outputs, policy_outputs, stats_start_date, stats_end_date)
    report.build_report()