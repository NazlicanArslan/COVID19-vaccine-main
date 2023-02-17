from pathlib import Path
import pandas as pd
import numpy as np
from Plot_Manager import Plot, find_central_path

from Report_Manager import Report
from InputOutputTools import import_stoch_reps_for_reporting

base_path = Path(__file__).parent
path_to_plot = base_path / "plots"
real_data_file_names = {}

tier_colors = ["green", "blue", "yellow", "orange", "red"]
surge_colors = ['moccasin', 'pink']


def plot_from_file(seeds, num_reps, instance, real_history_end_date, equivalent_thresholds, policy_name):
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, real_history_end_date, instance, policy_name)

    central_path_id = find_central_path(sim_outputs["ICU_history"],
                                        sim_outputs["IH_history"],
                                        instance.real_IH_history,
                                        instance.cal.calendar.index(real_history_end_date))

    for key, val in sim_outputs.items():
        print(key)
        if hasattr(instance, f"real_{key}"):
            real_data = getattr(instance, f"real_{key}")
        else:
            real_data = None

        if key == "ICU_history":
            plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id)
            plot.dali_plot(policy_outputs["tier_history"], tier_colors, instance.icu)

            plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors, instance.icu)

        elif key == "ToIHT_history":
            if "surge_history" in policy_outputs.keys():
                plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id,
                            color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              equivalent_thresholds,
                                              tier_colors)
            if "surge_history" in policy_outputs.keys():
                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_sum", central_path_id,
                            color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              policy_outputs["hosp_adm_thresholds"][0],
                                              tier_colors)

            if "CDC" not in policy_outputs["policy_type"][0]:
                plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id,
                            color=('k', 'silver'))
                plot.horizontal_plot(policy_outputs["lockdown_thresholds"][0], tier_colors)

        elif key == "IH_history":
            real_data = [
                ai - bi for (ai, bi) in zip(instance.real_IH_history, instance.real_ICU_history)
            ]
            if "surge_history" in policy_outputs.keys():
                plot = Plot(instance, real_history_end_date, real_data, val, f"{key}_average", central_path_id,
                            color=('k', 'silver'))
                plot.changing_horizontal_plot(policy_outputs["surge_history"],
                                              ["non_surge", "surge"],
                                              policy_outputs["staffed_bed_thresholds"][0],
                                              tier_colors)

            plot = Plot(instance, real_history_end_date, real_data, val, f"{key}", central_path_id,
                        color=('k', 'silver'))

            plot.vertical_plot(policy_outputs["tier_history"], tier_colors, instance.hosp_beds)

        elif key == "ToIY_history" and "surge_history" in policy_outputs.keys():
            # ToDo: Fix the data reading part here:
            filename = 'austin_real_case.csv'
            real_data = pd.read_csv(
                str(instance.path_to_data / filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
            )["admits"]
            # real_data = np.array(real_data) * 1.92
            plot = Plot(instance, real_history_end_date, real_data, val, "ToIY_history_sum", central_path_id)
            plot.vertical_plot(policy_outputs["surge_history"], surge_colors, policy_outputs["case_threshold"])

        elif key == "D_history":
            real_data = np.cumsum(
                np.array([ai + bi for (ai, bi) in zip(instance.real_ToIYD_history, instance.real_ToICUD_history)]))
            plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id)
            plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        # elif key == "ToIYD_history":
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        # elif key == "ToICUD_history":
        #     plot = Plot(instance, real_history_end_date, real_data, val, key, central_path_id)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)
        #
        # elif key == "S_history":
        #     real_data = None
        #     plot = Plot(instance, real_history_end_date, real_data, val, key)
        #     plot.vertical_plot(policy_outputs["tier_history"], tier_colors)


def report_from_file(seeds, num_reps, instance, history_end_date, stats_end_date, policy_name):
    sim_outputs, policy_outputs = import_stoch_reps_for_reporting(seeds, num_reps, history_end_date, instance, policy_name)
    report = Report(instance, sim_outputs, policy_outputs, history_end_date, stats_end_date)
    report.build_report()
