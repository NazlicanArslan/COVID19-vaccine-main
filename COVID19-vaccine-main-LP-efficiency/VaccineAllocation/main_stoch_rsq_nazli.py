from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
from Plotting import plot_from_file, report_from_file

import datetime as dt
import multiprocessing as mp
import numpy as np

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "variant.json",
    "transmission.csv",
    "austin_real_hosp_updated.csv",
    "austin_real_icu_updated.csv",
    "austin_hosp_ad_updated.csv",
    "austin_real_death_from_hosp_updated.csv",
    "austin_real_death_from_home.csv",
    "variant_prevalence.csv"
)

tiers = TierInfo("austin", "tiers5_opt_Final.json")
vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)

###############################################################################
history_end_time = dt.datetime(2020, 4, 30)
simulation_end_time = dt.datetime(2020, 8, 31)
seeds = [100, 102]
new_seeds = [200, 202]

num_reps = 1
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, -1, 10, 20, 20), "surge": (-1, -1, -1, 10, 10)}
staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}
percentage_cases = 0.4
ctp = CDCTierPolicy(austin, tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds, percentage_cases)
thresholds = (-1, 0, 15, 25, 50)
tiers = TierInfo("austin", "tiers5_opt_Final.json")
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")

if __name__ == '__main__':
    for i in range(len(seeds)):
        base_filename = f"{seeds[i]}_"
        evaluate_single_policy_on_sample_path(austin,
                                              vaccines,
                                              ctp,
                                              austin.cal.calendar.index(simulation_end_time),
                                              austin.cal.calendar.index(history_end_time),
                                              new_seeds[i],
                                              num_reps,
                                              base_filename)
    # for i in range(len(seeds)):
    #     base_filename = f"{austin.path_to_input_output}/{seeds[i]}_"
    #     p = mp.Process(target=evaluate_single_policy_on_sample_path,
    #
    #                    args=(austin,
    #                          vaccines,
    #                          mtp,
    #                          austin.cal.calendar.index(simulation_end_time),
    #                          austin.cal.calendar.index(history_end_time),
    #                          new_seeds[i],
    #                          num_reps,
    #                          base_filename)
    #                    )
    #     p.start()
    # for i in range(len(seeds)):
    #     p.join()
    #
    equivalent_thresholds = {"non_surge": (-1, -1, 28.57, 57.14, 57.14), "surge": (-1, -1, -1, 28.57, 28.57)}
    policy_name_mtp = str(thresholds)
    policy_name_ctp = f"CDC_{case_threshold}_{hosp_adm_thresholds}_{staffed_thresholds}"
    plot_from_file(seeds, num_reps, austin, history_end_time, equivalent_thresholds, policy_name_ctp)
    report_from_file(seeds, num_reps, austin, history_end_time, simulation_end_time, policy_name_ctp)
