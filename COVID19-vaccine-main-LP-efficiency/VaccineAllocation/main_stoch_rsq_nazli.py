from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
import datetime as dt
import multiprocessing as mp
from Plotting_nazli import plot_from_file

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "transmission_May2020.csv",
    "austin_real_hosp_updated.csv",
    "austin_real_icu_updated.csv",
    "austin_hosp_ad_updated.csv",
    "austin_real_death_from_hosp_updated.csv",
    "austin_real_death_from_home.csv",
    "delta_prevalence.csv",
    "omicron_prevalence.csv",
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
thresholds = (-1, 5, 15, 25, 50)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
time_points = [dt.datetime(2020, 4, 30)]
time_points = [austin.cal.calendar.index(date) for date in time_points]
time_end = dt.datetime(2020, 4, 30)
seeds = [1, 2]
num_reps = 2
if __name__ == '__main__':
    for i in seeds:
        p = mp.Process(target=get_sample_paths, args=(austin, vaccines, 0.75, num_reps, i, time_points))
        p.start()
    for i in range(2):
        p.join()

    case_threshold = 200
    hosp_adm_thresholds = {"non_surge": (-1, -1, 10, 20, 20), "surge": (-1, -1, -1, 10, 10)}
    staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}
    ctp = CDCTierPolicy(austin, tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds)
    new_seeds = [3, 4]
    end_time = austin.cal.calendar.index(dt.datetime(2020, 8, 31))
    for i in range(len(seeds)):
        base_filename = f"{austin.path_to_input_output}/{seeds[i]}_"
        p = mp.Process(target=evaluate_single_policy_on_sample_path,
                       args=(austin, vaccines, ctp, end_time, new_seeds[i], num_reps, base_filename))
        p.start()
    for i in range(2):
        p.join()

    real_history_end_date = dt.datetime(2020, 5, 1)
    equivalent_thresholds = {"non_surge": (-1, -1, 28.57, 57.14, 57.14), "surge": (-1, -1, -1, 28.57, 28.57)}
    plot_from_file(seeds, num_reps, austin, real_history_end_date, equivalent_thresholds)
