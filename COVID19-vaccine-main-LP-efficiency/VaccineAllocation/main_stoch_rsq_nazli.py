import copy

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import export_rep_to_json
from OptTools import get_sample_paths
from Plotting_nazli import plot_from_file
import datetime as dt
import multiprocessing as mp
# Import other Python packages
import numpy as np

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "transmission.csv",
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

if __name__ == '__main__':
    for i in range(2):
        p = mp.Process(target=get_sample_paths, args=(austin, vaccines, 0.75, 5, seeds[i], time_points))
        p.start()
    for i in range(2):
        p.join()

