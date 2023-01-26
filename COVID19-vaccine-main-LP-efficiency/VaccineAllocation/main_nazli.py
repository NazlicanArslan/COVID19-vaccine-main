from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from OptTools import evaluate_single_policy_on_sample_path, get_sample_paths
from InputOutputTools import export_rep_to_json
from Plotting import plot_from_file

# Import other Python packages
import numpy as np
import datetime as dt

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
thresholds = (-1, 5, 15, 25, 50)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, -1, 10, 20, 20), "surge": (-1, -1, -1, 10, 10)}
staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}

# CDC threshold uses 7-day sum of hospital admission per 100k. The equivalent values if we were to use 7-day avg.
# hospital admission instead are as follows. We use equivalent thresholds to plot and evaluate the results in our
# indicator. I used the same CDC thresholds all the time but if we decide to optimize CDC threshold, we can calculate
# the equivalent values in the model and save to the policy.json.
equivalent_thresholds = {"non_surge": (-1, -1, 28.57, 57.14, 57.14), "surge": (-1, -1, -1, 28.57, 28.57)}
ctp = CDCTierPolicy(austin, tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds)
seed = -1
rep = SimReplication(austin, vaccines, ctp, seed)

time_end = austin.cal.calendar.index(dt.datetime(2022, 5, 30))
fixed_kappa_end_time = austin.cal.calendar.index(dt.datetime(2022, 3, 30))
rep.simulate_time_period(time_end, fixed_kappa_end_time)
base_filename = f"{austin.path_to_input_output}/{seed}_1_{dt.datetime(2022, 3, 30).date()}"
export_rep_to_json(
    rep,
    f"{base_filename}_sim_updated.json",
    f"{base_filename}_v0.json",
    f"{base_filename}_v1.json",
    f"{base_filename}_v2.json",
    f"{base_filename}_v3.json",
    f"{base_filename}_policy.json"
)
fixed_kappa_end_time = dt.datetime(2022, 3, 30)
plot_from_file([seed], 1, austin, fixed_kappa_end_time, equivalent_thresholds)