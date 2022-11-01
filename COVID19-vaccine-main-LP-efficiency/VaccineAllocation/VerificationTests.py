import copy
from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

# Import other Python packages
import numpy as np
import time

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "transmission.csv",
    "austin_real_hosp_updated.csv",
    "delta_prevalence.csv",
    "omicron_prevalence.csv",
    "variant_prevalence.csv",
)

tiers = TierInfo("austin", "tiers5_opt_Final.json")

vaccines = Vaccine(
    austin,
    "austin",
    "vaccines.json",
    "booster_allocation_fixed.csv",
    "vaccine_allocation_fixed.csv",
)

start = time.time()

thresholds = (-1, 100, 200, 500, 1000)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
rep = SimReplication(austin, vaccines, mtp, 500)
rep.simulate_time_period(945)
print(rep.compute_rsq())
print(rep.compute_cost())
print(time.time() - start)

# -1.2246225612134243
# 841
# 7.5331807136535645

# python3 -m pdb main_allocation.py austin -f setup_data_Final.json -t tiers5_opt_Final.json -train_reps 1 -test_reps 1 -f_config austin_test_IHT.json -n_proc 1 -tr transmission.csv -hos austin_real_hosp_updated.csv  -v_allocation vaccine_allocation_fixed.csv -n_policy=7  -v_boost booster_allocation_fixed.csv # -gt [-1,5,15,30,50]
