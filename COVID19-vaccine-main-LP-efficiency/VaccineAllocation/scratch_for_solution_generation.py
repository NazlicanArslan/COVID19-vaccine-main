# sys.path.extend(['/Users/linda/Dropbox/RESEARCH/Summer2022/COVID19-vaccine-main/COVID19-vaccine-main-LP-efficiency/VaccineAllocation'])
#
# import copy
from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

# Import other Python packages
import numpy as np
import pandas as pd

###############################################################################

austin = City(city="austin",
              config_filename="austin_test_IHT.json",
              calendar_filename="calendar.csv",
              setup_filename="setup_data_Final.json",
              transmission_filename="transmission.csv",
              hospitalization_filename="austin_real_hosp_updated.csv",
              hosp_icu_filename="austin_real_icu_updated.csv",
              hosp_admission_filename="austin_hosp_ad_updated.csv",
              death_from_hosp_filename="austin_real_death_from_hosp_updated.csv",
              death_from_home_filename="austin_real_cum_total_death.csv",
              delta_prevalence_filename="delta_prevalence.csv",
              omicron_prevalence_filename="omicron_prevalence.csv",
              variant_prevalence_filename="variant_prevalence.csv")

tiers = TierInfo("austin", "tiers_test.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

###############################################################################

# scratch for feasible solution generation
#0.44

kappa = 1
cutoff = -1

threshold = (-1, cutoff, cutoff, cutoff, cutoff)
mtp = MultiTierPolicy(austin, tiers, threshold, None)
rep = SimReplication(austin, vaccines, mtp, -1)

kappa_history = []

rep.simulate_time_period(783)
for t in range(1, 945-783+1):
    if np.array(rep.ICU_history).sum(axis=(1,2))[-1] > np.array(rep.ICU_history).sum(axis=(1,2))[-2]:
        kappa = 1
    else:
        kappa = 0
    rep.policy.tiers[4]["school_closure"] = 0
    rep.policy.tiers[4]["cocooning"] = kappa
    rep.policy.tiers[4]["transmission_reduction"] = kappa
    # print(kappa)
    kappa_history.append(kappa)
    rep.simulate_time_period(783 + t)

print(np.average(kappa_history))
print(rep.compute_feasibility())
print(np.array(rep.ICU_history).sum(axis=(1, 2))[rep.t_historical_data_end:])

breakpoint()

###############################################################################

threshold = (-1, np.inf, np.inf, np.inf, np.inf)
threshold = (-1, 17, 17, 17, 17)
threshold = (-1, -1, -1, -1, -1)
mtp = MultiTierPolicy(austin, tiers, threshold, None)
rep = SimReplication(austin, vaccines, mtp, -1)
rep.simulate_time_period(945)

print(rep.compute_feasibility())

ICU_history = np.array(rep.ICU_history).sum(axis=(1, 2))[rep.t_historical_data_end:]

total_daily_admissions = np.array(rep.ToIHT_history).sum((1, 2))
moving_avg_admissions = pd.Series(total_daily_admissions).rolling(7).mean()
moving_avg_admissions = np.array(moving_avg_admissions)[rep.t_historical_data_end:]

day_ICU_violation = np.argwhere(ICU_history > 150)
if len(day_ICU_violation) > 0:
    first_day_ICU_violation = day_ICU_violation[0][0]
    print(moving_avg_admissions[first_day_ICU_violation])

breakpoint()

###############################################################################

# thresholds_array = OptTools.thresholds_generator((0, 14, 1), (0, 100, 10), (0, 100, 10), (0, 100, 20))

thresholds = OptTools.thresholds_generator((0, 10, 1), (0, 100, 1), (0, 100, 10), (0, 100, 10))

new_thresholds = []

for threshold in thresholds:
    if threshold[1] == threshold[2] and threshold[3] == threshold[4]:
        new_thresholds.append(threshold)

print(len(new_thresholds))

tier_histories = []
feasibilities = []
costs = []

for threshold in new_thresholds:
    mtp = MultiTierPolicy(austin, tiers, threshold, None)
    rep = SimReplication(austin, vaccines, mtp, -1)
    rep.simulate_time_period(945)

    # tier_histories.append(mtp.tier_history)
    feasible_flag = rep.compute_feasibility()
    feasibilities.append(feasible_flag)
    # costs.append(rep.compute_cost())

    print(threshold, feasible_flag)

# tier_histories_array = []

# for tier_history in tier_histories:
#     x = [tier for tier in tier_history if tier is not None]
#     tier_histories_array.append(x)

breakpoint()

###############################################################################

highest_stage_possible = 4
small_uniform_decrease = 10
large_uniform_decrease = 100

small_decrease_only = False
decrease = 0

breakpoint()

###############################################################################

thresholds = (-1, 100, 100, 900, 900)

mtp = MultiTierPolicy(austin, tiers, thresholds, None)
rep = SimReplication(austin, vaccines, mtp, -1)
rep.simulate_time_period(945)

cap = rep.instance.icu

feasibility_flag = rep.compute_feasibility()

while feasibility_flag == False:

    tier_history = np.array([tier for tier in mtp.tier_history if tier is not None])
    ICU_history = np.array(rep.ICU_history)[len(rep.ICU_history)-len(tier_history):].sum(axis=(1, 2))

    highest_stage = np.max(tier_history)
    first_day_highest_stage = np.min(np.argwhere(tier_history == highest_stage))
    violation_days = np.argwhere(ICU_history > cap)
    if len(violation_days) > 0:
        first_day_violation = np.min(violation_days)

    print(thresholds)
    print(feasibility_flag)
    print(rep.compute_cost())
    print(highest_stage)
    print(first_day_highest_stage)
    print(first_day_violation)
    print(len(violation_days))

    # if first_day_violation <= first_day_highest_stage:
    if True:
        new_thresholds = [-1]
        for i in range(1, highest_stage+1):
            new_thresholds.append(thresholds[i] - small_uniform_decrease)
        if highest_stage < highest_stage_possible:
            for i in range(highest_stage+1, highest_stage_possible+1):
                if small_decrease_only == False:
                    new_trigger_value = thresholds[i] - large_uniform_decrease
                else:
                    new_trigger_value = thresholds[i] - small_decrease_only
                if new_trigger_value >= new_thresholds[i-1]:
                    new_thresholds.append(new_trigger_value)
                else:
                    new_thresholds.append(thresholds[i])
        else:
            for i in range(highest_stage+1, highest_stage_possible+1):
                new_thresholds.append(thresholds[i])
        new_thresholds = tuple(new_thresholds)

    if new_thresholds == thresholds:
        if small_decrease_only == False:
            small_decrease_only = True
        else:
            break

    mtp = MultiTierPolicy(austin, tiers, new_thresholds, None)
    rep = SimReplication(austin, vaccines, mtp, -1)
    rep.simulate_time_period(945)
    feasibility_flag = rep.compute_feasibility()

    # breakpoint()

    thresholds = new_thresholds

tier_history = np.array([tier for tier in mtp.tier_history if tier is not None])
ICU_history = np.array(rep.ICU_history)[len(rep.ICU_history)-len(tier_history):].sum(axis=(1, 2))

highest_stage = np.max(tier_history)
first_day_highest_stage = np.min(np.argwhere(tier_history == highest_stage))
violation_days = np.argwhere(ICU_history > cap)
if len(violation_days) > 0:
    first_day_violation = np.min(violation_days)
else:
    first_day_violation = None

print(thresholds)
print(feasibility_flag)
print(rep.compute_cost())
print(highest_stage)
print(first_day_highest_stage)
print(first_day_violation)
print(len(violation_days))

breakpoint()