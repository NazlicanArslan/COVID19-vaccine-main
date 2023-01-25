###############################################################################

# main_param_fitting_example.py
# This document contains examples of how to use the least square fitting tool.
# Nazlican Arslan 2022

###############################################################################

from DataObjects import City, TierInfo, Vaccine
from ParamFittingTools import ParameterFitting

# Import other Python packages
import numpy as np
import datetime as dt
import pandas as pd

###############################################################################
# We need to define city, vaccine and tier object as the least square fit will
# run the deterministic simulation model.

austin = City(
    "austin",
    "austin_test_IHT.json",
    "calendar.csv",
    "setup_data_Final.json",
    "variant_lsq.json",
    "transmission_lsq_estimated_data.csv",
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


# We need to define time blocks where the transmission reduction (or the behaviour in the population) changes:
change_dates = [dt.date(2020, 2, 15),
                dt.date(2020, 3, 24),
                dt.date(2020, 5, 21),
                dt.date(2020, 6, 26),
                dt.date(2020, 8, 20),
                dt.date(2020, 10, 29),
                dt.date(2020, 11, 30),
                dt.date(2020, 12, 31),
                dt.date(2021, 1, 12),
                dt.date(2021, 3, 13),
                dt.date(2021, 6, 20),
                dt.date(2021, 7, 28),
                dt.date(2021, 8, 10),
                dt.date(2021, 9, 28),
                dt.date(2021, 10, 12),
                dt.date(2021, 11, 20),
                dt.date(2022, 1, 6),
                dt.date(2022, 2, 1),
                dt.date(2022, 3, 6),
                dt.date(2022, 3, 30)]

# We don't fit all the transmission reduction values from scratch as the least square fit cannot handle too many
# decision variables. We used the existing fitted values for the earlier fit data. In transmission_reduction and
# cocoon lists you can input the already existing values and for the new dates you need to estimate just input None.
transmission_reduction = [0.052257,
                          0.787752,
                          0.641986,
                          0.827015,
                          0.778334,
                          0.75298,
                          0.674321,
                          0.801538,
                          0.811144,
                          0.6849,
                          0.539101,
                          0.625945789,
                          0.665369334,
                          0.703965861,
                          0.538824759,
                          0.43,
                          0.46,
                          0.2,
                          0.0]
# for the high risk groups uses cocoon instead of contact reduction
cocoon = np.array([0,
                   0.787752,
                   0.787752,
                   0.827015,
                   0.827015,
                   0.787752,
                   0.827015,
                   0.801538,
                   0.811144,
                   0.6849,
                   0.539101,
                   0.625945789,
                   0.665369334,
                   0.703965861,
                   0.538824759,
                   0.43,
                   0.46,
                   0.2,
                   0.0])

end_date = []
for idx in range(len(change_dates[1:])):
    end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))

table = pd.DataFrame(
    {
        "start_date": change_dates[:-1],
        "end_date": end_date,
    }
)
# The initial guess of the variables to estimate:
initial_guess = np.array([3.5, 0.1, 3, 0.1])
# Lower and upper bound tuple:
x_bound = ([0, 0, 0, 0], [5, 1, 5, 1])

# Austin weights for the least-square fit:
# You can input the data you would like to use in the process and corresponding weights. Different data have different
# scales that's why we use weights.
objective_weights = {"IH_history": 1,
                     "ICU_history": 1.5,
                     "ToIHT_history": 7.583296,
                     "ToICUD_history": 7.583296 * 5,
                     "ToIYD_history": 7.583296 * 5}

# objective_weights = {"ToIHT_history": 1}

# We generally use the least square fit to find transmission reduction and cocooning in a population. But time to time
# we may need to estimate other parameters. Fitting transmission reduction is optional. In the current version of
# the parameter fitting you can input the name of parameter you would like to fit, and you don't need to change anything
# else in the source code.
variables = ["omicron alpha_gamma_ICU", "omicron alpha_IH", "omicron alpha_mu_ICU", "omicron alpha_IYD"]


# We can define the time frame we would like to use data from as follows:
time_frame = (austin.cal.calendar.index(dt.datetime(2021, 12, 4)), austin.cal.calendar.index(dt.datetime(2022, 3, 30)))

param_fitting = ParameterFitting(austin,
                                 vaccines,
                                 variables,
                                 initial_guess,
                                 x_bound,
                                 objective_weights,
                                 time_frame,
                                 change_dates,
                                 transmission_reduction,
                                 cocoon)
solution = param_fitting.run_fit()

