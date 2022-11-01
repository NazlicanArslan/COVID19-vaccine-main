import numpy as np
import numpy.linalg as la
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import datetime as dt
# get A and its inverse   

# hosp_ad_csv = pd.read_csv("../instances/cook/IL_hosp_ad.csv")
# y = hosp_ad_csv["admits"]
# ny = len(y)
# n = len(y) + 6

hosp_csv = pd.read_csv("./instances/cook/cook_hosp_region_sum_estimated.csv", parse_dates=['date'])
# hosp_csv = pd.read_csv("./instances/cook/smoothed_estimated_no_may_hosp.csv", parse_dates=['date'])
icu_csv = pd.read_csv("./instances/cook/icu.csv", parse_dates=['date'])


hosp_csv_first = hosp_csv.query("date < @ dt.datetime(2020, 6, 13)")
icu_csv_first = icu_csv.query("date < @ dt.datetime(2020, 6, 13)")

hosp = hosp_csv_first["hospitalized"].reset_index(drop=True)
icu = hosp_csv_first["hospitalized"].reset_index(drop=True)
print(hosp)
m = gp.Model("admits")
# m.params.NonConvex = 2
len_x = len(hosp)
leading_days = 18
x = m.addVars(len_x + leading_days, vtype=GRB.CONTINUOUS)

y = m.addVars((len(x) - 1) * 2, vtype=GRB.CONTINUOUS, lb=0)


m.setObjective(gp.quicksum(y[i] for i in range(len(y))), GRB.MINIMIZE)

counter = 0
for i in range(len(x) - 1):
    m.addConstr(x[i + 1] - x[i] == y[counter] - y[counter + 1])
    counter += 2
# counter1 = 0
# counter2 = 1
# leading_days_plus = 0

# while counter2 != 38:
#     # print("counter1", counter1)
#     print("counter2", counter2)
#     linexpr = 0
#     for j in range(leading_days + leading_days_plus):
#         linexpr += x[counter2 + leading_days - j]
#     if counter2 <= 27:
#         if counter2 % 9 == 0:
#             leading_days_plus += 2
#     print("leading_day", leading_days_plus)
#     counter2 += 1
#     m.addConstr(linexpr == hosp[counter2-1])

for i in range(len_x):
    linexpr = 0
    for j in range(leading_days):
        linexpr += x[i + j]
    m.addConstr(linexpr == hosp[i])
    

m.write("./test_updated.lp")
m.optimize()
for i in range(len_x + leading_days):
    print(x[i].X)

cook_ad_data = pd.read_csv("./instances/cook/cook_hosp_ad_region_sum.csv", index_col=0)
start_date = dt.datetime(2020,6,12)
times = [(start_date - dt.timedelta(days=x)).strftime("%m/%d/%y") for x in range(len_x + leading_days)][::-1]
admission_data = [x[i].X for i in range(len_x + leading_days)]
syn_admission_data = pd.DataFrame({"date": times, "hospitalized":admission_data})

cook_ad_data = pd.concat([syn_admission_data, cook_ad_data]).reset_index(drop=True)
cook_ad_data.to_csv("./instances/cook/cook_hosp_ad_region_sum_18_syn.csv")





# # print(y)

# A = (np.tril(np.ones((n,n)),-1) - np.tril(np.ones((n,n)),-8))/7.
# A = A[7:, :]

# # pA = la.pinv(A)
# # print(np.dot(pA, y))
# # print(A)

