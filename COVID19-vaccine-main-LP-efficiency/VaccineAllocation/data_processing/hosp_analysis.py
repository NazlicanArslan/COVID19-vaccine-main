import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


hosp_ad_csv = pd.read_csv("./instances/cook/IL_hosp_ad_region_sum.csv", parse_dates=['date'], index_col=0)
hosp_ad = hosp_ad_csv["hospitalized"]

cook_total_beds_region_sum = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", parse_dates=['date'], index_col=0)
cook_beds = cook_total_beds_region_sum["hospitalized"]
print(cook_beds)
cook_ad_csv = pd.read_csv("./instances/cook/cook_hosp_ad_region_sum.csv", parse_dates=["date"],index_col=0)

cook_ad = cook_ad_csv["hospitalized"]

il_hosp_csv = pd.read_csv("./instances/cook/IL_hosp.csv", parse_dates=["date"],index_col=0)
il_icu_csv = pd.read_csv("./instances/cook/IL_hosp_icu.csv", parse_dates=["date"],index_col=0)
cook_hosp_csv = pd.read_csv("./instances/cook/cook_hosp_estimated.csv", parse_dates=["date"],index_col=0)
cook_hosp_csv_copy = cook_hosp_csv.query("date <= @dt.datetime(2022, 4, 7) and date >= @dt.datetime(2020, 6, 13)")
hosp_csv_copy = il_hosp_csv.query("date <= @dt.datetime(2022, 4, 7) and date >= @dt.datetime(2020, 6, 13)")
# icu_csv_copy = il_icu_csv.query("date <= @dt.datetime(2022, 4, 7) and date >= @dt.datetime(2020, 6, 13)")

# hosp_csv_first = il_hosp_csv.query("date < @ dt.datetime(2020, 6, 13)")
# icu_csv_first = il_icu_csv.query("date < @ dt.datetime(2020, 6, 13)")

# ad_ratio = cook_ad / hosp_ad



il_hosp = hosp_csv_copy["hospitalized"]
# il_icu = icu_csv_copy["hospitalized"]

# cook_hosp_first = hosp_csv_first["hospitalized"] * 0.4
# cook_icu_first = icu_csv_first["hospitalized"] * 0.4
# cook_hosp = np.array(il_hosp) * np.array(ad_ratio)
# cook_icu = np.array(il_icu) * np.array(ad_ratio)

# cook_hosp_final = np.append(cook_hosp_first, cook_hosp)
# cook_icu_final = np.append(cook_icu_first, cook_icu)


# cook_hosp_df = pd.DataFrame({"date":il_hosp_csv["date"][:726], "hospitalized": np.round_(cook_hosp_final)})
# cook_icu_df = pd.DataFrame({"date":il_hosp_csv["date"][:726], "hospitalized": np.round_(cook_icu_final)})

# cook_hosp_df.to_csv("../instances/cook/updated_cook_hosp.csv")
# cook_icu_df.to_csv("../instances/cook/updated_cook_icu.csv")


hosp_csv = pd.read_csv("./instances/cook/updated_cook_hosp.csv", parse_dates=['date'], index_col=0)
icu_csv = pd.read_csv("./instances/cook/updated_cook_icu.csv", parse_dates=['date'],index_col=0)

# hosp_csv_copy = hosp_csv.query("date <= @dt.datetime(2022, 4, 7) and date >= @dt.datetime(2020, 6, 13)")
# icu_csv_copy = icu_csv.query("date <= @dt.datetime(2022, 4, 7) and date >= @dt.datetime(2020, 6, 13)")
for i in range(7, 20):
    # print(il_hosp)
    cook_ad_csv["SMA" + str(i)] = cook_ad.rolling(i).sum()
    cook_ad_csv["SMA_ratio" + str(i)] = np.array(cook_ad_csv["SMA"+str(i)]) / np.array(cook_hosp_csv_copy["hospitalized"])
#     plt.plot(cook_ad_csv["SMA_ratio" + str(i)])
# plt.show()

# cook_ad_csv.dropna(inplace=True)

# plt.plot(cook_ad_csv["SMA_ratio" + str(i)])

cook_ad_csv.to_csv("hosp_ad_MAsum_cook.csv")
# hosp_ad_csv["SMA12_0.2"] = cook_ad_csv["SMA12"] * 0.2

# fig, ax = plt.subplots()
# plt.xlabel("date")
# plt.ylabel("12-day hospital admission total * 0.2 / icu census")
# x = hosp_ad_csv['date']
# y = np.array(hosp_ad_csv["SMA12_0.2"]) / np.array(icu_csv_copy["hospitalized"])
# # print(y)
# plt.plot(x, y)

# plt.axhline(y=1)
# plt.legend()
# myFmt = mdates.DateFormatter('%y-%m')
# ax.xaxis.set_major_formatter(myFmt)
# plt.show()



fig, ax = plt.subplots()
plt.xlabel("date")
plt.ylabel("n-day hospital admission total / hospital census")

x = cook_ad_csv['date']
for i in range(7, 20):
    # y = np.array(cook_ad_csv["SMA"+str(i)]) / np.array(cook_hosp_csv_copy["hospitalized"])

    y = np.array(cook_ad_csv["SMA"+str(i)]) / np.array(cook_beds)
    # y = hosp_ad_csv["SMA_ratio" + str(i)]
    plt.plot(x[:60], y[:60],label = '%s-day total'%i)

plt.axhline(y=1)
plt.legend()
myFmt = mdates.DateFormatter('%y-%m')
ax.xaxis.set_major_formatter(myFmt)
plt.show()
plt.savefig("./cook_admission_census_ratio.png")
# plt.close()


# fig, ax = plt.subplots()
# plt.xlabel("date")
# plt.ylabel("icu census / hospital census")
# plt.plot(hosp_csv['date'], icu_csv["hospitalized"] / hosp_csv["hospitalized"])
# myFmt = mdates.DateFormatter('%y-%m')
# ax.xaxis.set_major_formatter(myFmt)
# plt.show()


