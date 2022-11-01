import urllib.request
import json
import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

filepath = "./instances/cook/"
def scrape_admission_total_beds_regions(input_regions=[10,11], outfile_name="cook"):
    regions = input_regions

    json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(11)+'&daysIncluded=0')
    data = json.loads(json_url.read())

    total_admission_df = pd.DataFrame(0, index=np.arange(len(data)), columns=['date', 'hospitalized'])
    total_beds_df = pd.DataFrame(0, index=np.arange(len(data)), columns=['date', 'hospitalized'])

    for region in regions:
        # read this file - this is where they put the data.  It's a json file, so need to re-format into a dataframe...
        json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetResurgenceData?regionID='+str(region)+'&daysIncluded=0')
        data = json.loads(json_url.read())
        mydf = pd.json_normalize(data)
        admission_df = mydf[['ReportDate','CLIAdmissions_RollingAvg']]
        bed_df = mydf[['ReportDate','COVIDHospitalBedsInUse']]
        admission_df = admission_df.rename(columns = {"ReportDate": "date", "CLIAdmissions_RollingAvg": "hospitalized"})
        bed_df = bed_df.rename(columns = {"ReportDate": "date", "COVIDHospitalBedsInUse": "hospitalized"})
        
        admission_df["date"] = pd.to_datetime(admission_df.date, format='%Y-%m-%dT%H:%M:%S')
        admission_df["date"] = admission_df["date"].dt.strftime("%m/%d/%y")
        # print(data[0])

        bed_df["date"] = pd.to_datetime(bed_df.date, format='%Y-%m-%dT%H:%M:%S')
        bed_df["date"] = bed_df["date"].dt.strftime("%m/%d/%y")

        total_admission_df['hospitalized'] += admission_df['hospitalized']
        total_beds_df['hospitalized'] += bed_df['hospitalized']
        total_admission_df['date'] = admission_df['date']
        total_beds_df['date'] = bed_df['date']

    total_admission_df.to_csv(os.path.join(filepath, outfile_name + '_hosp_ad_region_sum.csv'))
    total_beds_df.to_csv(os.path.join(filepath, outfile_name + '_total_beds_region_sum.csv'))


def scrape_cook_death_date():
    cook_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=cook')
    chicago_death_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetCountyTestResultsTimeSeries?CountyName=chicago')
    cook_death = json.loads(cook_death_url.read())
    cook_death_df = pd.json_normalize(cook_death)
    cook_death_df = cook_death_df[['ReportDate','Deaths']]
    cook_death_df = cook_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

    cook_death_df["date"] = pd.to_datetime(cook_death_df.date, format='%Y-%m-%dT%H:%M:%S')
    cook_death_df["date"] = cook_death_df["date"].dt.strftime("%m/%d/%y")


    chicago_death = json.loads(chicago_death_url.read())
    chicago_death_df = pd.json_normalize(chicago_death)
    chicago_death_df = chicago_death_df[['ReportDate','Deaths']]
    chicago_death_df = chicago_death_df.rename(columns = {"ReportDate": "date", "Deaths": "hospitalized"})

    chicago_death_df["date"] = pd.to_datetime(chicago_death_df.date, format='%Y-%m-%dT%H:%M:%S')
    chicago_death_df["date"] = chicago_death_df["date"].dt.strftime("%m/%d/%y")


    total_death_df = pd.DataFrame(0, index=np.arange(len(chicago_death)), columns=['date', 'hospitalized'])
    total_death_df["hospitalized"] = chicago_death_df["hospitalized"] + cook_death_df["hospitalized"]
    total_death_df["date"] = chicago_death_df["date"]
    total_death_df.to_csv(os.path.join(filepath,'cook_deaths.csv'))


def scrape_state_hosp_data():
    json_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalUtilizationResults')
    data = json.loads(json_url.read())
    mydf = pd.json_normalize(data)

    print(mydf.info())

    icu_df = mydf[["ReportDate", "ICUInUseBedsCOVID"]]
    icu_df = icu_df.rename(columns = {"ReportDate": "date", "ICUInUseBedsCOVID": "hospitalized"})

    icu_df["hospitalized"] = icu_df["hospitalized"]
    icu_df["date"] = pd.to_datetime(icu_df.date, format='%Y-%m-%dT%H:%M:%S')
    icu_df["date"] = icu_df["date"].dt.strftime("%m/%d/%y")
    icu_df.to_csv(os.path.join(filepath,'IL_hosp_icu.csv'))
    
    generalward_df = mydf[["ReportDate", "TotalInUseBedsCOVID"]]
    generalward_df = generalward_df.rename(columns = {"ReportDate": "date", "TotalInUseBedsCOVID": "hospitalized"})

    generalward_df["hospitalized"] = generalward_df["hospitalized"]
    generalward_df["date"] = pd.to_datetime(generalward_df.date, format='%Y-%m-%dT%H:%M:%S')
    generalward_df["date"] = generalward_df["date"].dt.strftime("%m/%d/%y")
    generalward_df.to_csv(os.path.join(filepath,'IL_hosp.csv'))

def get_current_hosp_utilization_region():
    hospital_url = urllib.request.urlopen('https://idph.illinois.gov/DPHPublicInformation/api/COVIDExport/GetHospitalizationResultsRegion')
    hosp_data = json.loads(hospital_url.read())

def calculate_cook_IL_ratio():
    cook_bed_data = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", parse_dates=["date"], index_col=0)
    IL_bed_data = pd.read_csv("./instances/cook/IL_total_beds_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    cook_bed_data["date"] = cook_bed_data["date"].dt.strftime("%m/%d/%y")

    cook_ad_data = pd.read_csv("./instances/cook/cook_hosp_ad_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    IL_ad_data = pd.read_csv("./instances/cook/IL_hosp_ad_region_sum.csv", parse_dates=["date"], index_col=0).reset_index()
    
    ratio_bed_cook_IL = cook_bed_data["hospitalized"] / IL_bed_data["hospitalized"]
    ratio_ad_cook_IL = cook_ad_data["hospitalized"] / IL_ad_data["hospitalized"]
    print(ratio_bed_cook_IL)
    print(ratio_ad_cook_IL)

    # state data 
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    state_icu_data = pd.read_csv("./instances/cook/IL_hosp_icu.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    IL_total_beds = state_bed_data.query("date >= @ dt.datetime(2020, 6, 13) & date <= @ dt.datetime(2022, 4, 7)").reset_index()
    IL_icu_beds = state_icu_data.query("date >= @ dt.datetime(2020, 6, 13) & date <= @ dt.datetime(2022, 4, 7)").reset_index()
    first_IL_beds = state_bed_data.query("date < @ dt.datetime(2020, 6, 13)")
    first_icu_beds = state_icu_data.query("date < @ dt.datetime(2020, 6, 13)")

    total_beds_copy = cook_bed_data
    first_icu_beds["date"] = first_icu_beds["date"].dt.strftime("%m/%d/%y")
    first_IL_beds["date"] = first_IL_beds["date"].dt.strftime("%m/%d/%y")
    estimated_icu_cook = IL_icu_beds["hospitalized"] * ratio_bed_cook_IL
    icu_copy = pd.DataFrame({"date":cook_bed_data["date"], "hospitalized": estimated_icu_cook})
   
    estimated_hosp_cook = IL_total_beds["hospitalized"] * ratio_bed_cook_IL
    total_beds_copy["hospitalized"] =  estimated_hosp_cook
    first_IL_beds["hospitalized"] = first_IL_beds["hospitalized"] * 0.7
    first_icu_beds["hospitalized"] = first_icu_beds["hospitalized"] * 0.7

    estimated_icu_cook_df = pd.concat([first_icu_beds, icu_copy]).reset_index(drop=True)
    estimated_beds_cook_df = pd.concat([first_IL_beds, total_beds_copy]).reset_index(drop=True)
    estimated_icu_cook_df.to_csv(os.path.join(filepath,'cook_icu_estimated.csv'))
    estimated_beds_cook_df.to_csv(os.path.join(filepath,'cook_hosp_estimated.csv'))

def icu_hosp_ratio():
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", index_col=0).reset_index(drop=True)
    state_icu_data = pd.read_csv("./instances/cook/IL_hosp_icu.csv", index_col=0).reset_index(drop=True)
    icu_hosp_ratio = state_icu_data["hospitalized"] / state_bed_data["hospitalized"]
    cook_bed_data = pd.read_csv("./instances/cook/cook_hosp_region_sum_estimated.csv")
    cook_icu_data = cook_bed_data["hospitalized"] * icu_hosp_ratio
    cook_icu_data_df = pd.DataFrame({"date": state_bed_data["date"], "hospitalized": cook_icu_data})
    cook_icu_data_df.to_csv(os.path.join(filepath,'cook_icu_region_sum_estimated.csv'))

def hosp_region_sum_estimation():
    cook_bed_data = pd.read_csv("./instances/cook/cook_total_beds_region_sum.csv", index_col=0)
    state_bed_data = pd.read_csv("./instances/cook/IL_hosp.csv", parse_dates=["date"], index_col=0).reset_index(drop=True)
    first_IL_beds = state_bed_data.query("date < @ dt.datetime(2020, 6, 13)")
    first_IL_beds["date"] = first_IL_beds["date"].dt.strftime("%m/%d/%y")
    first_IL_beds["hospitalized"] = first_IL_beds["hospitalized"] * 1.05 * 0.7
    estimated_beds_cook_df = pd.concat([first_IL_beds, cook_bed_data]).reset_index(drop=True)
    estimated_beds_cook_df.to_csv(os.path.join(filepath,'cook_hosp_region_sum_estimated.csv'))



def add_shifted_scaled_austin_data_to_cook():
    austin_files = ["austin_real_hosp_no_may.csv",
                "austin_real_icu_no_may.csv",
                "austin_hosp_ad_no_may.csv",]
    cook_files = ["cook_hosp_region_sum_estimated.csv",
                "cook_icu_region_sum_estimated.csv",
                "cook_hosp_ad_region_sum_18_syn.csv",]

    outfile_names = ["hosp", "icu", "admission",]

    for (austin_file, cook_file, outfile_name) in zip(austin_files, cook_files, outfile_names):
        df_austin_hosp = pd.read_csv(
                "./instances/austin/" + austin_file,
                parse_dates=['date'],
                date_parser=pd.to_datetime,
            )
        df_cook_hosp = pd.read_csv(
                    "./instances/cook/" + cook_file,
                    parse_dates=['date'],
                    date_parser=pd.to_datetime,
                )
        austin_peaks, _ = find_peaks(df_austin_hosp["hospitalized"], distance=60)
        second_austin_peak_idx = austin_peaks[1]
        austin_date = df_austin_hosp["date"]
        austin_hosp_data = df_austin_hosp["hospitalized"]
        austin_peak_date = austin_date[second_austin_peak_idx]
        austin_hosp_peak_val = austin_hosp_data[second_austin_peak_idx]
        cook_peaks, _ = find_peaks(df_cook_hosp["hospitalized"],distance=60)
        first_cook_peak_idx = cook_peaks[0]
        cook_date = df_cook_hosp["date"]
        cook_peak_date = cook_date[first_cook_peak_idx]
        cook_hosp_data = df_cook_hosp["hospitalized"]
        cook_hosp_peak_val = cook_hosp_data[first_cook_peak_idx]
        time_delta = (cook_peak_date - austin_peak_date).days
        # print(time_delta)
        scale_factor = cook_hosp_peak_val / austin_hosp_peak_val
        print(outfile_name)
        print(scale_factor)

        first_cook_date = cook_date[0]
        if "admission" in outfile_name:
            shifted_first_date_austin = austin_date[0] + dt.timedelta(time_delta)
            shifted_austin_dates = austin_date + np.timedelta64(time_delta, 'D')
            # print(shifted_austin_dates[:-first_time_delta])
            first_time_delta = (shifted_first_date_austin - first_cook_date).days
            shifted_austin_dates = shifted_austin_dates[:-first_time_delta]
            austin_shifted_scaled_data = austin_hosp_data[:-first_time_delta] * scale_factor
        else:
            shifted_first_date_austin = austin_date[0] + dt.timedelta(time_delta)
            shifted_austin_dates = austin_date + np.timedelta64(time_delta, 'D')
            first_time_delta = (shifted_first_date_austin - first_cook_date).days
            # print(shifted_austin_dates[:-first_time_delta])
            shifted_austin_dates = shifted_austin_dates[:-first_time_delta]
            austin_shifted_scaled_data = austin_hosp_data[:-first_time_delta] * scale_factor
        print(shifted_first_date_austin)
        print(first_cook_date)
        # print(first_time_delta)
       
        dates = pd.concat([shifted_austin_dates, cook_date]).dt.strftime("%m/%d/%y")
        hosp_data = pd.concat([austin_shifted_scaled_data, cook_hosp_data])
        df = pd.DataFrame({'date': dates,
                    'hospitalized': hosp_data}).reset_index(drop=True)
        # print(df[20:70])
        df.to_csv("./instances/cook/" + "smoothed_region_sum_estimated_no_may_" + outfile_name + ".csv")

# scrape_admission_total_beds_regions()
# calculate_cook_IL_ratio()
# scrape_state_hosp_data()
add_shifted_scaled_austin_data_to_cook()
# hosp_region_sum_estimation()
# icu_hosp_ratio()