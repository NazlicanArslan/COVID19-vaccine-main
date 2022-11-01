###############################################################################

# ParamFittingTools.py
# This module contains Opt(imization) Tools, and includes functions
#   for generating realistic sample paths, enumerating candidate policies,
#   and optimization.
# This module is not used to run the SEIR model. This module contains
#   functions "on top" of the SEIR model.

# Guyi Chen 2022

###############################################################################

import numpy as np

from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from scipy.optimize import least_squares
import datetime as dt
import InputOutputTools
import copy
import itertools
import pandas as pd

def run_deterministic_path(city, vaccine_data):
    rep = SimReplication(city, vaccine_data, None, -1)
    return rep

def run_fit(city,
            vaccines,
            change_dates,
            x_bound,
            initial_guess,
            w_ih,
            w_icu,
            w_iyih, 
            w_d, 
            w_iyd,
            start_date,
            end_date):
   
    kwargs  = {'change_dates' : change_dates,
               'city' : city,
               'vaccine_data' : vaccines,
               't_start':city.cal.calendar.index(start_date),
               't_end':city.cal.calendar.index(end_date),\
                'w_ih': w_ih, 
                'w_icu': w_icu,
                'w_iyih': w_iyih,
                'w_d' : w_d,
                'w_iyd' : w_iyd,
               }
    ########## ########## ##########
    #Run least squares and choose a method
    res = least_squares_fit(initial_guess, x_bound, kwargs)
    SSE = res.cost

    ########## ########## ##########
    #Get variable value
    opt_tr_reduction = res.x
    if city.city == "austin":
        contact_reduction = np.array([
                opt_tr_reduction[1],
                opt_tr_reduction[2],
                opt_tr_reduction[3],
                ])
        
        cocoon = np.array([
                opt_tr_reduction[4],
                opt_tr_reduction[5],
                opt_tr_reduction[6],
                ])
        print('beta_0:', city.epi_rand.beta)   
        print('SSE:', SSE)   
        # print('alpha1', opt_tr_reduction[0])
        # print('alpha2', opt_tr_reduction[1])
        # print('alpha3', opt_tr_reduction[2])
        # print('alpha4', opt_tr_reduction[3])
        # print('rIH', opt_tr_reduction[0])
        # contact_reduction = np.array([  0.052257,
        #         0.787752,
        #         0.641986,
        #         0.827015,
        #         0.778334,
        #         0.752980,
        #         0.674321,
        #         0.801538,
        #         0.811144,
        #         0.6849,
        #         0.5551,
        #         0.6446,
        #         0.6869,
        #         0.7186,
        #         opt_tr_reduction[5],
        #         opt_tr_reduction[6],
        #         opt_tr_reduction[7]])
        
        # cocoon = np.array([0,
        #         0.787752,
        #         0.787752,
        #         0.827015,
        #         0.827015,
        #         0.787752,
        #         0.827015,
        #         0.801538,
        #         0.811144,
        #         0.6849,
        #         0.5551,
        #         0.6446,
        #         0.6869,
        #         0.7186,
        #         opt_tr_reduction[5],
        #         opt_tr_reduction[6],
        #         opt_tr_reduction[7]])
        
        # print('beta_0:', city.epi_rand.beta)   
        # print('SSE:', SSE)   
        # print('alpha1_omic=', opt_tr_reduction[0])
        # print('alpha2_omic=', opt_tr_reduction[1])
        # print('alpha3_omic=', opt_tr_reduction[2])
        # print('alpha4_omic=', opt_tr_reduction[3])
        # print('immune escape', opt_tr_reduction[4])

    elif city.city == "cook":
        contact_reduction = np.array([
                opt_tr_reduction[0],
                opt_tr_reduction[1],
                opt_tr_reduction[2],
                opt_tr_reduction[3],
               ])
        
        cocoon = np.array([
            # opt_tr_reduction[3],
             opt_tr_reduction[4],
                opt_tr_reduction[5],
                opt_tr_reduction[6],
                opt_tr_reduction[7],
                # opt_tr_reduction[8],
                # opt_tr_reduction[9],
                ])
        print('beta_0:', city.epi_rand.beta)   
        print('SSE:', SSE)   
        # print('alpha1', opt_tr_reduction[0])
        # print('alpha2', opt_tr_reduction[1])
        # print('alpha3', opt_tr_reduction[2])
        # print('alpha4', opt_tr_reduction[3])
        # print('rIH', opt_tr_reduction[0])

                                                                                
    betas = city.epi_rand.beta*(1 - (contact_reduction))
    end_date = []
    for idx in range(len(change_dates[1:])):
        end_date.append(str(change_dates[1:][idx] - dt.timedelta(days=1)))
    
    
    #breakpoint()
    # for the high risk groups uses cocoon instead of contact reduction
    table = pd.DataFrame({'start_date': change_dates[:-1], 'end_date': end_date, 'contact_reduction': contact_reduction, 'beta': betas, 'cocoon': cocoon})
    print(table)
    
    #Save optimized values to transmission_new.csv
    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([contact_reduction[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
    
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc, 'cocooning': cocoon_reduc}
    df_transmission = pd.DataFrame(data=d)

    return df_transmission
 

def save_output(transmission, instance):  
    file_path = instance.path_to_data / 'transmission_lsq_estimated_data.csv'
    transmission.to_csv(file_path, index = False)



def residual_error(x_variables, **kwargs):
    change_dates = kwargs['change_dates']
    city = kwargs['city']
    vaccine_data = kwargs['vaccine_data']
    
    t_start = kwargs['t_start']
    t_end = kwargs['t_end']
    print(t_end)
    w_ih = kwargs['w_ih']
    w_icu = kwargs['w_icu']
    w_iyih = kwargs['w_iyih']
    w_d = kwargs['w_d']
    w_iyd = kwargs['w_iyd']
    #############Change the transmission reduction and cocconing accordingly
    if city.city == "austin":
        beta = [x_variables[1],
                x_variables[2],
                x_variables[3],]
        
        cocoon = [x_variables[4],
                x_variables[5],
                x_variables[6],]
        # beta = [ 0.052257,
        #         0.787752,
        #         0.641986,
        #         0.827015,
        #         0.778334,
        #         0.752980,
        #         0.674321,
        #         0.801538,
        #         0.811144,
        #         0.6849,
        #         0.5551,
        #         0.6446,
        #         0.6869,
        #         0.7186,
        #         x_variables[5],
        #         x_variables[6],
        #         x_variables[7]]
        
        # cocoon = [0,
        #         0.787752,
        #         0.787752,
        #         0.827015,
        #         0.827015,
        #         0.787752,
        #         0.827015,
        #         0.801538,
        #         0.811144,
        #         0.6849,
        #         0.5551,
        #         0.6446,
        #         0.6869,
        #         0.7186,
        #         x_variables[5],
        #         x_variables[6],
        #         x_variables[7]]
    elif city.city == "cook":
        beta = [x_variables[0],
                x_variables[1],
                x_variables[2],
                x_variables[3],]
        
        cocoon = [x_variables[4],
                x_variables[5],
                x_variables[6],
                x_variables[7],]
    tr_reduc = []
    date_list = []
    cocoon_reduc = []
    for idx in range(len(change_dates[:-1])):
        tr_reduc.extend([beta[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
        date_list.extend([str(change_dates[idx] + dt.timedelta(days=x)) for x in range((change_dates[idx + 1] - change_dates[idx]).days)])
        cocoon_reduc.extend([cocoon[idx]] * (change_dates[idx + 1] - change_dates[idx]).days)
    
    nodaysrem = (city.end_date.date() - change_dates[-1]).days 
    date_list.extend([str(change_dates[-1] + dt.timedelta(days=x)) for x in range(nodaysrem)])
    tr_reduc.extend([beta[0]] * nodaysrem)
    cocoon_reduc.extend([cocoon[0]] * nodaysrem)
        
    d = {'date': pd.to_datetime(date_list), 'transmission_reduction': tr_reduc}
    df_transmission = pd.DataFrame(data=d)
    transmission_reduction = [(d, tr) for (d, tr) in zip(df_transmission['date'], df_transmission['transmission_reduction'])]
    city.cal.load_fixed_transmission_reduction(transmission_reduction)
    
    d = {'date': pd.to_datetime(date_list), 'cocooning': cocoon_reduc}
    df_cocooning = pd.DataFrame(data=d)
    cocooning = [(d, c) for (d, c) in zip(df_cocooning['date'], df_cocooning['cocooning'])]
    city.cal.load_fixed_cocooning(cocooning)
    #############
    #logger.info(f'beta: {str(beta)}')

    # Define additional variables included in the fit
    if city.city == "cook":
        # city.epi.alpha1 = x_variables[0]
        # city.epi.alpha2 = x_variables[1]
        # city.epi.alpha3 = x_variables[2]
        # city.epi.alpha4 = x_variables[3]
        # city.base_epi.rIH = x_variables[0]
        pass
        # city.epi.immune_escape_rate = x_variables[4]
    elif city.city == "austin":
        # city.base_epi.alpha1_omic = x_variables[0]
        # city.base_epi.alpha2_omic = x_variables[1]
        # city.base_epi.alpha3_omic = x_variables[2]
        # city.base_epi.alpha4_omic = x_variables[3]
        # city.base_epi.immune_escape_rate = x_variables[4]
        city.base_epi.rIH = x_variables[0]
    print('new value: ', x_variables)
  
                    
    rep = run_deterministic_path(city, vaccine_data)
    rep.simulate_time_period(t_end)

    city.epi_rand = rep.epi_rand

    real_hosp_total = city.real_hosp[t_start:t_end + 1]
    real_hosp_icu = city.real_hosp_icu[t_start:t_end + 1]
    real_hosp_ad = city.real_hosp_ad[t_start:t_end + 1]
    real_death_total = city.real_death_total[t_start:t_end + 1]
    real_death_hosp = city.real_death_hosp[t_start:t_end + 1]

    hosp_benchmark = None
    real_hosp = [a_i - b_i for a_i, b_i in zip(real_hosp_total, real_hosp_icu)] 
    print(len(real_hosp))

    hosp_benchmark = [rep.IH_history[t].sum() for t in range(t_start, t_end + 1)]
    residual_error_IH = [a_i - b_i for a_i, b_i in zip(real_hosp, hosp_benchmark)]
    print(len(residual_error_IH))
    icu_benchmark = [rep.ICU_history[t].sum() for t in range(t_start, t_end + 1)]
    residual_error_ICU = [w_ih * (a_i - b_i) for a_i, b_i in zip(real_hosp_icu, icu_benchmark)]
    residual_error_ICU = [element * w_icu for element in residual_error_ICU]
    residual_error_IH.extend(residual_error_ICU)


    daily_ad_benchmark = [rep.ToIHT_history[t].sum() for t in range(t_start, t_end)] 
    print('hospital admission: ', sum(daily_ad_benchmark))
    residual_error_IYIH = [a_i - b_i for a_i, b_i in zip(real_hosp_ad, daily_ad_benchmark)]
    residual_error_IYIH = [element * w_iyih for element in residual_error_IYIH]
    residual_error_IH.extend(residual_error_IYIH)
    

    daily_death_benchmark = [rep.D_history[t+1].sum() - rep.D_history[t].sum() for t in range(t_start, t_end)] 
    daily_death_benchmark.insert(0, 0)
    daily_death_benchmark = [rep.ToICUD_history[t].sum() for t in range(t_start, t_end)] 
    daily_death_benchmark.insert(0, 0)
    residual_error_death = [a_i - b_i for a_i, b_i in zip(real_death_hosp, daily_death_benchmark)]
    residual_error_death = [element * w_d for element in residual_error_death]
    residual_error_IH.extend(residual_error_death)     
    
    real_toIYD = [a_i - b_i for a_i, b_i in zip(real_death_total, real_death_hosp)]
    daily_toIYD_benchmark = [rep.ToIYD_history[t].sum() for t in range(t_start, t_end)] 
    daily_death_benchmark.insert(0, 0)
    residual_error_death = [a_i - b_i for a_i, b_i in zip(real_toIYD, daily_toIYD_benchmark)]
    residual_error_death = [element * w_iyd for element in residual_error_death]
    residual_error_IH.extend(residual_error_death)

    #breakpoint()
    # print('residual error:', residual_error_IH)
    return residual_error_IH 



def least_squares_fit(initial_guess, x_bound, kwargs):
    # Function that runs the least squares fit
    result = least_squares(residual_error,
                           initial_guess,
                           bounds = x_bound,
                           method='trf', verbose=2,
                           kwargs = kwargs)
    return result 
