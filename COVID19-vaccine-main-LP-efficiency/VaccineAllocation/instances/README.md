# Input files Overview

## setup_data_Final.json
This file has information about population size, epidemiological parameters etc. for a city instance. 
- "start_date" and "end_date"(not used anymore): start and end date of the simulation
- "hosp_beds": hospital bed capacity.
- "icu": ICU bed capacity.
- "epi_params":To learn more about epi_params please check Haoxiang et al.
- "school_closure": school closure calendar of Austin. The model distinguishes if the schools are open or closed. 
Don't forget to add the school closure schedule if you run the simulation for later dates.


## transmission.csv

Fixed transmission reduction and cocooning values. The file has additional information about changing hospital dynamics.
Care of COVID-19 patient improved over time. Refer to Haoxiang et al. for more information about parameters.

## tiers5_opt_Final.json or tiers_CDC.json
These files contain information about staged-alert system tiers. tiers5_opt_Final.json contain the tier information 
about the Austin's staged-alert system. tiers_CDC.json contains information about the CDC community thresholds. 
The keywords can be different depending on the staged-alert system that has been used.

## variant.json

This file includes the information about changes in model parameters when a new variant emerges.

Key words:
- "immune_evasion": In the current Austin model we assume there is immune evasion only when a new variant emerges. 
This key word includes all the information related to the immune evasion due to emergence of new variant.
  
  1. "half-life": we assume immunity wane at an exponential rate with a certain half-life. 
  2. "start_date": we use a triangular shape to adjust immune evasion (as the new variant start circulating the immune 
  evasion will increase. After a while as more people got infected with the new variant the immune evasion rate will 
  decrease.)
  3. "peak_date": the date where immune evasion peak.
  
## variant_prevalence.csv
 
We assume that the prevalence of a new variant follows an S-shaped logistic curve. This file contains the points from 
that logistic curve for different variants.

## vaccines.json

This files contain information about the vaccine efficacy etc.
- "beta_reduct": vaccine efficacy against infection under different vaccine levels.
- "tau_reduct": vaccine efficacy against symptomatic infection under different vaccine levels.
- "pi_reduct": vaccine efficacy against hospitalization under different vaccine levels.
- "second_dose_time": number of days between first and second dose.
- "effect_time": vaccine start protecting after effect_time amount of days.

## vaccine_allocation_fixed.csv
Fixed vaccine allocation percentages for each age risk group for first dose vaccination. 
(The data is from Texas DSHS.)

## booster_allocation_fixed.csv

Fixed booster allocation data, we assume the booster schedule is similar to first dose schedule but with uptake rate. 

## calendar.csv
