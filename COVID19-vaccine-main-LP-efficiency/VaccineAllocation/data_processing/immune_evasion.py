from math import log
import csv
import pandas as pd
from copy import deepcopy
from datetime import datetime as dt

"""
This is separate from the SEIR simulation module. Create a csv file for the different immune evasion rates.

Nazlican Arslan 2022
"""


def calculate_immune_evasion(half_life, half_life_base, start_date, peak_date):
    """
    We assume the immunity evade with exponential rate.
    Calculate the changing immune evasion rate according to variant prevalence.
    Assume that the immune evasion follows a piecewise linear shape.
    It increases as the prevalence of variant increases and peak when the prevalence is around 80%
    and starts to decrease.

    :param half_life: half-life of the vaccine or natural infection induced protection.
    :param half_life_base: half-life of base level of immune evasion before the variant
    :param start_date: the date the immune evasion starts to increase
    :param peak_date: the date the immune evasion reaches the maximum level.
    """

    # calculate the rate of exponential immune evasion according to half-life (median) value:
    immune_max = log(2) / (half_life * 30)
    immune_base = log(2) / (half_life_base * 30) if half_life_base != 0 else 0
    days = (peak_date - start_date).days
    immune_evasion = []
    for t in range(days):
        immune_evasion.append(t * (immune_max - immune_base) / days + immune_base)
    immune_evasion_r = deepcopy(immune_evasion)
    immune_evasion_r.reverse()
    df = pd.DataFrame(immune_evasion + immune_evasion_r, columns=['immune_evasion'])
    df.to_csv('immune_evasion.csv')


if __name__ == "__main__":
    half_life = 2
    half_life_base = 0
    start_date = dt(2021, 12, 5)
    peak_date = dt(2021, 12, 29)
    calculate_immune_evasion(half_life, half_life_base, start_date, peak_date)
