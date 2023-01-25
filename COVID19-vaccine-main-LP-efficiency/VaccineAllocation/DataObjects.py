###############################################################################

# DataObjects.py

###############################################################################

import json
from math import log

import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from itertools import product

base_path = Path(__file__).parent

datetime_formater = "%Y-%m-%d %H:%M:%S"

WEEKDAY = 1
WEEKEND = 2
HOLIDAY = 3
LONG_HOLIDAY = 4


class SimCalendar:
    """
    A simulation calendar to map time steps to days. This class helps
    to determine whether a time step t is a weekday or a weekend, as well
    as school calendars.
    """

    def __init__(self, start_date, sim_length,
                 school_closure_blocks,
                 ts_transmission_reduction, ts_cocooning,
                 holidays, long_holidays,
                 real_IH_history):
        self.start = start_date
        self.calendar = [self.start + dt.timedelta(days=t) for t in range(sim_length)]
        self.calendar_ix = {d: d_ix for (d_ix, d) in enumerate(self.calendar)}
        self._is_weekday = [d.weekday() not in [5, 6] for d in self.calendar]
        self._day_type = [WEEKDAY if iw else WEEKEND for iw in self._is_weekday]
        self.lockdown = None
        self.schools_closed = None
        self.fixed_transmission_reduction = None
        self.fixed_cocooning = None
        self.real_hosp = real_IH_history

        self.load_school_closure(school_closure_blocks)
        self.load_fixed_transmission_reduction(ts_transmission_reduction)
        self.load_fixed_cocooning(ts_cocooning)
        self.load_holidays(holidays, long_holidays)

    def load_school_closure(self, school_closure_blocks):
        """
        Load fixed decisions on school closures and saves
        it on attribute schools_closed
        Args:
            school_closure_blocks (list of tuples): a list with blocks in which schools are closed
            (e.g. [(datetime.date(2020,3,24),datetime.date(2020,8,28))])
        """
        self.schools_closed = []
        for d in self.calendar:
            closedDay = False
            for blNo in range(len(school_closure_blocks)):
                if (
                        school_closure_blocks[blNo][0] <= d <= school_closure_blocks[blNo][1]
                ):
                    closedDay = True
            self.schools_closed.append(closedDay)

    def load_fixed_transmission_reduction(self, ts_transmission_reduction):
        """
        Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
        If a value is not given, the transmission reduction is None.
        Args:
            ts_transmission_reduction (list of tuple): a list with the time series of
                transmission reduction (datetime, float).
        """
        self.fixed_transmission_reduction = [None for d in self.calendar]
        for (d, tr) in ts_transmission_reduction:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_transmission_reduction[d_ix] = tr

    def load_fixed_cocooning(self, ts_cocooning):
        """
        Load fixed decisions on transmission reduction and saves it on attribute fixed_transmission_reduction.
        If a value is not given, the transmission reduction is None.
        Args:
            ts_cocooning (list of tuple): a list with the time series of
                transmission reduction (datetime, float).
        """
        self.fixed_cocooning = [None for d in self.calendar]
        for (d, tr) in ts_cocooning:
            if d in self.calendar_ix:
                d_ix = self.calendar_ix[d]
                self.fixed_cocooning[d_ix] = tr

    def load_holidays(self, holidays=[], long_holidays=[]):
        """
        Change the day_type for holidays
        """
        for hd in holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = HOLIDAY

        for hd in long_holidays:
            dt_hd = dt.datetime(hd.year, hd.month, hd.day)
            if dt_hd in self.calendar:
                self._day_type[self.calendar_ix[dt_hd]] = LONG_HOLIDAY


class City:
    def __init__(
            self,
            city,
            config_filename,
            calendar_filename,
            setup_filename,
            variant_filename,
            transmission_filename,
            hospitalization_filename,
            hosp_icu_filename,
            hosp_admission_filename,
            death_from_hosp_filename,
            death_from_home_filename,
            variant_prevalence_filename,
    ):
        self.city = city
        self.path_to_data = base_path / "instances" / f"{city}"
        # Extra folder to store the json files for inputting outputting simulation data:
        self.path_to_input_output = base_path / "input_output_folder" / f"{city}"

        with open(str(self.path_to_data / config_filename), "r") as input_file:
            self.config = json.load(input_file)

        self.epi_rand = None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  Load data from setup_filename
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with open(str(self.path_to_data / setup_filename), "r") as input_file:
            data = json.load(input_file)
            assert self.city == data["city"], "Data file does not match city."

            # LP note: smooth this later -- not all of these attributes
            #   are actually used and some of them are redundant
            for (k, v) in data.items():
                setattr(self, k, v)


            # Load demographics information
            self.N = np.array(data["population"])
            self.I0 = np.array(data["IY_ini"])

            # Load simulation dates
            self.start_date = dt.datetime.strptime(
                data["start_date"], datetime_formater
            )

            self.end_date = dt.datetime.strptime(
                data["end_date"], datetime_formater)

            # the school_closure_period attribute is a list (of lists) indicating
            #   historical periods of school closure
            # each element of the list is a 2-element list,
            #   where the 1st element corresponds to the start date of school closure
            #   and the 2nd element corresponds to the end date of school closure
            self.school_closure_period = []
            for blSc in range(len(data["school_closure"])):
                self.school_closure_period.append(
                    [
                        dt.datetime.strptime(
                            data["school_closure"][blSc][0], datetime_formater
                        ),
                        dt.datetime.strptime(
                            data["school_closure"][blSc][1], datetime_formater
                        ),
                    ]
                )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set up base_epi
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.base_epi = EpiSetup(data["epi_params"], self.end_date)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load hospitalization-related data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hosp_related_data_filenames = (hospitalization_filename,
                                       hosp_icu_filename,
                                       hosp_admission_filename,
                                       death_from_hosp_filename,
                                       death_from_home_filename)

        real_history_hosp_related_data_vars = ("real_IH_history",
                                               "real_ICU_history",
                                               "real_ToIHT_history",
                                               "real_ToICUD_history",
                                               "real_ToIYD_history")

        for i in range(len(hosp_related_data_filenames)):
            filename = hosp_related_data_filenames[i]
            var = real_history_hosp_related_data_vars[i]
            if filename is not None:
                setattr(self, var, self.read_hosp_related_data(filename))
            else:
                setattr(self, var, None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load prevalence data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read the combined variant files instead of a separate file for each new variant:
        df_variant = pd.read_csv(
                str(self.path_to_data / variant_prevalence_filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
        )
        with open(self.path_to_data / variant_filename, "r") as input_file:
            variant_data = json.load(input_file)
        self.variant_pool = VariantPool(variant_data, df_variant)
        self.variant_start = df_variant["date"][0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define dimension variables & others
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Number of age and risk groups
        self.A = len(self.N)
        self.L = len(self.N[0])

        # Maximum simulation length
        self.T = 1 + (self.end_date - self.start_date).days

        self.otherInfo = {}

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load data from calendar_filename
        # Build simulation calendar
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        cal_df = pd.read_csv(
            str(self.path_to_data / calendar_filename),
            parse_dates=["Date"],
            date_parser=pd.to_datetime,
        )
        self.weekday_holidays = tuple(cal_df["Date"][cal_df["Calendar"] == 3])
        self.weekday_longholidays = tuple(cal_df["Date"][cal_df["Calendar"] == 4])

        self.cal = self.build_calendar(transmission_filename)

    def read_hosp_related_data(self, hosp_filename):
        '''
        Helper function to read a hospitalization data file
            and return an array with hospitalization counts.
        '''

        df_hosp = pd.read_csv(
            str(self.path_to_data / hosp_filename),
            parse_dates=["date"],
            date_parser=pd.to_datetime,
        )

        df_hosp = df_hosp[df_hosp["date"] <= self.end_date]

        # if hospitalization data starts before self.start_date
        if df_hosp["date"][0] <= self.start_date:
            df_hosp = df_hosp[df_hosp["date"] >= self.start_date]
            df_hosp = list(df_hosp["hospitalized"])
        else:
            df_hosp = [0] * (df_hosp["date"][0] - self.start_date).days + list(
                df_hosp["hospitalized"]
            )
        return df_hosp

    def build_calendar(self, transmission_filename):
        """
        Compute couple parameters (i.e., parameters that depend on the input)
        and build the simulation calendar.
        """
        try:
            df_transmission = pd.read_csv(
                str(self.path_to_data / transmission_filename),
                parse_dates=["date"],
                date_parser=pd.to_datetime,
                float_precision="round_trip",
            )
            transmission_reduction = [
                (d, tr)
                for (d, tr) in zip(
                    df_transmission["date"], df_transmission["transmission_reduction"]
                )
            ]

            try:
                cocooning = [
                    (d, co)
                    for (d, co) in zip(
                        df_transmission["date"], df_transmission["cocooning"]
                    )
                ]
            except:
                cocooning = [(d, 0.0) for d in df_transmission["date"]]

        except FileNotFoundError:
            # Initialize empty if no file available
            transmission_reduction = []

        cal = SimCalendar(self.start_date, self.T,
                          self.school_closure_period,
                          transmission_reduction, cocooning,
                          self.weekday_holidays, self.weekday_longholidays,
                          self.real_IH_history)

        for dfk in df_transmission.keys():
            if (
                    dfk != "date"
                    and dfk != "transmission_reduction"
                    and dfk != "cocooning"
            ):
                self.otherInfo[dfk] = {}
                for (d, dfv) in zip(df_transmission["date"], df_transmission[dfk]):
                    if d in cal.calendar_ix:
                        d_ix = cal.calendar_ix[d]
                        self.otherInfo[dfk][d_ix] = dfv
        return cal


class TierInfo:
    def __init__(self, city, tier_filename):
        self.path_to_data = base_path / "instances" / f"{city}"
        with open(str(self.path_to_data / tier_filename), "r") as tier_input:
            tier_data = json.load(tier_input)
            self.tier = tier_data["tiers"]


class Vaccine:
    """
    Vaccine class to define epidemiological characteristics, supply and fixed allocation schedule of vaccine.
    """

    def __init__(
            self,
            instance,
            city,
            vaccine_filename,
            booster_filename,
            vaccine_allocation_filename):

        self.path_to_data = base_path / "instances" / f"{city}"

        with open(str(self.path_to_data / vaccine_filename), "r") as vaccine_input:
            vaccine_data = json.load(vaccine_input)

        vaccine_allocation_data = pd.read_csv(
            str(self.path_to_data / vaccine_allocation_filename),
            parse_dates=["vaccine_time"],
            date_parser=pd.to_datetime,
        )

        if booster_filename is not None:
            booster_allocation_data = pd.read_csv(
                str(self.path_to_data / booster_filename),
                parse_dates=["vaccine_time"],
                date_parser=pd.to_datetime,
            )
        else:
            booster_allocation_data = None

        self.effect_time = vaccine_data["effect_time"]
        self.second_dose_time = vaccine_data["second_dose_time"]
        self.beta_reduct = vaccine_data["beta_reduct"]
        self.tau_reduct = vaccine_data["tau_reduct"]
        self.pi_reduct = vaccine_data["pi_reduct"]
        self.instance = instance


        self.actual_vaccine_time = [
            time for time in vaccine_allocation_data["vaccine_time"]
        ]
        self.first_dose_time = [
            time + dt.timedelta(days=self.effect_time)
            for time in vaccine_allocation_data["vaccine_time"]
        ]
        self.second_dose_time = [
            time + dt.timedelta(days=self.second_dose_time + self.effect_time)
            for time in self.first_dose_time
        ]

        self.vaccine_proportion = [
            amount for amount in vaccine_allocation_data["vaccine_amount"]
        ]
        self.vaccine_start_time = np.where(
            np.array(instance.cal.calendar) == self.actual_vaccine_time[0]
        )[0]

        self.vaccine_allocation = self.define_supply(vaccine_allocation_data,
                                                     booster_allocation_data,
                                                     instance.N,
                                                     instance.A,
                                                     instance.L
                                                    )
        self.event_lookup_dict = self.build_event_lookup_dict()

    def build_event_lookup_dict(self):
        """
        Must be called after self.vaccine_allocation is updated using self.define_supply

        This method creates a mapping between date and "vaccine events" in historical data
            corresponding to that date -- so that we can look up whether a vaccine group event occurs,
            rather than iterating through all items in self.vaccine_allocation

        Creates event_lookup_dict, a dictionary of dictionaries, with the same keys as self.vaccine_allocation,
            where each key corresponds to a vaccine group ("v_first", "v_second", "v_booster", "v_wane")
        self.event_lookup_dict[vaccine_type] is a dictionary
            the same length as self.vaccine_allocation[vaccine_ID]
        Each key in event_lookup_dict[vaccine_type] is a datetime object and the corresponding value is the
            i (index) of self.vaccine_allocation[vaccine_type] such that
            self.vaccine_allocation[vaccine_type][i]["supply"]["time"] matches the datetime object
        """

        event_lookup_dict = {}
        for key in self.vaccine_allocation.keys():
            d = {}
            counter = 0
            for allocation_item in self.vaccine_allocation[key]:
                d[allocation_item["supply"]["time"]] = counter
                counter += 1
            event_lookup_dict[key] = d
        return event_lookup_dict

    def event_lookup(self, vaccine_type, date):
        """
        Must be called after self.build_event_lookup_dict builds the event lookup dictionary

        vaccine_type is one of the keys of self.vaccine_allocation ("v_first", "v_second", "v_booster")
        date is a datetime object

        Returns the index i such that self.vaccine_allocation[vaccine_type][i]["supply"]["time"] == date
        Otherwise, returns None
        """

        if date in self.event_lookup_dict[vaccine_type].keys():
            return self.event_lookup_dict[vaccine_type][date]

    def get_num_eligible(
            self, total_population, total_risk_gr, vaccine_group_name, v_in, v_out, date
    ):
        """
        :param total_population: integer, usually N parameter such as instance.N
        :param total_risk_gr: instance.A x instance.L
        :param vaccine_group_name: string of vaccine_group_name (see VaccineGroup)
             ("unvax", "first_dose", "second_dose", "waned")
        :param v_in: tuple with strings of vaccine_types going "in" to that vaccine group
        :param v_out: tuple with strings of vaccine_types going "out" of that vaccine group
        :param date: datetime object
        :return: list of number eligible people for vaccination at that date, where each element corresponds
        to age/risk group (list is length A * L).
                For instance, only those who received their first-dose three weeks ago are eligible to get
                their second dose vaccine.
        """

        N_in = np.zeros((total_risk_gr, 1))
        N_out = np.zeros((total_risk_gr, 1))

        # LP note to self: could combine v_in and v_out as one function since
        #   they essentially repeat themselves (one with N_in and one with N_out)
        for vaccine_type in v_in:
            event = self.event_lookup(vaccine_type, date)
            if event is not None:
                for i in range(event):
                    N_in += self.vaccine_allocation[vaccine_type][i][
                        "assignment"
                    ].reshape((total_risk_gr, 1))
            else:
                if date > self.vaccine_allocation[vaccine_type][0]["supply"]["time"]:
                    i = 0
                    event_date = self.vaccine_allocation[vaccine_type][i]["supply"][
                        "time"
                    ]
                    while event_date < date:
                        N_in += self.vaccine_allocation[vaccine_type][i][
                            "assignment"
                        ].reshape((total_risk_gr, 1))
                        if i + 1 == len(self.vaccine_allocation[vaccine_type]):
                            break
                        i += 1
                        event_date = self.vaccine_allocation[vaccine_type][i]["supply"][
                            "time"
                        ]

        for vaccine_type in v_out:
            event = self.event_lookup(vaccine_type, date)
            if event is not None:
                for i in range(event):
                    N_out += self.vaccine_allocation[vaccine_type][i][
                        "assignment"
                    ].reshape((total_risk_gr, 1))
            else:
                if date > self.vaccine_allocation[vaccine_type][0]["supply"]["time"]:
                    i = 0
                    event_date = self.vaccine_allocation[vaccine_type][i]["supply"][
                        "time"
                    ]
                    while event_date < date:
                        N_out += self.vaccine_allocation[vaccine_type][i][
                            "assignment"
                        ].reshape((total_risk_gr, 1))
                        if i + 1 == len(self.vaccine_allocation[vaccine_type]):
                            break
                        i += 1
                        event_date = self.vaccine_allocation[vaccine_type][i]["supply"][
                            "time"
                        ]

        if vaccine_group_name == "unvax":
            N_eligible = total_population.reshape((total_risk_gr, 1)) - N_out
        elif vaccine_group_name == "waned":
            # Waned compartment does not have incoming vaccine schedule but has outgoing scheduled vaccine. People
            # enter waned compartment with binomial draw. This calculation would return negative value
            return None
        else:
            N_eligible = N_in - N_out

        assert (N_eligible > -1e-2).all(), (
            f"fPop negative eligible individuals for vaccination in vaccine group {vaccine_group_name}"
            f"{N_eligible} at time {date}"
        )
        return N_eligible

    def define_supply(self, vaccine_allocation_data, booster_allocation_data, N, A, L):
        """
        Load vaccine supply and allocation data, and process them.
        Shift vaccine schedule for waiting vaccine to be effective,
            second dose and vaccine waning effect and also for booster dose.
        """

        # Each of the following are lists
        # Each element of the list is a dictionary with keys
        #   "assignment", "proportion", "within_proportion", "supply"
        v_first_allocation = []
        v_second_allocation = []
        v_booster_allocation = []

        # 10 of these age-risk groups (5 age groups, 2 risk groups)
        age_risk_columns = [
            column
            for column in vaccine_allocation_data.columns
            if "A" and "R" in column
        ]

        # LP note to self: can also combine the following because
        #   the logic is redundant for the different types of allocations

        # Fixed vaccine allocation:
        for i in range(len(vaccine_allocation_data["A1-R1"])):
            vac_assignment = np.array(
                vaccine_allocation_data[age_risk_columns].iloc[i]
            ).reshape((A, L))


            if np.sum(vac_assignment) > 0:
                pro_round = vac_assignment / np.sum(vac_assignment)
            else:
                pro_round = np.zeros((A, L))
            within_proportion = vac_assignment / N

            # First dose vaccine allocation:
            supply_first_dose = {
                "time": self.first_dose_time[i],
                "amount": self.vaccine_proportion[i],
                "type": "first_dose",
            }
            allocation_item = {
                "assignment": vac_assignment,
                "supply": supply_first_dose,
                "from": "unvax"
            }
            v_first_allocation.append(allocation_item)

            # Second dose vaccine allocation:
            if i < len(self.second_dose_time):
                supply_second_dose = {
                    "time": self.second_dose_time[i],
                    "amount": self.vaccine_proportion[i],
                    "type": "second_dose",
                }
                allocation_item = {
                    "assignment": vac_assignment,
                    "supply": supply_second_dose,
                    "from": "first_dose"
                }
                v_second_allocation.append(allocation_item)

        # Fixed booster vaccine allocation:
        if booster_allocation_data is not None:
            self.booster_time = [
                time for time in booster_allocation_data["vaccine_time"]
            ]
            self.booster_proportion = np.array(
                booster_allocation_data["vaccine_amount"]
            )
            for i in range(len(booster_allocation_data["A1-R1"])):
                vac_assignment = np.array(
                    booster_allocation_data[age_risk_columns].iloc[i]

                ).reshape((A, L))


                if np.sum(vac_assignment) > 0:
                    pro_round = vac_assignment / np.sum(vac_assignment)
                else:
                    pro_round = np.zeros((A, L))
                within_proportion = vac_assignment / N

                # Booster dose vaccine allocation:
                supply_booster_dose = {
                    "time": self.booster_time[i],
                    "amount": self.booster_proportion[i],
                    "type": "booster_dose"
                }
                allocation_item = {
                    "assignment": vac_assignment,
                    "proportion": pro_round,
                    "within_proportion": within_proportion,
                    "supply": supply_booster_dose,
                    "from": "waned"
                }
                v_booster_allocation.append(allocation_item)

        return {
            "v_first": v_first_allocation,
            "v_second": v_second_allocation,
            "v_booster": v_booster_allocation
        }


class EpiSetup:
    """
    A setup for the epidemiological parameters.
    Scenarios 6 corresponds to best guess parameters for UT group.
    """

    def __init__(self, params, end_date):

        self.load_file(params)

        try:
            self.qInt["testStart"] = dt.datetime.strptime(
                self.qInt["testStart"], datetime_formater
            )
        except:
            setattr(
                self,
                "qInt",
                {
                    "testStart": end_date,
                    "qRate": {"IY": 0, "IA": 0, "PY": 0, "PA": 0},
                    "randTest": 0,
                },
            )

        # Parameters that are randomly sampled for each replication
        self.random_params_dict = {}

    def load_file(self, params):
        for (k, v) in params.items():
            if isinstance(v, list):
                if v[0] == "rnd_inverse" or v[0] == "rnd":
                    setattr(self, k, ParamDistribution(*v))
                else:
                    setattr(self, k, np.array(v))
            else:
                setattr(self, k, v)

    def sample_random_params(self, rng):
        """
        Generates random parameters from a given random stream.
        Coupled parameters are updated as well.
        Args:
            rng (np.random.default_rng): a default_rng instance from numpy.
        """

        # rng = None  #rng
        tempRecord = {}
        for k in vars(self):
            v = getattr(self, k)
            # if the attribute is random variable, generate a deterministic version
            if isinstance(v, ParamDistribution):
                tempRecord[v.param_name] = v.sample(rng)
            elif isinstance(v, np.ndarray):
                listDistrn = True
                # if it is a list of random variable, generate a list of deterministic values
                vList = []
                outList = []
                outName = None
                for vItem in v:
                    try:
                        vValue = ParamDistribution(*vItem)
                        outList.append(vValue.sample(rng))
                        outName = vValue.param_name
                    except:
                        vValue = 0
                    vList.append(vValue)
                    listDistrn = listDistrn and isinstance(vValue, ParamDistribution)
                if listDistrn:
                    tempRecord[outName] = np.array(outList)

        self.random_params_dict = tempRecord

        for k in tempRecord.keys():
            setattr(self, k, tempRecord[k])

    def setup_base_params(self):

        # See Yang et al. (2021) and Arslan et al. (2021)

        # tau: proportion of exposed individuals who become symptomatic
        # mu_ICU: rate from ICU to death (for each age group)

        # IFR: infected fatality ratio (%)

        self.beta = self.beta0  # Unmitigated transmission rate
        self.YFR = self.IFR / self.tau  # symptomatic fatality ratio (%)
        self.pIH0 = self.pIH
        self.YHR0 = self.YHR  # % of symptomatic infections that go to hospital
        self.YHR_overall0 = self.YHR_overall

        # if gamma_IH and mu are lists, reshape them for right dimension
        if isinstance(self.gamma_IH, np.ndarray):
            self.gamma_IH = self.gamma_IH.reshape(self.gamma_IH.size, 1)
            self.gamma_IH0 = self.gamma_IH.copy()
        if isinstance(self.etaICU, np.ndarray):
            self.etaICU = self.etaICU.reshape(self.etaICU.size, 1)
            self.etaICU0 = self.etaICU.copy()
        if isinstance(self.gamma_ICU, np.ndarray):
            self.gamma_ICU = self.gamma_ICU.reshape(self.gamma_ICU.size, 1)
            self.gamma_ICU0 = self.gamma_ICU.copy()
        if isinstance(self.mu_ICU, np.ndarray):
            self.mu_ICU = self.mu_ICU.reshape(self.mu_ICU.size, 1)
            self.mu_ICU0 = self.mu_ICU.copy()

        self.update_YHR_params()
        self.update_nu_params()

        # Formerly updated under update_hosp_duration() function in original code
        # See Yang et al. (2021) pg. 9 -- add constant parameters (alphas)
        #   to better estimate durations in ICU and general ward.
        self.gamma_ICU = self.gamma_ICU0 * (1 + self.alpha_gamma_ICU)
        self.gamma_IH = self.gamma_IH0 * (1 - self.alpha_IH)
        self.mu_ICU = self.mu_ICU0 * (1 + self.alpha_mu_ICU)

    def variant_update_param(self, new_params):
        """
            Update parameters according to variant prevalence.
            Combined all variant of concerns: delta, omicron, and a new hypothetical variant.
        """
        for (k, v) in new_params.items():
            if k == "sigma_E":
                setattr(self, k, v)
            else:
                setattr(self, k, v * getattr(self, k))
        self.update_YHR_params()
        self.update_nu_params()
        self.gamma_ICU = self.gamma_ICU0 * (1 + self.alpha_gamma_ICU)
        self.gamma_IH = self.gamma_IH0 * (1 - self.alpha_IH)
        self.mu_ICU = self.mu_ICU0 * (1 + self.alpha_mu_ICU)

    def update_icu_params(self, rdrate):
        # update the ICU admission parameter HICUR and update nu
        self.HICUR = self.HICUR * rdrate
        self.nu = (
                self.gamma_IH
                * self.HICUR
                / (self.etaICU + (self.gamma_IH - self.etaICU) * self.HICUR)
        )
        self.pIH = 1 - (1 - self.pIH) * rdrate

    def update_icu_all(self, t, otherInfo):
        if "pIH" in otherInfo.keys():
            if t in otherInfo["pIH"].keys():
                self.pIH = otherInfo["pIH"][t]
            else:
                self.pIH = self.pIH0
        if "HICUR" in otherInfo.keys():
            if t in otherInfo["HICUR"].keys():
                self.HICUR = otherInfo["HICUR"][t]
            else:
                self.HICUR = self.HICUR0
        if "etaICU" in otherInfo.keys():
            if t in otherInfo["etaICU"].keys():
                self.etaICU = self.etaICU0.copy() / otherInfo["etaICU"][t]
            else:
                self.etaICU = self.etaICU0.copy()
        self.nu = (
                self.gamma_IH
                * self.HICUR
                / (self.etaICU + (self.gamma_IH - self.etaICU) * self.HICUR)
        )

    def update_YHR_params(self):
        # Arslan et al. (2021) pg. 7
        # omega_P: infectiousness of pre-symptomatic relative to symptomatic
        self.omega_P = np.array(
            [
                (
                        self.tau
                        * self.omega_IY
                        * (
                                self.YHR_overall[a] / self.Eta[a]
                                + (1 - self.YHR_overall[a]) / self.gamma_IY
                        )
                        + (1 - self.tau) * self.omega_IA / self.gamma_IA
                )
                / (self.tau * self.omega_IY + (1 - self.tau) * self.omega_IA)
                * self.rho_Y
                * self.pp
                / (1 - self.pp)
                for a in range(len(self.YHR_overall))
            ]
        )
        self.omega_PA = self.omega_IA * self.omega_P
        self.omega_PY = self.omega_IY * self.omega_P

        # pi is computed using risk based hosp rate
        self.pi = np.array(
            [
                self.YHR[a]
                * self.gamma_IY
                / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
                for a in range(len(self.YHR))
            ]
        )

        # symptomatic fatality ratio divided by symptomatic hospitalization rate
        self.HFR = self.YFR / self.YHR

    def update_nu_params(self):
        try:
            self.HICUR0 = self.HICUR
            self.nu = (
                    self.gamma_IH
                    * self.HICUR
                    / (self.etaICU + (self.gamma_IH - self.etaICU) * self.HICUR)
            )
            self.nu_ICU = (
                    self.gamma_ICU
                    * self.ICUFR
                    / (self.mu_ICU + (self.gamma_ICU - self.mu_ICU) * self.ICUFR)
            )
        except:
            self.nu = (
                    self.gamma_IH
                    * self.HFR
                    / (self.etaICU + (self.gamma_IH - self.etaICU) * self.HFR)
            )

    def effective_phi(self, school, cocooning, social_distance, demographics, day_type):
        """
        school (int): yes (1) / no (0) schools are closed
        cocooning (float): percentage of transmission reduction [0,1]
        social_distance (int): percentage of social distance (0,1)
        demographics (ndarray): demographics by age and risk group
        day_type (int): 1 Weekday, 2 Weekend, 3 Holiday, 4 Long Holiday
        """

        A = len(demographics)  # number of age groups
        L = len(demographics[0])  # number of risk groups
        d = demographics  # A x L demographic data
        phi_all_extended = np.zeros((A, L, A, L))
        phi_school_extended = np.zeros((A, L, A, L))
        phi_work_extended = np.zeros((A, L, A, L))
        for a, b in product(range(A), range(A)):
            phi_ab_split = np.array(
                [
                    [d[b, 0], d[b, 1]],
                    [d[b, 0], d[b, 1]],
                ]
            )
            phi_ab_split = phi_ab_split / phi_ab_split.sum(1)
            phi_ab_split = 1 + 0 * phi_ab_split / phi_ab_split.sum(1)
            phi_all_extended[a, :, b, :] = self.phi_all[a, b] * phi_ab_split
            phi_school_extended[a, :, b, :] = self.phi_school[a, b] * phi_ab_split
            phi_work_extended[a, :, b, :] = self.phi_work[a, b] * phi_ab_split

        # Apply school closure and social distance
        # Assumes 95% reduction on last age group and high risk cocooning
        if day_type == 1:  # Weekday
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - school * phi_school_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = phi_all_extended - school * phi_school_extended
        elif day_type == 2 or day_type == 3:  # is a weekend or holiday
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - phi_school_extended - phi_work_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = (
                        phi_all_extended - phi_school_extended - phi_work_extended
                )
        else:
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - phi_school_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = phi_all_extended - phi_school_extended
        if cocooning > 0:
            # High risk cocooning and last age group cocooning
            phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
            phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
        assert (phi_age_risk >= 0).all()
        return phi_age_risk


class VariantPool:
    """
    A class that contains all the variant of concerns.
    """

    def __init__(self, variants_data: list, variants_prev: list):
        """
        :param variants_data: list of updates for each variant.
        :param variants_prev: prevalence of each variant of concern.
        """
        self.variants_data = variants_data
        self.variants_prev = variants_prev
        for (k, v) in self.variants_data['epi_params']["immune_evasion"].items():
            # calculate the rate of exponential immune evasion according to half-life (median) value:
            v["immune_evasion_max"] = log(2) / (v["half_life"] * 30) if v["half_life"] != 0 else 0
            v["start_date"] = dt.datetime.strptime(v["start_date"], datetime_formater)
            v["peak_date"] = dt.datetime.strptime(v["peak_date"], datetime_formater)
            v['days'] = (v["peak_date"] - v["start_date"]).days

    def update_params_coef(self, t: int, sigma_E: float):
        """
        update epi parameters and vaccine parameters according to prevalence of different variances.
        :param t: current date
        :param sigma_E: current sampled sigma_E value in the simulation.
        :return: new set of params and total variant prev.
        """
        new_epi_params_coef = {}
        new_vax_params = {}
        for (key, val) in self.variants_data['epi_params'].items():
            var_prev = sum(self.variants_prev[v][t] for v in val)
            if key == "sigma_E":
                # The parameter value of the triangular distribution is shifted with the Delta variant. Instead of
                # returning a percent increase in the parameter value, directly calculate the new sigma_E.
                new_epi_params_coef[key] = sum(1 / (1 / sigma_E - val[v]) * self.variants_prev[v][t] for v in val) + (
                            1 - var_prev) * sigma_E
            if key == "immune_evasion":
                pass
            else:
                # For other parameters calculate the change in the value as a coefficent:
                new_epi_params_coef[key] = 1 + sum(self.variants_prev[v][t] * (val[v] - 1) for v in val)

        # Calculate the new vaccine efficacy according to the variant values:
        for (key, val) in self.variants_data['vax_params'].items():
            for (k_dose, v_dose) in val.items():
                new_vax_params[(key, k_dose)] = sum(self.variants_prev[v][t] * v_dose[v] for v in v_dose)
        return new_epi_params_coef, new_vax_params, var_prev

    def immune_evasion(self, immune_evasion_base: float, t: dt.datetime):
        """
        I was planning to read the immune evasion value from the variant csv file, but we decide to run lsq on the
        immune evasion function, so I am integrating the piecewise linear function into the code.

        We assume the immunity evade with exponential rate.
        Calculate the changing immune evasion rate according to variant prevalence.
        Assume that the immune evasion follows a piecewise linear shape.
        It increases as the prevalence of variant increases and peak
        and starts to decrease.

        I assume the immune evasion functions of different variants do not overlap.

        half_life: half-life of the vaccine or natural infection induced protection.
        half_life_base: half-life of base level of immune evasion before the variant
        start_date: the date the immune evasion starts to increase
        peak_date: the date the immune evasion reaches the maximum level.

        :param immune_evasion_base: base immune evasion rate before the variant.
        :param t: current iterate
        :return: the immune evasion rate for a particular date.
        """
        for (k, v) in self.variants_data['epi_params']["immune_evasion"].items():
            if v["start_date"] <= t <= v["peak_date"]:
                return (t - v["start_date"]).days * (v["immune_evasion_max"] - immune_evasion_base) / v["days"] + immune_evasion_base
            elif v["peak_date"] <= t <= v["peak_date"] + dt.timedelta(days=v["days"]):
                return (v["peak_date"] + dt.timedelta(days=v["days"]) - t).days * (v["immune_evasion_max"] - immune_evasion_base) / v["days"] + immune_evasion_base

        return immune_evasion_base
        # return self.variants_prev['immune_evasion'][t]


class ParamDistribution:
    """
    A class to encapsulate epi paramters that are random
    Attrs:
        is_inverse (bool): if True, the parameter is used in the model as 1 / x.
        param_name (str): Name of the parameter, used in EpiParams as attribute name.
        distribution_name (str): Name of the distribution, matching functions in np.random.
        det_val (float): Value of the parameter for deterministic simulations.
        params (list): paramters if the distribution
    """

    def __init__(self, inv_opt, param_name, distribution_name, det_val, params):
        if inv_opt == "rnd_inverse":
            self.is_inverse = True
        elif inv_opt == "rnd":
            self.is_inverse = False
        self.param_name = param_name
        self.distribution_name = distribution_name
        self.det_val = det_val
        self.params = params

    def sample(self, rng, dim=1):
        """
        Sample random variable with given distribution name, parameters and dimension.
        Args:
            rng (np.random.default_rng): a random stream. If None, det_val is returned.
            dim (int or tuple): dimmention of the parameter (default is 1).
        """
        if rng is not None:
            dist_func = getattr(rng, self.distribution_name)
            args = self.params
            if self.is_inverse:
                return np.squeeze(1 / dist_func(*args, dim))
            else:
                return np.squeeze(dist_func(*args, dim))
        else:
            if self.is_inverse:
                return 1 / self.det_val
            else:
                return self.det_val
