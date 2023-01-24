###############################################################################

# SimModel.py
# This module contains the SimReplication class. Each instance holds
#   a City instance, an EpiSetup instance, a Vaccine instance,
#   VaccineGroup instance(s), and optionally a MultiTierPolicy instance.

###############################################################################

import numpy as np
from SimObjects import VaccineGroup
import copy
import datetime as dt

datetime_formater = "%Y-%m-%d %H:%M:%S"
import time


###############################################################################


class SimReplication:
    def __init__(self, instance, vaccine, policy, rng_seed):
        """
        :param instance: [obj] instance of City class
        :param vaccine: [obj] instance of Vaccine class
        :param policy: [obj] instance of MultiTierPolicy
            class, or [None]
        :param rng_seed: [int] or [None] either a
            non-negative integer, -1, or None
        """

        # Save arguments as attributes
        self.instance = instance
        self.vaccine = vaccine
        self.policy = policy
        self.rng_seed = rng_seed

        self.step_size = self.instance.config["step_size"]
        self.t_historical_data_end = len(self.instance.real_IH_history)

        # A is the number of age groups
        # L is the number of risk groups
        # Many data arrays in the simulation have dimension A x L
        A = self.instance.A
        L = self.instance.L

        # Important steps critical to initializing a replication
        # Initialize random number generator
        # Sample random parameters
        # Create new VaccineGroup instances
        self.init_rng()
        self.init_epi()
        self.init_vaccine_groups()

        # Initialize data structures to track ICU, IH, ToIHT, ToIY
        # These statistics or data we look at changes a lot over time
        # better to keep them in a list to modify.
        self.history_vars = ("ICU",
                             "IH",
                             "D",
                             "R",
                             "ToIHT",
                             "ToIY",
                             "ToICUD",
                             "ToIYD")

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        # The next t that is simulated (automatically gets updated after simulation)
        # This instance has simulated up to but not including time next_t
        self.next_t = 0

        # Tuples of variable names for organization purposes
        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = (
            "IYIH",
            "IYICU",
            "IHICU",
            "ToICU",
            "ToIHT",
            "ToICUD",
            "ToIYD",
            "ToIA",
            "ToIY",
        )

    def init_rng(self):
        """
        Assigns self.rng to a newly created random number generator
            initialized with seed self.rng_seed.
        If self.rng_seed is None (not specified) or -1, then self.rng
            is set to None, so no random number generator is created
            and the simulation will run deterministically.

        :return: [None]
        """

        if self.rng_seed:
            if self.rng_seed >= 0:
                self.rng = np.random.default_rng(self.rng_seed)
            else:
                self.rng = None
        else:
            self.rng = None

    def init_epi(self):
        """
        Assigns self.epi_rand to an instance of EpiSetup that
            inherits some attribute values (primitives) from
            the "base" object self.instance.base_epi and
            also generates new values for other attributes.
        These new values come from randomly sampling
            parameters using the random number generator
            self.rng.
        If no random number generator is given, these
            randomly sampled parameters are set to the
            expected value from their distributions.
        After random sampling, some basic parameters
            are updated.

        :return: [None]
        """

        # Create a deep copy of the "base" EpiSetup instance
        #   to inherit some attribute values (primitives)
        epi_rand = copy.deepcopy(self.instance.base_epi)

        # On this copy, sample random parameters and
        #   do some basic updating based on the results
        #   of this sampling
        epi_rand.sample_random_params(self.rng)
        epi_rand.setup_base_params()

        # Assign self.epi_rand to this copy
        self.epi_rand = epi_rand

    def init_vaccine_groups(self):
        """
        Creates 4 vaccine groups:
            group 0 / "v_0": unvaccinated
            group 1 / "v_1": partially vaccinated
            group 2 / "v_2": fully vaccinated
            group 3 / "v_3": waning efficacy

        We assume there is one type of vaccine with 2 doses.
        After 1 dose, individuals move from group 0 to 1.
        After 2 doses, individuals move from group 1 to group 2.
        After efficacy wanes, individuals move from group 2 to group 3.
        After booster shot, individuals move from group 3 to group 2.
                 - one type of vaccine with two-doses

        :return: [None]
        """

        N = self.instance.N
        I0 = self.instance.I0
        A = self.instance.A
        L = self.instance.L
        step_size = self.instance.config["step_size"]

        self.vaccine_groups = []
        self.vaccine_groups.append(VaccineGroup("v_0", 0, 0, 0, 0, 0, N, I0, A, L, step_size))
        self.vaccine_groups.append(
            VaccineGroup("v_1", self.vaccine.beta_reduct[1], self.vaccine.tau_reduct[1],
                         self.vaccine.beta_reduct_delta[1], self.vaccine.tau_reduct_delta[1],
                         self.vaccine.tau_reduct_omicron[1], N, I0, A, L, step_size)
        )
        self.vaccine_groups.append(
            VaccineGroup("v_2", self.vaccine.beta_reduct[2], self.vaccine.tau_reduct[2],
                         self.vaccine.beta_reduct_delta[2], self.vaccine.tau_reduct_delta[2],
                         self.vaccine.tau_reduct_omicron[2], N, I0, A, L, step_size)
        )
        self.vaccine_groups.append(
            VaccineGroup("v_3", self.vaccine.beta_reduct[0], self.vaccine.tau_reduct[0],
                         self.vaccine.beta_reduct_delta[0], self.vaccine.tau_reduct_delta[0],
                         self.vaccine.tau_reduct_omicron[0], N, I0, A, L, step_size)
        )
        self.vaccine_groups = tuple(self.vaccine_groups)

    def compute_cost(self):
        """
        If a policy is attached to this replication, return the
            cumulative cost of its enforced tiers (from
            the end of the historical data time period to the
            current time of the simulation).
        If no policy is attached to this replication, return
            None.

        :return: [float] or [None] cumulative cost of the
            attached policy's enforced tiers (returns None
            if there is no attached policy)
        """

        if self.policy:
            return sum(
                self.policy.tiers[i]["daily_cost"]
                for i in self.policy.tier_history
                if i is not None
            )
        else:
            return None

    def compute_feasibility(self):
        """
        If a policy is attached to this replication, return
            True/False if the policy is estimated to be
            feasible (from the end of the historical data time period
            to the current time of the simulation).
        If no policy is attached to this replication or the
            current time of the simulation is still within
            the historical data time period, return None.

        :return: [Boolean] or [None] corresponding to whether
            or not the policy is estimated to be feasible
        """

        if self.policy is None:
            return None
        elif self.next_t < self.t_historical_data_end:
            return None

        # Check whether ICU capacity has been violated
        if np.any(
                np.array(self.ICU_history).sum(axis=(1, 2))[self.t_historical_data_end:]
                > self.instance.icu
        ):
            return False
        else:
            return True

    def compute_rsq(self):
        """
        Return R-squared type statistic based on historical hospital
            data (see pg. 10 in Yang et al. 2021), comparing
            thus-far-simulated hospital numbers (starting from t = 0
            up to the current time of the simulation) to the
            historical data hospital numbers (over this same time
            interval).

        Note that this statistic is not exactly R-squared --
            and as a result it takes values outside of [-1, 1].

        :return: [float] current R-squared value
        """

        f_benchmark = self.instance.real_IH_history

        IH_sim = np.array(self.ICU_history) + np.array(self.IH_history)
        IH_sim = IH_sim.sum(axis=(2, 1))
        IH_sim = IH_sim[: self.t_historical_data_end]

        if self.next_t < self.t_historical_data_end:
            IH_sim = IH_sim[: self.next_t]
            f_benchmark = f_benchmark[: self.next_t]

        rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark)) ** 2)) / sum(
            (np.array(f_benchmark) - np.mean(np.array(f_benchmark))) ** 2
        )

        return rsq

    def immune_escape(self, immune_escape_rate, t):
        """
        This function moves recovered and vaccinated individuals to waning
            efficacy susceptible compartment after Omicron becomes the prevalent
            virus type.

        :return: [None]
        """

        for v_groups in self.vaccine_groups:
            moving_people = v_groups._R[self.step_size] * immune_escape_rate
            v_groups.R -= moving_people
            self.vaccine_groups[3].S += moving_people

            if v_groups.v_name == "v_1" or v_groups.v_name == "v_2":
                moving_people = v_groups._S[self.step_size] * immune_escape_rate
                v_groups.S -= moving_people
                self.vaccine_groups[3].S += moving_people

    def simulate_time_period(self, time_end):

        """
        Advance the simulation model from time_start up to
            but not including time_end.

        Calls simulate_t as a subroutine for each t between
            time_start and self.next_t, the last point at which it
            left off.

        :param time_end: [int] nonnegative integer -- time t (number of days)
            to simulate up to.
        :return: [None]
        """

        time_start = self.next_t
        for t in range(time_start, time_end):

            self.next_t += 1

            self.simulate_t(t)

            A = self.instance.A
            L = self.instance.L

            for attribute in self.state_vars + self.tracking_vars:
                setattr(self, attribute, np.zeros((A, L)))

            for attribute in self.state_vars + self.tracking_vars:
                sum_across_vaccine_groups = 0
                for v_group in self.vaccine_groups:
                    sum_across_vaccine_groups += getattr(v_group, attribute)
                setattr(self, attribute, sum_across_vaccine_groups)

            for attribute in self.history_vars:
                getattr(self, f"{attribute}_history").append(getattr(self, attribute))

            total_imbalance = np.sum(
                self.S
                + self.E
                + self.IA
                + self.IY
                + self.R
                + self.D
                + self.PA
                + self.PY
                + self.IH
                + self.ICU
            ) - np.sum(self.instance.N)

            assert (
                    np.abs(total_imbalance) < 1e-2
            ), f"fPop unbalanced {total_imbalance} at time {self.instance.cal.calendar[t]}, {t}"

    def simulate_t(self, t_date):

        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.calendar

        t = t_date

        epi = copy.deepcopy(self.epi_rand)
        if self.instance.cal.fixed_transmission_reduction[t] is not None:
            # We can fix the transmission reduction as part of projections.
            # It is better to keep real historical data separate from the tiers.
            phi_t = epi.effective_phi(
                self.instance.cal.schools_closed[t],
                self.instance.cal.fixed_cocooning[t],
                self.instance.cal.fixed_transmission_reduction[t],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )
        else:
            self.policy(
                t,
                self.ToIHT_history,
                self.IH_history,
                self.ToIY_history,
                self.ICU_history,
            )
            current_tier = self.policy.tier_history[t]
            phi_t = epi.effective_phi(
                self.policy.tiers[current_tier]["school_closure"],
                self.policy.tiers[current_tier]["cocooning"],
                self.policy.tiers[current_tier]["transmission_reduction"],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )

        if calendar[t] >= self.instance.delta_start:
            days_since_delta_start = (calendar[t] - self.instance.delta_start).days
            for v_groups in self.vaccine_groups:
                v_groups.delta_update(self.instance.delta_prev[days_since_delta_start])
            epi.delta_update_param(self.instance.delta_prev[days_since_delta_start])

        # Update epi parameters for omicron:
        if calendar[t] >= self.instance.omicron_start:
            days_since_omicron_start = (calendar[t] - self.instance.omicron_start).days
            epi.omicron_update_param(
                self.instance.omicron_prev[days_since_omicron_start]
            )
            for v_groups in self.vaccine_groups:
                v_groups.omicron_update(
                    self.instance.omicron_prev[days_since_omicron_start]
                )

        # Assume an imaginary new variant in May, 2022:
        if epi.new_variant:
            days_since_variant_start = (calendar[t] - self.instance.variant_start).days
            if calendar[t] >= self.instance.variant_start:
                epi.variant_update_param(
                    self.instance.variant_prev[days_since_variant_start]
                )

        if self.instance.otherInfo == {}:
            rd_start = dt.datetime.strptime(
                self.instance.config["rd_start"], datetime_formater
            )
            rd_end = dt.datetime.strptime(
                self.instance.config["rd_end"], datetime_formater
            )
            if self.instance.cal.calendar.index(
                    rd_start
            ) < t <= self.instance.cal.calendar.index(rd_end):
                epi.update_icu_params(self.instance.config["rd_rate"])
        else:
            epi.update_icu_all(t, self.instance.otherInfo)

        discrete_approx = self.discrete_approx
        step_size = self.step_size
        get_binomial_transition_quantity = self.get_binomial_transition_quantity

        rate_E = discrete_approx(epi.sigma_E, step_size)
        rate_IYR = discrete_approx(
            np.array(
                [
                    [
                        (1 - epi.pi[a, l]) * epi.gamma_IY * (1 - epi.alpha4)
                        for l in range(L)
                    ]
                    for a in range(A)
                ]
            ),
            step_size,
        )
        rate_IYD = discrete_approx(
            np.array(
                [
                    [(1 - epi.pi[a, l]) * epi.gamma_IY * epi.alpha4 for l in range(L)]
                    for a in range(A)
                ]
            ),
            step_size,
        )

        rate_IAR = discrete_approx(np.full((A, L), epi.gamma_IA), step_size)
        rate_PAIA = discrete_approx(np.full((A, L), epi.rho_A), step_size)
        rate_PYIY = discrete_approx(np.full((A, L), epi.rho_Y), step_size)

        rate_IYH = discrete_approx(
            np.array(
                [
                    [(epi.pi[a, l]) * epi.Eta[a] * epi.pIH for l in range(L)]
                    for a in range(A)
                ]
            ),
            step_size,
        )
        rate_IYICU = discrete_approx(
            np.array(
                [
                    [(epi.pi[a, l]) * epi.Eta[a] * (1 - epi.pIH) for l in range(L)]
                    for a in range(A)
                ]
            ),
            step_size,
        )
        rate_IHICU = discrete_approx(epi.nu * epi.etaICU, step_size)
        rate_IHR = discrete_approx((1 - epi.nu) * epi.gamma_IH, step_size)
        rate_ICUD = discrete_approx(epi.nu_ICU * epi.mu_ICU, step_size)
        rate_ICUR = discrete_approx((1 - epi.nu_ICU) * epi.gamma_ICU, step_size)

        start = time.time()

        if t >= 711:  # date corresponding to 02/07/2022
            rate_immune = discrete_approx(epi.immune_evasion, step_size)

        for _t in range(step_size):
            # Dynamics for dS

            for v_groups in self.vaccine_groups:

                dSprob_sum = np.zeros((5, 2))

                for v_groups_temp in self.vaccine_groups:
                    # Vectorized version for efficiency. For-loop version commented below
                    temp1 = (
                            np.matmul(np.diag(epi.omega_PY), v_groups_temp._PY[_t, :, :])
                            + np.matmul(np.diag(epi.omega_PA), v_groups_temp._PA[_t, :, :])
                            + epi.omega_IA * v_groups_temp._IA[_t, :, :]
                            + epi.omega_IY * v_groups_temp._IY[_t, :, :]
                    )

                    temp2 = np.sum(N, axis=1)[np.newaxis].T
                    temp3 = np.divide(
                        np.multiply(epi.beta * phi_t / step_size, temp1), temp2
                    )

                    dSprob = np.sum(temp3, axis=(2, 3))
                    dSprob_sum = dSprob_sum + dSprob

                if (
                        t >= 711 and v_groups.v_name == "v_2"
                ):  # date corresponding to 02/07/2022
                    _dS = get_binomial_transition_quantity(
                        v_groups._S[_t],
                        rate_immune + (1 - v_groups.v_beta_reduct) * dSprob_sum,
                    )
                    # Dynamics for E
                    _dSE = (
                            _dS
                            * ((1 - v_groups.v_beta_reduct) * dSprob_sum)
                            / (rate_immune + (1 - v_groups.v_beta_reduct) * dSprob_sum)
                    )

                    E_out = get_binomial_transition_quantity(v_groups._E[_t], rate_E)
                    v_groups._E[_t + 1] = v_groups._E[_t] + _dSE - E_out

                    _dSR = _dS - _dSE
                    self.vaccine_groups[3]._S[_t + 1] = (
                            self.vaccine_groups[3]._S[_t + 1] + _dSR
                    )

                else:
                    _dS = get_binomial_transition_quantity(
                        v_groups._S[_t], (1 - v_groups.v_beta_reduct) * dSprob_sum
                    )
                    # Dynamics for E
                    E_out = get_binomial_transition_quantity(v_groups._E[_t], rate_E)
                    v_groups._E[_t + 1] = v_groups._E[_t] + _dS - E_out

                if t >= 711 and v_groups.v_name != "v_3":
                    immune_escape_R = get_binomial_transition_quantity(
                        v_groups._R[_t], rate_immune
                    )
                    self.vaccine_groups[3]._S[_t + 1] = (
                            self.vaccine_groups[3]._S[_t + 1] + immune_escape_R
                    )

                v_groups._S[_t + 1] = v_groups._S[_t + 1] + v_groups._S[_t] - _dS

                # Dynamics for PY
                EPY = get_binomial_transition_quantity(
                    E_out, epi.tau * (1 - v_groups.v_tau_reduct)
                )
                PYIY = get_binomial_transition_quantity(v_groups._PY[_t], rate_PYIY)
                v_groups._PY[_t + 1] = v_groups._PY[_t] + EPY - PYIY

                # Dynamics for PA
                EPA = E_out - EPY
                PAIA = get_binomial_transition_quantity(v_groups._PA[_t], rate_PAIA)
                v_groups._PA[_t + 1] = v_groups._PA[_t] + EPA - PAIA

                # Dynamics for IA
                IAR = get_binomial_transition_quantity(v_groups._IA[_t], rate_IAR)
                v_groups._IA[_t + 1] = v_groups._IA[_t] + PAIA - IAR

                # Dynamics for IY
                IYR = get_binomial_transition_quantity(v_groups._IY[_t], rate_IYR)
                IYD = get_binomial_transition_quantity(v_groups._IY[_t] - IYR, rate_IYD)
                v_groups._IYIH[_t] = get_binomial_transition_quantity(
                    v_groups._IY[_t] - IYR - IYD, rate_IYH
                )
                v_groups._IYICU[_t] = get_binomial_transition_quantity(
                    v_groups._IY[_t] - IYR - IYD - v_groups._IYIH[_t], rate_IYICU
                )
                v_groups._IY[_t + 1] = (
                        v_groups._IY[_t]
                        + PYIY
                        - IYR
                        - IYD
                        - v_groups._IYIH[_t]
                        - v_groups._IYICU[_t]
                )

                # Dynamics for IH
                IHR = get_binomial_transition_quantity(v_groups._IH[_t], rate_IHR)
                v_groups._IHICU[_t] = get_binomial_transition_quantity(
                    v_groups._IH[_t] - IHR, rate_IHICU
                )
                v_groups._IH[_t + 1] = (
                        v_groups._IH[_t] + v_groups._IYIH[_t] - IHR - v_groups._IHICU[_t]
                )

                # Dynamics for ICU
                ICUR = get_binomial_transition_quantity(v_groups._ICU[_t], rate_ICUR)
                ICUD = get_binomial_transition_quantity(
                    v_groups._ICU[_t] - ICUR, rate_ICUD
                )
                v_groups._ICU[_t + 1] = (
                        v_groups._ICU[_t]
                        + v_groups._IHICU[_t]
                        - ICUD
                        - ICUR
                        + v_groups._IYICU[_t]
                )
                v_groups._ToICU[_t] = v_groups._IYICU[_t] + v_groups._IHICU[_t]
                v_groups._ToIHT[_t] = v_groups._IYICU[_t] + v_groups._IYIH[_t]

                # Dynamics for R
                if t >= 711 and v_groups.v_name != "v_3":
                    v_groups._R[_t + 1] = (
                            v_groups._R[_t] + IHR + IYR + IAR + ICUR - immune_escape_R
                    )
                else:
                    v_groups._R[_t + 1] = v_groups._R[_t] + IHR + IYR + IAR + ICUR

                # Dynamics for D
                v_groups._D[_t + 1] = v_groups._D[_t] + ICUD + IYD
                v_groups._ToICUD[_t] = ICUD
                v_groups._ToIYD[_t] = IYD
                v_groups._ToIA[_t] = PAIA
                v_groups._ToIY[_t] = PYIY

        start = time.time()

        for v_groups in self.vaccine_groups:
            # End of the daily discretization
            for attribute in self.state_vars:
                setattr(
                    v_groups,
                    attribute,
                    getattr(v_groups, "_" + attribute)[step_size].copy(),
                )

            for attribute in self.tracking_vars:
                setattr(
                    v_groups, attribute, getattr(v_groups, "_" + attribute).sum(axis=0)
                )

        if calendar[t] == self.instance.omicron_start:
            # Move almost half of the people from recovered to susceptible:
            self.immune_escape(epi.immune_escape_rate, t)

        if t >= self.vaccine.vaccine_start_time:

            S_before = np.zeros((5, 2))

            for v_groups in self.vaccine_groups:
                S_before += v_groups.S

            for v_groups in self.vaccine_groups:

                out_sum = np.zeros((A, L))
                S_out = np.zeros((A * L, 1))
                N_out = np.zeros((A * L, 1))

                for vaccine_type in v_groups.v_out:
                    event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                    if event is not None:

                        S_out = np.reshape(
                            self.vaccine.vaccine_allocation[vaccine_type][event][
                                "assignment"
                            ],
                            (A * L, 1),
                        )
                        if calendar[t] >= self.instance.omicron_start:
                            if v_groups.v_name == "v_1" or v_groups.v_name == "v_2":
                                S_out = epi.immune_escape_rate * np.reshape(
                                    self.vaccine.vaccine_allocation[vaccine_type][
                                        event
                                    ]["assignment"],
                                    (A * L, 1),
                                )

                        N_out = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_groups.v_name,
                            v_groups.v_in,
                            v_groups.v_out,
                            calendar[t],
                        )

                        ratio_S_N = np.array(
                            [
                                0 if N_out[i] == 0 else float(S_out[i] / N_out[i])
                                for i in range(len(N_out))
                            ]
                        ).reshape((A, L))

                        out_sum += ratio_S_N * v_groups._S[step_size]

                in_sum = np.zeros((A, L))
                S_in = np.zeros((A * L, 1))
                N_in = np.zeros((A * L, 1))

                for vaccine_type in v_groups.v_in:

                    for v_g in self.vaccine_groups:
                        if (
                                v_g.v_name
                                == self.vaccine.vaccine_allocation[vaccine_type][0]["from"]
                        ):
                            v_temp = v_g

                    event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                    if event is not None:
                        S_in = np.reshape(
                            self.vaccine.vaccine_allocation[vaccine_type][event][
                                "assignment"
                            ],
                            (A * L, 1),
                        )

                        if calendar[t] >= self.instance.omicron_start:
                            if (
                                    v_groups.v_name == "v_3" and v_temp.v_name == "v_2"
                            ) or (v_groups.v_name == "v_2" and v_temp.v_name == "v_1"):
                                S_in = epi.immune_escape_rate * np.reshape(
                                    self.vaccine.vaccine_allocation[vaccine_type][
                                        event
                                    ]["assignment"],
                                    (A * L, 1),
                                )

                        N_in = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_temp.v_name,
                            v_temp.v_in,
                            v_temp.v_out,
                            calendar[t],
                        )
                        ratio_S_N = np.array(
                            [
                                0 if N_in[i] == 0 else float(S_in[i] / N_in[i])
                                for i in range(len(N_in))
                            ]
                        ).reshape((A, L))

                        in_sum += ratio_S_N * v_temp._S[step_size]

                v_groups.S = v_groups.S + (np.array(in_sum - out_sum))

                S_after = np.zeros((5, 2))

            for v_groups in self.vaccine_groups:
                S_after += v_groups.S

            imbalance = np.abs(np.sum(S_before - S_after, axis=(0, 1)))

            assert (imbalance < 1e-2).any(), (
                f"fPop inbalance in vaccine flow in between compartment S "
                f"{imbalance} at time {calendar[t]}, {t}"
            )

        for v_groups in self.vaccine_groups:

            for attribute in self.state_vars:
                setattr(v_groups, "_" + attribute, np.zeros((step_size + 1, A, L)))
                vars(v_groups)["_" + attribute][0] = vars(v_groups)[attribute]

            for attribute in self.tracking_vars:
                setattr(v_groups, "_" + attribute, np.zeros((step_size, A, L)))

    def reset(self):

        A = self.instance.A
        L = self.instance.L

        self.init_vaccine_groups()

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        self.next_t = 0

    def get_binomial_transition_quantity(self, n, p, round_opt=1):

        if self.rng is None:
            return n * p
        else:
            if round_opt:
                nInt = np.round(n)
                return self.rng.binomial(nInt.astype(int), p)
            else:
                return self.rng.binomial(n, p)

    def discrete_approx(self, rate, timestep):
        return 1 - np.exp(-rate / timestep)
