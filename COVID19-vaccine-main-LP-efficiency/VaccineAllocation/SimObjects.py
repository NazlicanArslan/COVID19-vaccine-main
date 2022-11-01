import numpy as np

datetime_formater = "%Y-%m-%d %H:%M:%S"


class MultiTierPolicy:
    """
    A multi-tier policy allows for multiple tiers of lock-downs.
    Attrs:
        tiers (list of dict): a list of the tiers characterized by a dictionary
            with the following entries:
                {
                    "transmission_reduction": float [0,1)
                    "cocooning": float [0,1)
                    "school_closure": int {0,1}
                }

        lockdown_thresholds (list of list): a list with the thresholds for every
            tier. The list must have n-1 elements if there are n tiers. Each threshold
            is a list of values for evert time step of simulation.
        community_tranmission: (deprecated) CDC's old community tranmission threshold for staging.
                                Not in use anymore.
    """

    def __init__(self, instance, tiers, lockdown_thresholds, community_transmission):
        self._instance = instance
        self.tiers = tiers.tier

        self.community_transmission = community_transmission
        self.lockdown_thresholds = lockdown_thresholds
        self.tier_history = None

    def reset(self):
        self.tier_history = None

    def __repr__(self):
        return str(self.lockdown_thresholds)

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        """
        Function that makes an instance of a policy a callable.
        Args:
            t (int): time period in the simulation
            z (object): deprecated, but maintained to avoid changes in the simulate function
            criStat (ndarray): the trigger statistics, previously daily admission, passed by the simulator
            IH (ndarray): hospitalizations admissions, passed by the simulator
            ** kwargs: additional parameters that are passed and are used elsewhere
        """
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]

        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        IH = np.array(IH)
        ToIY = np.array(ToIY)
        ICU = np.array(ICU)

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - self._instance.config["moving_avg_len"])
        criStat_total = ToIHT.sum((1, 2))
        criStat_avg = criStat_total[moving_avg_start:].mean()

        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = (
                ToIY.sum((1, 2))[moving_avg_start:].sum()
                * 100000
                / np.sum(N, axis=(0, 1))
            )
        else:
            ToIY_avg = 0

        current_tier = self.tier_history[t - 1]

        # find new tier
        counter = 0
        lb_threshold = 0
        for lt in self.lockdown_thresholds:
            if criStat_avg >= lt:
                lb_threshold = counter
                counter += 1
                if counter == len(self.lockdown_thresholds):
                    break

        new_tier = lb_threshold

        # Check if community_tranmission rate is included:
        if self.community_transmission == "blue":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2
            elif new_tier == 1:
                if ToIY_avg > 10:
                    new_tier = 2
        elif self.community_transmission == "green":
            if new_tier == 0:
                if ToIY_avg > 5:
                    if ToIY_avg < 10:
                        new_tier = 1
                    else:
                        new_tier = 2

        if current_tier is None:  # bump to the next tier
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]

        elif new_tier > current_tier:
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]

        elif new_tier < current_tier:  # relax one tier, if safety trigger allows
            IH_total = IH[-1].sum()
            assert_safety_trigger = (
                IH_total
                < self._instance.hosp_beds
                * self._instance.config["safety_threshold_frac"]
            )
            new_tier = new_tier if assert_safety_trigger else current_tier
            t_delta = (
                self.tiers[new_tier]["min_enforcing_time"]
                if assert_safety_trigger
                else 1
            )
            t_end = t + t_delta

        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]


class VaccineGroup:
    def __init__(
        self,
        v_name,
        v_beta_reduct,
        v_tau_reduct,
        v_beta_reduct_delta,
        v_tau_reduct_delta,
        v_tau_reduct_omicron,
        instance,
    ):
        """
        Define each vaccine status as a group. Define each set of compartments for vaccine group.

        Parameters
        ----------
        v_name : string
            vaccine status type.
        v_beta_reduct : float [0,1]
            reduction in transmission.
        v_tau_reduct : float [0,1]
            reduction in symptomatic infection.
        instance :
            problem instance.

        """
        self.v_beta_reduct = v_beta_reduct
        self.v_tau_reduct = v_tau_reduct
        self.v_beta_reduct_delta = v_beta_reduct_delta
        self.v_tau_reduct_delta = v_tau_reduct_delta
        self.v_tau_reduct_omicron = v_tau_reduct_omicron
        self.v_name = v_name

        if self.v_name == "v_0":
            self.v_in = ()
            self.v_out = ("v_first",)

        elif self.v_name == "v_1":
            self.v_in = ("v_first",)
            self.v_out = ("v_second",)

        elif self.v_name == "v_2":
            self.v_in = ("v_second", "v_booster")
            self.v_out = ("v_wane",)

        else:
            self.v_in = ("v_wane",)
            self.v_out = ("v_booster",)

        self.N = instance.N
        self.I0 = instance.I0

        A = instance.A
        L = instance.L
        step_size = instance.config["step_size"]

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

        for attribute in self.state_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size + 1, A, L)))

        for attribute in self.tracking_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size, A, L)))

        if self.v_name == "v_0":
            # Initial Conditions (assumed)
            self.PY = self.I0
            self.R = 0
            self.S = self.N - self.PY - self.IY

        for attribute in self.state_vars:
            vars(self)["_" + attribute][0] = getattr(self, attribute)

    def delta_update(self, prev):
        """
        Update efficacy according to delta variant (VoC) prevelance.
        """

        self.v_beta_reduct = (
            self.v_beta_reduct * (1 - prev) + self.v_beta_reduct_delta * prev
        )  # decreased efficacy against infection.
        self.v_tau_reduct = (
            self.v_tau_reduct * (1 - prev) + self.v_tau_reduct_delta * prev
        )  # decreased efficacy against symptomatic infection.

    def omicron_update(self, prev):
        """
        Update efficacy according to omicron variant (VoC) prevelance.
        """

        self.v_tau_reduct = (
            self.v_tau_reduct * (1 - prev) + self.v_tau_reduct_omicron * prev
        )
