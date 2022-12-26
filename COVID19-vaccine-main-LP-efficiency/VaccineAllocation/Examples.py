###############################################################################

# Examples.py
# This document contains examples of how to use the simulation code.

# To launch the examples, either
# (1) Run the following command in the OS command line or Terminal:
#   python3 Examples.py
# (2) Copy and paste the code of this document into an interactive
#   Python console.

# Note that if modules cannot be found, this is a path problem.
# In both fixes below, replace <NAME_OF_YOUR_DIRECTORY> with a string
#   containing the directory in which the modules reside.
# (1) The following path can be updated via the OS command line or
#   Terminal (e.g. on a cluster):
#   export PYTHONPATH=<NAME_OF_YOUR_DIRECTORY>:$PYTHONPATH
# (2) The following code can be added to the main .py script file
#   (e.g. can be added to the top of this document):
#   import sys
#   sys.path.append(<NAME_OF_YOUR_DIRECTORY>)

# Linda Pei 2022

###############################################################################

# Import other code modules
# SimObjects contains classes of objects that change within a simulation
#   replication.
# DataObjects contains classes of objects that do not change
#   within simulation replications and also do not change *across*
#   simulation replications -- they contain data that are "fixed"
#   for an overall problem.
# SimModel contains the class SimReplication, which runs
#   a simulation replication and stores the simulation data
#   in its object attributes.
# InputOutputTools contains utility functions that act on
#   instances of SimReplication to load and export
#   simulation states and data.
# OptTools contains utility functions for optimization purposes.

import copy
from SimObjects import MultiTierPolicy, CDCTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
import InputOutputTools
import OptTools

# Import other Python packages
import numpy as np

###############################################################################

# Mandatory definitions from user-input files
# In general, these following lines must be in every script that uses
#   the simulation code. The specific names of the files used may change.
# Each simulation replication requires these 3 instances
#   (built from reading .json and .csv files)
# (1) City instance that holds calendar, city-specific epidemiological
#   parameters, historical hospital data, and fitted transmission parameters
# (2) TierInfo instance that contains information about the tiers in a
#   social distancing threshold policy
# (3) Vaccine instance that holds vaccine groups and historical
#   vaccination data

austin = City("austin",
              "austin_test_IHT.json",
              "calendar.csv",
              "setup_data_Final.json",
              "transmission.csv",
              "austin_real_hosp_updated.csv",
              "austin_real_icu_updated.csv",
              "austin_hosp_ad_updated.csv",
              "austin_real_death_from_hosp_updated.csv",
              "austin_real_total_death.csv",
              "delta_prevalence.csv",
              "omicron_prevalence.csv",
              "variant_prevalence.csv")

tiers = TierInfo("austin", "tiers5_opt_Final.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

###############################################################################

# The following examples build on each other, so it is
#   recommended to study them in order.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example A: Simulating a threshold policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In general, simulating a policy requires the steps
# (1) Create a MultiTierPolicy instance with desired thresholds.
# (2) Create a SimReplication instance with  aforementioned policy
#   and a random number seed -- this seed dictates the randomly sampled
#   epidemiological parameters for that replication as well as the
#   binomial random variable transitions between compartments in the
#   SEIR-type model.
# (3) Advance simulation time.

# Specify the 5 thresholds for a 5-tier policy
thresholds = (-1, 100, 200, 500, 1000)

# Create an instance of MultiTierPolicy using
#   austin, tiers (defined above)
#   thresholds (defined above)
#   "green" as the community_transmission toggle
# Prof Morton mentioned that setting community_transmission to "green"
#   was a government official request to stop certain "drop-offs"
#   in active tiers.
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")

# Create an instance of SimReplication with seed 500.
rep = SimReplication(austin, vaccines, mtp, 500)

# Note that specifying a seed of -1 creates a simulation replication
#   with average values for the "random" epidemiological parameter
#   values and deterministic binomial transitions
#   (also taking average values).

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
rep.simulate_time_period(945)

# After simulating, we can query the R-squared.
# If the simulation has been simulated for fewer days than the
#   timeframe of the historical time period, the R-squared is
#   computed for this subset of days.
print(rep.compute_rsq())

# After simulating, we can query the cost of the specified policy.
print(rep.compute_cost())

# We can also query whether the specified policy is
#   feasible, i.e. whether it prevents an ICU capacity violation.
print(rep.compute_feasibility())

# If we want to test the same policy on a different sample path,
#   we can still use the same policy object as long as we clear it.
mtp.reset()

# Now we create an instance of SimReplication with seed 1010.
rep = SimReplication(austin, vaccines, mtp, 1010)

# Compare the R-squared and costs of seed 1010 versus seed 500.
# Note that so far we are not simulating our policy on
#   a "realistic" sample path -- here we just picked an arbitrary seed.
rep.simulate_time_period(945)
print(rep.compute_rsq())
print(rep.compute_cost())

# Notice that the cost is 0. We can sanity-check this cost
#   by confirming that no stage-alert thresholds were crossed
#   and no social distancing cost was incurred over this
#   time horizon.
# Each policy has its tier history saved as an attribute.
print(rep.policy.tier_history)

# Note that calling compute_rsq, compute_cost, or compute_feasibility
#   if rep has not yet been simulated, or it has been cleared, leads to
#   an error.
# Calling compute_cost if there is no policy attached
#   to the replication (i.e., if rep.policy = None) leads to an error.
#   Similarly, we also need to simulate rep (with a policy attached)
#   *past* the historical time period so that the policy goes
#   into effect before querying its cost.

# Clearing an instance of SimReplication is a bit tricky, so
#   be careful of this nuance. The following reset() method
#   clears the replication ("zero-ing" any saved data
#   as well as the current time).
# However, the randomly sampled parameters remain the same!
#   These are untouched.
# The random number generator is also untouched after
#   reset(), so simulating rep will draw random numbers
#   from where the random number generator last left off
#   (before the reset).
rep.reset()

# Due to the nuances of the random number generation,
#   in many cases it is more straightforward and less
#   risky to simply create a new replication rather than
#   reset the replication. We recommend this particularly
#   when debugging and comparing numbers during code
#   testing (since being careful with random number
#   generation is necessary for perfectly replicable
#   results).

###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example B: Stopping and starting a simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Here we demonstrate how to start and stop a simulation replication
#   both within a Python session and across Python sessions,
#   potentially across different computers.

# First we demonstrate starting and stopping a replication
#   within a Python session.
# Here we create a new a policy and attach it to a new replication.
thresholds = (-1, 1, 10, 20, 100)
mtp = MultiTierPolicy(austin, tiers, thresholds, "green")
rep = SimReplication(austin, vaccines, mtp, 1000)

# Now we simulate the replication up to time 100
rep.simulate_time_period(100)

# Within a Python session, we can simulate from where
#   we left off, at time 100.
# We can simulate incrementally as many times as we want
#   and in whatever increments we want, so long as the
#   times are integer-valued.
# Note that we can only start and stop at the beginning
#   and end of whole days -- we cannot stop in the
#   middle of a day or stop at a discretized step
#   at time t.
rep.simulate_time_period(200)

# The method simulate_time_period will automatically start
#   simulating from the last place we left off.
rep.simulate_time_period(300)
rep.simulate_time_period(945)
print(rep.compute_rsq())

# The attribute next_t in a simulation replication
#   is the next time that will be simulated.
print(rep.next_t)

# Let's compare r-squared values to confirm that
#   starting and stopping the simulation until time t
#   within a session is identical to running
#   the simulation continuously until time t.
rep.policy.reset()
rep = SimReplication(austin, vaccines, mtp, 1000)
rep.simulate_time_period(945)
print(rep.compute_rsq())

# Next we demonstrate stopping and starting a simulation rep
#   across a Python session, including across computers.

# Let's simulate our policy up to time 300.
rep.policy.reset()
rep = SimReplication(austin, vaccines, mtp, 1000)
rep.simulate_time_period(800)

# Now we export the current state of the simulation
#   as multiple .json files.
# We pass the SimReplication object, a filename
#   for the SimReplication object data, filenames
#   for the 4 VaccineGroup objects, an optional
#   filename for the MultiTierPolicy object, and
#   an optional filename for the random parameters
#   sampled in the EpiSetup object.
# These files will save in the same directory as the
#   main .py file.
InputOutputTools.export_rep_to_json(rep, "sim_rep.json",
                                    "v0.json", "v1.json", "v2.json", "v3.json",
                                    "policy.json", "random_params.json")

# To read-in previously saved simulation states,
#   we create a new SimReplication instance and apply
#   import_rep_from_json on it.
# The files must be in the same directory as the main .py file.
# Note the line about rep.rng -- we will explain this later.
rep = SimReplication(austin, vaccines, mtp, 1000)
InputOutputTools.import_rep_from_json(rep, "sim_rep.json",
                                      "v0.json", "v1.json", "v2.json", "v3.json",
                                      "policy.json", "random_params.json")
rep.rng = np.random.default_rng(10000)

# Now rep.next_time is 800, where we last left off.
# We can simulate rep.next_time from this point.
rep.simulate_time_period(945)

# As another sanity check, let's compare the costs
#   of this exported and imported simulation replication
#   with a simulation replication that does not export or
#   import data.
print(rep.compute_cost())

# Here are results from "internal" starting and stopping.
rep.policy.reset()
rep = SimReplication(austin, vaccines, mtp, 1000)
rep.simulate_time_period(800)
rep.rng = np.random.default_rng(10000)
rep.simulate_time_period(945)
print(rep.compute_cost())

# Notice the line rep.rng = np.random.default_rng(10000)
#   that is called after we import .json files for time 800
#   when "externally" starting and stopping and
#   is called after we simulate to time 800 when
#   "internally" starting and stopping.
# This resets the random number generator using seed
#   10000. This is not required for starting and stopping
#   a simulation, either externally or internally.
# As discussed in Example A, to get perfectly repeatable
#   results, we need to manage random number generation
#   carefully. We cannot export the current state
#   of a random number generator using np.random.default_rng.
#   Therefore, to compare results of the "externally stopped"
#   repl and the "internally stopped" rep for debugging,
#   we need both replications to resume simulating from
#   time 800 with the same random numbers. Therefore, we
#   set the rng attribute to a random number generator object
#   starting at the same place, using seed 10000.
# The choice of seed numbers is arbitrary here, of course.
#   The point is that we start random number generation
#   from the same pointers.

###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example C: Running the simulation with the CDC staged-alert system
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is very similar to how we run the usual code, we just need to define the staged-alert policy with the
# CDC system method. The system has three different indicators; Case counts, Hospital admissions and  Percent hospital
# beds. Depending on the case count threshold the other two indicators take different values. I define them as
# "non_surge" and "surge" but we can change those later if we want to do more general systems.

case_threshold = 200
hosp_adm_thresholds = {"non_surge": (-1, -1, 10, 20, 20), "surge": (-1, -1, -1, 10, 10)}
staffed_thresholds = {"non_surge": (-1, -1, 0.1, 0.15, 0.15), "surge": (-1, -1, -1, 0.1, 0.1)}

# CDC threshold uses 7-day sum of hospital admission per 100k. The equivalent values if we were to use 7-day avg.
# hospital admission instead are as follows. We use equivalent thresholds to plot and evaluate the results in our
# indicator. I used the same CDC thresholds all the time but if we decide to optimize CDC threshold, we can calculate
# the equivalent values in the model and save to the policy.json.
equivalent_thresholds = {"non_surge": (-1, -1, 28.57, 57.14, 57.14), "surge": (-1, -1, -1, 28.57, 28.57)}
ctp = CDCTierPolicy(austin, tiers, case_threshold, hosp_adm_thresholds, staffed_thresholds)

rep = SimReplication(austin, vaccines, ctp, -1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example D: Parameter fitting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ToDo: I am not very happy with the parameter tuning file and I'll update it.
# But here is an example of parameter fitting:


