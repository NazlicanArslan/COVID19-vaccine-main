###############################################################################

# OptTools.py
# This module contains Opt(imization) Tools, and includes functions
#   for generating realistic sample paths, enumerating candidate policies,
#   and optimization.
# This module is not used to run the SEIR model. This module contains
#   functions "on top" of the SEIR model.

# Each function in this module can run on a single processor,
#   and can be parallelized by passing a unique processor_rank to
#   each function call.

# In this code, "threshold" refers to a 5-tuple of the thresholds for a policy
#   and "policy" is an instance of MultiTierPolicy -- there's a distinction
#   between the identifier for an object versus the actual object.

# Linda Pei 2022

###############################################################################

import numpy as np

from SimObjects import MultiTierPolicy
from DataObjects import City, TierInfo, Vaccine
from SimModel import SimReplication
from InputOutputTools import import_rep_from_json
import copy
import itertools

###############################################################################

# An example of how to use multiprocessing on the cluster to
#   naively parallelize sample path generation
# import multiprocessing
# for i in range(multiprocessing.cpu_count()):
#     p = multiprocessing.Process(target=get_sample_paths, args=(i,))
#     p.start()


def get_sample_paths(
    city,
    vaccine_data,
    rsq_cutoff,
    goal_num_good_reps,
    processor_rank=0,
    seed_assignment_func=(lambda rank: rank),
    timepoints=(25, 100, 200, 400, 783),
):
    """
    This function uses an accept-reject procedure to
        "realistic" sample paths , using a "time blocks"
        heuristic (see Algorithm 1 in Yang et al. 2021) and using
        an R-squared type statistic based on historical hospital
        data (see pg. 10 in Yang et al. 2021).
    Realistic sample paths are exported as .json files so that
        they can be loaded in another session.
    One primary use of exporting realistic sample paths is that
        testing a policy (for example for optimization)
        only requires simulating the policy from the end
        of the historical time period. We can simulate any
        number of policies starting from the last timepoint of a
        pre-generated sample path, rather than starting from
        scratch at a timepoint t=0.
    Note that in the current version of the code, t=783
        is the end of the historical time period (and policies
        go in effect after this point).

    This function can be parallelized by passing a unique
        processor_rank to each function call.

    We use "sample path" and "replication" interchangeably here
        (Sample paths refer to instances of SimReplication).

    :param city: instance of City
    :param vaccine_data: instance of Vaccine
    :param rsq_cutoff: [float] non-negative number between [0,1]
        corresponding to the minimum R-squared value needed
        to "accept" a sample path as realistic
    :param goal_num_good_reps: [int] positive integer
        corresponding to number of "accepted" sample paths
        to generate
    :param processor_rank: [int] non-negative integer
        identifying the parallel processor
    :param seed_assignment_func: [func] optional function
        mapping processor_rank to the random number seed
        that instantiates the random number generator
        -- by default, the processor_rank is used
        as the random number seed
    :param timepoints: [tuple] optional tuple of
        any positive length that specifies timepoints
        at which to pause the simulation of a sample
        path and check the R-squared value
    :return: [None]
    """

    # Create an initial replication, using the random number seed
    #   specified by seed_assignment_func
    seed = seed_assignment_func(processor_rank)
    rep = SimReplication(city, vaccine_data, None, seed)

    # Instantiate variables
    num_good_reps = 0
    total_reps = 0

    # These variables are mostly for information-gathering
    # We track the number of sample paths eliminated at each
    #   user-specified timepoint
    # We also save the R-squared of every sample path generated,
    #   even those eliminated due to low R-squared values
    num_elim_per_stage = np.zeros(len(timepoints))
    all_rsq = []

    while num_good_reps < goal_num_good_reps:
        total_reps += 1
        valid = True

        # Use time block heuristic, simulating in increments
        #   and checking R-squared to eliminate bad
        #   sample paths early on
        for i in range(len(timepoints)):
            rep.simulate_time_period(timepoints[i])
            rsq = rep.compute_rsq()
            if rsq < rsq_cutoff:
                num_elim_per_stage[i] += 1
                valid = False
                all_rsq.append(rsq)
                break

        # If the sample path's R-squared is above rsq_cutoff
        #   at all timepoints, we accept it
        if valid == True:
            num_good_reps += 1
            all_rsq.append(rsq)
            identifier = str(processor_rank) + "_" + str(num_good_reps)
            export_rep_to_json(
                rep,
                identifier + "_sim.json",
                identifier + "_v0.json",
                identifier + "_v1.json",
                identifier + "_v2.json",
                identifier + "_v3.json",
                None,
                identifier + "_epi_params.json",
            )

        # Internally save the state of the random number generator
        #   to hand to the next sample path
        next_rng = rep.rng

        rep = SimReplication(city, vaccine_data, None, None)
        rep.rng = next_rng

        # Use the handed-over RNG to sample random parameters
        #   for the sample path, and compute other initial parameter
        #   values that depend on these random parameters
        epi_rand = copy.deepcopy(rep.instance.base_epi)
        epi_rand.sample_random_params(rep.rng)
        epi_rand.setup_base_params()
        rep.epi_rand = epi_rand

        # Every 1000 reps, export the information-gathering variables as a .csv file
        if total_reps % 1000 == 0:
            np.savetxt(
                str(processor_rank) + "_num_elim_per_stage.csv",
                np.array(num_elim_per_stage),
                delimiter=",",
            )
            np.savetxt(
                str(processor_rank) + "_all_rsq.csv", np.array(all_rsq), delimiter=","
            )


###############################################################################


def thresholds_generator(stage2_info, stage3_info, stage4_info, stage5_info):
    """
    Creates a list of 5-tuples, where each 5-tuple has the form
        (-1, t2, t3, t4, t5) with 0 <= t2 <= t3 <= t4 <= t5 < inf.
    The possible values t2, t3, t4, and t5 can take come from
        the grid generated by stage2_info, stage3_info, stage4_info,
        and stage5_info respectively.
    Stage 1 threshold is always fixed to -1 (no social distancing).

    :param stage2_info: [3-tuple] with elements corresponding to
        start point, end point, and step size (all must be integers)
        for candidate values for stage 2
    :param stage3_info: same as above but for stage 3
    :param stage4_info: same as above but for stage 4
    :param stage5_info: same as above but for stage 5
    :return: [array] of 5-tuples
    """

    # Create an array (grid) of potential thresholds for each stage
    stage2_options = np.arange(stage2_info[0], stage2_info[1], stage2_info[2])
    stage3_options = np.arange(stage3_info[0], stage3_info[1], stage3_info[2])
    stage4_options = np.arange(stage4_info[0], stage4_info[1], stage4_info[2])
    stage5_options = np.arange(stage5_info[0], stage5_info[1], stage5_info[2])

    # Using Cartesian products, create a list of 5-tuple combos
    stage_options = [stage2_options, stage3_options, stage4_options, stage5_options]
    candidate_feasible_combos = []
    for combo in itertools.product(*stage_options):
        candidate_feasible_combos.append((-1,) + combo)

    # Eliminate 5-tuples that do not satisfy monotonicity constraint
    # However, ties in thresholds are allowed
    feasible_combos = []
    for combo in candidate_feasible_combos:
        if np.all(np.diff(combo) >= 0):
            feasible_combos.append(combo)

    return feasible_combos


###############################################################################


def evaluate_policies_on_sample_paths(
    city,
    tiers,
    vaccines,
    thresholds_array,
    end_time,
    RNG,
    num_reps,
    base_filename,
    processor_rank,
    processor_count_total,
):
    """
    Creates a MultiTierPolicy object for each threshold in
        thresholds_array, partitions these policies amongst
        processor_count_total processors, simulates these
        policies starting from pre-saved sample paths up to
        time end_time, and exports the results.

    There must be num_reps sample paths saved and located
        in the same working directory as the main script calling
        this function.
    We assume each sample path has 6 .json files with the following
        filename format:
        base_json_filename + "sim.json"
        base_json_filename + "v0.json"
        base_json_filename + "v1.json"
        base_json_filename + "v2.json"
        base_json_filename + "v3.json"
        base_json_filename + "epi_params.json"
    See module InputOutputTools for fields of each type of .json file.

    Results are saved for each replication in 3 .csv files with the
        following filename suffixes:
        [1] "thresholds_identifiers.csv"
        [2] "costs_data.csv"
        [3] "feasibility_data.csv"
    If processor_rank was assigned num_policies policies to simulate,
        then the above .csv files each contain num_policies entries.
    For each replication, .csv file type [1]'s ith entry is the
        (unique) threshold identifier of the ith policy that processor_rank
        simulated. File type [2]'s ith entry is the realized cost
        of the ith policy that processor_rank simulated on that replication.
        File type [3]'s ith entry is whether the ith policy that processor_rank
        simulated is feasible on that replication.

    This function can be parallelized by passing a unique
        processor_rank to each function call.

    :param city: [obj] instance of City
    :param tiers: [obj] instance of TierInfo
    :param vaccines: [obj] instance of Vaccine
    :param thresholds_array: [list of tuples] arbitrary-length list of
        5-tuples, where each 5-tuple has the form (-1, t2, t3, t4, t5)
         with 0 <= t2 <= t3 <= t4 <= t5 < inf, corresponding to
         thresholds for each tier.
    :param end_time: [int] nonnegative integer, time at which to stop
        simulating and evaluating each policy -- must be greater (later than)
        the time at which the sample paths stopped
    :param RNG: [obj] instance of np.random.RandomState(),
        a random number generator
    :param num_reps: [int] number of sample paths to test policies on
    :param base_filename: [str] prefix common to all filenames
    :param processor_rank: [int] nonnegative unique identifier of
        the parallel processor
    :param processor_count_total: [int] total number of processors
    :return: [None]
    """

    # Create an array of MultiTierPolicy objects, one for each threshold
    policies_array = np.array(
        [
            MultiTierPolicy(city, tiers, thresholds, "green")
            for thresholds in thresholds_array
        ]
    )

    # Assign each processor its own set of MultiTierPolicy objects to simulate
    # Some processors have min_num_policies_per_processor
    # Others have min_num_policies_per_processor + 1
    num_policies = len(policies_array)
    min_num_policies_per_processor = int(np.floor(num_policies / processor_count_total))
    leftover_num_policies = num_policies % processor_count_total

    if processor_rank in np.arange(leftover_num_policies):
        start_point = processor_rank * (min_num_policies_per_processor + 1)
        policies_ix_processor = np.arange(
            start_point, start_point + (min_num_policies_per_processor + 1)
        )
    else:
        start_point = (min_num_policies_per_processor + 1) * leftover_num_policies + (
            processor_rank - leftover_num_policies
        ) * min_num_policies_per_processor
        policies_ix_processor = np.arange(
            start_point, start_point + min_num_policies_per_processor
        )

    # Iterate through each replication
    for rep in range(num_reps):

        # Load the sample path from .json files for each replication
        base_json_filename = base_filename + str(rep + 1) + "_"
        base_rep = SimReplication(city, vaccines, None, 1)
        import_rep_from_json(
            base_rep,
            base_json_filename + "sim.json",
            base_json_filename + "v0.json",
            base_json_filename + "v1.json",
            base_json_filename + "v2.json",
            base_json_filename + "v3.json",
            None,
            base_json_filename + "epi_params.json",
        )
        if rep == 0:
            base_rep.rng = RNG

        thresholds_identifiers = []
        costs_data = []
        feasibility_data = []

        # Iterate through each policy
        for policy in policies_array[policies_ix_processor]:
            base_rep.policy = policy
            base_rep.simulate_time_period(end_time)

            thresholds_identifiers.append(base_rep.policy.lockdown_thresholds)
            costs_data.append(base_rep.compute_cost())
            feasibility_data.append(base_rep.compute_feasibility())

            # Clear the policy and simulation replication history
            base_rep.policy.reset()
            base_rep.reset()

        # Save results
        base_csv_filename = "proc" + str(processor_rank) + "_rep" + str(rep + 1) + "_"
        np.savetxt(
            base_csv_filename + "thresholds_identifiers.csv",
            np.array(thresholds_identifiers),
            delimiter=",",
        )
        np.savetxt(
            base_csv_filename + "costs_data.csv", np.array(costs_data), delimiter=","
        )
        np.savetxt(
            base_csv_filename + "feasibility_data.csv",
            np.array(feasibility_data),
            delimiter=",",
        )
