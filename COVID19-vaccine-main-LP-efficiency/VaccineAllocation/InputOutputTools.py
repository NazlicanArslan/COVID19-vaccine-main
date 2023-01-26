###############################################################################

# InputOutputTools.py
# Use this module to import and export simulation state data
#   so that a simulation can be loaded on another computer
#   and run from a previous starting point t > 0.
# Use this module to import simulation state data
#   from "realistic sample paths" that match historical data
#   with an R-squared > 0.75, to simulate policies starting
#   from the end of the historical data period onwards.

# Note that importing / exporting does not work for partial days
#   (discrete steps within a day), so we assume days are fully completed.

#   The code uses numpy.random.default_rng.
# (Is the following part stil true?) There is currently no way to export or load the state of a random
#   number generator to or from .json files. However, within a single
#   Python session on the same computer, the state of a random number generator
#   can be saved and started from the last saved point.


# Linda Pei 2022

###############################################################################

# Imports
import json
import numpy as np
import copy
import datetime as dt
###############################################################################

# Tuples containing names of attributes of SimReplication objects,
#   VaccineGroup objects, and MultiTierPolicy objects that must be
#   imported/exported to save/load a simulation replication correctly.

# The relevant instance of EpiSetup has an attribute
#   "random_params_dict" that is a dictionary of the randomly sampled
#   random variables -- this information is stored on that instance
#   rather than in this module. This is because the user specifies
#   which random variables

# List of names of SimReplication attributes to be serialized as a .json file
#   for saving a simulation replication or loading a simulation replication
#   from a timepoint t > 0 (rather than starting over from scratch)
SimReplication_IO_var_names = (
    "rng_seed",
    "ICU_history",
    "IH_history",
    "ToIHT_history",
    "ToIY_history",
    "S_history",
    "D_history",
    "ToICUD_history",
    "ToIYD_history",
    "ToRS_history",
    "ToSS_history",
    "next_t",
    "S",
    "E",
    "IA",
    "IY",
    "PA",
    "PY",
    "R",
    "D",
    "IH",
    "ICU",
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

# List of names of SimReplication attributes that are lists of arrays
SimReplication_IO_list_of_arrays_var_names = (
    "ICU_history",
    "S_history",
    "IH_history",
    "ToIHT_history",
    "ToIY_history",
    "D_history",
    "ToICUD_history",
    "ToIYD_history",
    "ToRS_history",
    "ToSS_history",
)

# List of names of SimReplication attributes that are arrays
SimReplication_IO_arrays_var_names = (
    "S",
    "E",
    "IA",
    "IY",
    "PA",
    "PY",
    "R",
    "D",
    "IH",
    "ICU",
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

# List of names of VaccineGroup attributes to be serialized as a .json file
VaccineGroup_IO_var_names = (
                                "v_beta_reduct",
                                "v_tau_reduct",
                                "v_pi_reduct"
                            ) + SimReplication_IO_arrays_var_names

# List of names of VaccineGroup attributes that are arrays
VaccineGroup_IO_arrays_var_names = SimReplication_IO_arrays_var_names

# List of names of MultiTierPolicy attributes to be serialized as a .json file
MultiTierPolicy_IO_var_names = (
    "community_transmission",
    "lockdown_thresholds",
    "case_threshold",
    "hosp_adm_thresholds",
    "staffed_bed_thresholds",
    "tier_history",
    "surge_history"
)

plot_var_names = ["ICU_history",
                  "ToIY_history",
                  "ToIHT_history",
                  "IH_history",
                  "D_history",
                  "ToICUD_history",
                  "ToIYD_history",
                  "ToRS_history",
                  "ToSS_history",
                  "S_history"]


###############################################################################


def import_rep_from_json(
        sim_rep,
        sim_rep_filename,
        vaccine_group_v0_filename,
        vaccine_group_v1_filename,
        vaccine_group_v2_filename,
        vaccine_group_v3_filename,
        multi_tier_policy_filename=None,
        random_params_filename=None,
):
    """
    Modifies a SimReplication object sim_rep in place to match the
        last state of a previously run simulation replication
    Updates sim_rep attributes according to the data in sim_rep_filename
    Updates vaccine group attributes for each instance of VaccineGroup in
        sim_rep.vaccine_groups according to the data in vaccine_group_v0_filename,
        vaccine_group_v1_filename, vaccine_group_v2_filename,
        and vaccine_group_v3_filename
    Updates sim_rep.policy attributes according to the data in
        multi_tier_policy_filename (this can be None, meaning there
        is no relevant policy data)
    Updates sim_rep.epi_rand according to the data in random_params_filename
        (this can be None, meaning that new parameters will be randomly sampled
        and these parameters are different from the ones that generated the
        loaded simulation replication)

    :param sim_rep: [SimReplication obj]
    :param sim_rep_filename: [str] .json file with entries corresponding to
        SimReplication_IO_var_names
    :param vaccine_group_v0_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_0
    :param vaccine_group_v1_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_1
    :param vaccine_group_v2_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_2
    :param vaccine_group_v3_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_3
    :param multi_tier_policy_filename: [str] .json file with entries
        corresponding to MultiTierPolicy_IO_var_names
    :param random_params_filename: [str] .json file with entries
        corresponding to sim_rep.epi_rand.random_params_dict,
        i.e. parameters that are randomly sampled at the beginning
        of the replication
    :return: [None]
    """

    # Update sim_rep variables
    d = json.load(open(sim_rep_filename))
    load_vars_from_dict(sim_rep, d, sim_rep.state_vars + sim_rep.tracking_vars)

    # Update vaccine group variables
    vaccine_group_filenames = [
        vaccine_group_v0_filename,
        vaccine_group_v1_filename,
        vaccine_group_v2_filename,
        vaccine_group_v3_filename,
    ]

    for i in range(len(vaccine_group_filenames)):
        vaccine_group = sim_rep.vaccine_groups[i]
        d = json.load(open(vaccine_group_filenames[i]))
        load_vars_from_dict(
            vaccine_group, d, sim_rep.state_vars + sim_rep.tracking_vars
        )

        # Modify the first step of the next day so that the
        #   discretization (with steps) of the next day is correct
        for attribute in sim_rep.state_vars:
            vars(vaccine_group)["_" + attribute][0] = getattr(vaccine_group, attribute)

    # (Optional) update policy variables
    if multi_tier_policy_filename is not None:
        d = json.load(open(multi_tier_policy_filename))
        load_vars_from_dict(sim_rep.policy, d)

    # (Optional) update epidemiological parameters
    if random_params_filename is not None:
        # Create a copy of the base epi parameters that do not change
        #   across simulation replications
        # Load randomly sampled epi parameters
        epi_rand = copy.deepcopy(sim_rep.instance.base_epi)
        d = json.load(open(random_params_filename))
        load_vars_from_dict(epi_rand, d, d.keys())

        # Update sim_rep.epi_rand accordingly
        # Update the dictionary storing randomly sampled parameters
        # Recompute key quantities that depend on the randomly
        #   sampled parameters
        # Modify sim_rep.epi_rand in place to reflected loaded changes
        epi_rand.random_params_dict = d
        epi_rand.setup_base_params()
        sim_rep.epi_rand = epi_rand


def load_vars_from_dict(simulation_object, loaded_dict, keys_to_convert_to_array=[]):
    """
    Helper function to assign attribute values to simulation_object according to
        loaded_dict. Modification occurs in-place.
    :param simulation_object: instance of SimulationRep, MultiTierPolicy, VaccineGroup,
        or EpiSetup
    :param loaded_dict: [dict] with data to unpack and assign to simulation_object
        attributes. Keys must be in SimReplication_IO_var_names,
        MultiTierPolicy_IO_var_names, VaccineGroup_IO_arrays_var_names,
        or simulation_object.epi_rand.random_params_dict if
        simulation_object is an instance of EpiSetup
    :param keys_to_convert_to_array: [list, optional] list of strings
        (subset of loaded_dict.keys()) with values to convert from lists to arrays
        when assigned to simulation_object.
    :return: [None]
    """
    for k in loaded_dict.keys():
        if k in keys_to_convert_to_array and isinstance(loaded_dict[k], list):
            setattr(simulation_object, k, np.array(loaded_dict[k]))
        else:
            setattr(simulation_object, k, loaded_dict[k])


def export_rep_to_json(
        sim_rep,
        sim_rep_filename,
        vaccine_group_v0_filename,
        vaccine_group_v1_filename,
        vaccine_group_v2_filename,
        vaccine_group_v3_filename,
        multi_tier_policy_filename=None,
        random_params_filename=None,
):
    """
    Does not modify any simulation objects. Exports key sim_rep attributes,
        key attributes of each vaccine group in sim_rep.vaccine_groups,
        key sim_rep.policy attributes (optional), and key sim_rep.epi_rand attributes
        (optional) to respective .json files, so that the current simulation
        replication state can be saved and started from the last save point.

    Function parameters are the same as function import_rep_from_json parameters.

    :param sim_rep: [SimReplication obj]
    :param sim_rep_filename: [str] .json file with entries corresponding to
        SimReplication_IO_var_names
    :param vaccine_group_v0_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_0
    :param vaccine_group_v1_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_1
    :param vaccine_group_v2_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_2
    :param vaccine_group_v3_filename: [str] .json file with entries
        corresponding to VaccineGroup_IO_var_names for vaccine group v_3
    :param multi_tier_policy_filename: [str] .json file with entries
        corresponding to MultiTierPolicy_IO_var_names
    :param random_params_filename: [str] .json file with entries
        corresponding to sim_rep.epi_rand.random_params_dict,
        i.e. parameters that are randomly sampled at the beginning
        of the replication
    :return: [None]
    """

    # Export sim_rep variables
    # Numpy arrays must be converted to lists to be serializable
    d = {}
    for k in SimReplication_IO_var_names:
        if k in SimReplication_IO_list_of_arrays_var_names:
            list_of_lists = [matrix.tolist() if type(matrix) == np.ndarray else matrix for matrix in
                             getattr(sim_rep, k)]
            d[k] = list_of_lists
        elif k in SimReplication_IO_arrays_var_names:
            d[k] = getattr(sim_rep, k).tolist()
        else:
            d[k] = getattr(sim_rep, k)
    json.dump(d, open(sim_rep_filename, "w"))

    # Export vaccine group variables
    vaccine_group_filenames = [
        vaccine_group_v0_filename,
        vaccine_group_v1_filename,
        vaccine_group_v2_filename,
        vaccine_group_v3_filename,
    ]

    for i in range(len(vaccine_group_filenames)):
        vaccine_group = sim_rep.vaccine_groups[i]
        d = {}
        for k in VaccineGroup_IO_var_names:
            if k in VaccineGroup_IO_arrays_var_names:
                d[k] = [matrix.tolist() for matrix in getattr(vaccine_group, k)]
            else:
                d[k] = getattr(vaccine_group, k)
        json.dump(d, open(vaccine_group_filenames[i], "w"))

    # Export sim_rep.policy variables
    if multi_tier_policy_filename is not None:
        d = {}
        for k in MultiTierPolicy_IO_var_names:
            if hasattr(sim_rep.policy, k):
                d[k] = getattr(sim_rep.policy, k)
        json.dump(d, open(multi_tier_policy_filename, "w"))

    # Export sim_rep.epi_rand variables
    if random_params_filename is not None:
        d = sim_rep.epi_rand.random_params_dict
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                d[k] = d[k].tolist()
        json.dump(d, open(random_params_filename, "w"))


def import_stoch_reps_for_reporting(seeds: list, num_reps: int, history_end_date: dt.datetime, instance: object):
    """
    Import simulation results for each sample paths and combine them in a list.
    The resulting outputs used for plotting and calculating key statistics over all sample paths.
    seeds and num_reps are used to define the filename where the simulation results are stored.
    :param history_end_date: the end date of the historical data.
    :param seeds: list of seeds used in simulation.
    :param num_reps: number of replication from each seeds.
    :param instance:
    :return: list of simulation outputs and policy data for all the sample paths.
    """
    sim_outputs = {}
    for var in SimReplication_IO_list_of_arrays_var_names:
        sim_outputs[var] = []
    policy_outputs = {}
    for i in seeds:
        for j in range(num_reps):
            filename = f"{instance.path_to_input_output}/{i}_{j + 1}_{history_end_date.date()}_sim_updated.json"
            with open(filename) as file:
                data = json.load(file)
                for var in plot_var_names:
                    if var in SimReplication_IO_list_of_arrays_var_names:
                        sim_outputs[var].append(data[var])
                    else:
                        print('The data is not outputted')
                        pass

            policy_filename = f"{instance.path_to_input_output}/{i}_{j + 1}_{history_end_date.date()}_policy.json"
            with open(policy_filename) as file:
                policy_data = json.load(file)
                for key, val in policy_data.items():
                    if key not in policy_outputs.keys():
                        policy_outputs[key] = [val]
                    else:
                        policy_outputs[key].append(val)

    return sim_outputs, policy_outputs
