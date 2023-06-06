import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pickle

from preset_bionetworks import VALID_BIONAMES
from utils_io import pickle_load
from utils_networkx import check_tree_isomorphism, draw_from_adjacency, check_tree_isomorphism_with_insect


NAMES_IMPORTANT_GRAPHS = ['background', 'onecell'] + VALID_BIONAMES
NAMES_IMPORTANT_GRAPHS_ID = {idx: v for idx, v in enumerate(NAMES_IMPORTANT_GRAPHS)}
NUMBER_IMPORTANT_GRAPHS = len(NAMES_IMPORTANT_GRAPHS)


def handler_param_diffusion_arg(param_values):
    """
    For the case that param_name == 'diffusion_arg', and diffusion_arg is an array-like (not scalar)
    - check that we only vary along one axis
    - convert params_values to scalar by slicing along that axis
    """
    # (a) find along which dimension we are modifying the diffusion rate
    res = np.argwhere((param_values[1] - param_values[0]) != 0)
    assert len(res) == 1
    idx_diffusion_sweep = res[0][0]
    # (b) convert to array of scalars from array vectors
    param_values = param_values[:, idx_diffusion_sweep]
    return param_values, idx_diffusion_sweep


def get_isomorphism_int(A_run):
    if A_run.shape[0] == 1:
        variety_integer = 1
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[2]):
        variety_integer = 2
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[3]):
        variety_integer = 3
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[4]):
        variety_integer = 4
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[5]):
        variety_integer = 5
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[6]):
        variety_integer = 6
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[7]):
        variety_integer = 7
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[8]):
        variety_integer = 8
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[9]):
        variety_integer = 9
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[10]):
        variety_integer = 10
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[11]):
        variety_integer = 11
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[12]):
        variety_integer = 12
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[13]):
        variety_integer = 13
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[14]):
        variety_integer = 14
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[15]):
        variety_integer = 15
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[16]):
        variety_integer = 16
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[17]):
        variety_integer = 17
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[18]):
        variety_integer = 18
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[19]):
        variety_integer = 19
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[20]):
        variety_integer = 20
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[21]):
        variety_integer = 21
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[22]):
        variety_integer = 22
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[23]):
        variety_integer = 23
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[24]):
        variety_integer = 24
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[25]):
        variety_integer = 25
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[26]):
        variety_integer = 26
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[27]):
        variety_integer = 27
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[28]):
        variety_integer = 28
    elif check_tree_isomorphism_with_insect(A_run, NAMES_IMPORTANT_GRAPHS_ID[29]):
        variety_integer = 29
    else:
        variety_integer = 0
    return variety_integer


def wrapper_load_or_digest_sweep(dir_sweep):
    """
    Function overview:
        0) Assert that sweep.pkl is in dir_digest and load it
        1) Check if dir_digest has expected digest files listed below. If so load, validate, and return them.
        2) Otherwise, need to perform the sweep digest (slow). Save the digest files to dir_digest.

    Expected folder layouts for dir_digest:
        A) sweep.pkl ---> need to perform digest (slow)
        B) sweep.pkl
           unique_networks_dict.pkl ---> load and return as unique_networks_dict
           results_of_theta.npz     ---> load and return as results_dict
    """
    fpath_sweep_pkl = dir_sweep + os.sep + 'sweep.pkl'
    fpath_uniques_pkl = dir_sweep + os.sep + 'unique_networks_dict.pkl'
    fpath_results_npz = dir_sweep + os.sep + 'results_of_theta.npz'

    # Step 0)
    sweep = pickle_load(fpath_sweep_pkl)
    # hotswap directory attribute in case sweep folder was archived
    try:
        attr_dir_sweep = sweep.dir_sweep
    except AttributeError:
        attr_dir_sweep = sweep.sweep_dir  # old alias
        sweep.dir_sweep = attr_dir_sweep
    if attr_dir_sweep != dir_sweep:
        print('temporarily resetting sweep attribute "dir_sweep" from %s to %s' % (attr_dir_sweep, dir_sweep))
        sweep.dir_sweep = dir_sweep
        sweep.sweep_dir = dir_sweep  # this is old alias for dir_sweep attr

    # Step 1)
    if os.path.exists(fpath_uniques_pkl) and os.path.exists(fpath_results_npz):
        print('Sweep digest files already exist. Will load and return them...')
        # 1 (a)
        unique_networks_dict = pickle_load(fpath_uniques_pkl)
        # 1 (b)
        npz_loaded = np.load(fpath_results_npz)
        results_of_theta = dict()
        results_of_theta['num_cells'] = npz_loaded['num_cells']
        results_of_theta['unique_network_id'] = npz_loaded['unique_network_id']
        results_of_theta['isos'] = npz_loaded['isos']
        results_of_theta['exitstatus'] = npz_loaded['exitstatus']
        results_of_theta['end_time'] = npz_loaded['end_time']
        # validate file structure against sweep attributes
        assert sweep.sizes == list(results_of_theta['num_cells'].shape)
    # Step 2)
    else:
        print('Sweep digest files do not exist; perform digest to generate them...')
        unique_networks_dict, results_of_theta = digest_sweep(sweep, dir_save=dir_sweep)

    # return sweep object and the digest data (which was re-loaded or freshly generated and saved)
    return sweep, unique_networks_dict, results_of_theta


def digest_sweep(sweep, dir_save=None):
    """
    Input: a sweep_cellgraph object on which a sweep has been performed (i.e. results_dict attr populated)
    Returns:
        - unique_networks_dict -- dict of unique graphs
        - results_of_theta     -- dict, stores np.arrays of size sweep_cellgraph.sizes (e.g. 3d array size 40x40x61)
            - num_cells         -> (np.arr - int  - size sweep.sizes)
            - unique_network_id -> (np.arr - int  - size sweep.sizes)
            - isos              -> (np.arr - int  - size sweep.sizes)
            - exitstatus        -> (np.arr - bool - size sweep.sizes) -- did run not complete due to e.g. iso check failure?
            - end_time          -> (np.arr - float - size sweep.sizes)

    Structure of unique_networks_dict: dict
        key: num_cells
        value: a dict of
            key: A_id, e.g. "7_v0" means 7 cell graph. variant 0 (enumerating observed non-isomorphic 7 cell graphs)
            value: {'adjacency': np array,
                    'runs': list of run indices where this variant was observed,
                    'iso': integer defining the variant id (0, 1, 2 etc) observed for graphs of size num_cells
                    'unique_int': this is the integer that goes into isos_of_theta, taking into account all graphs of different sizes observed so far
                    'bio_int': integer label associated with different known structures e.g. drosophila}
    """
    assert len(NAMES_IMPORTANT_GRAPHS) == 30

    print("Visualizing data for sweep label:", sweep.sweep_label)
    sweep.printer()
    results = sweep.results_dict

    # these objects will be filled in below -- see docstring
    unique_networks_dict = {}
    num_cells_of_theta = np.zeros(sweep.sizes, dtype=int)
    unique_network_id_of_theta = np.zeros(sweep.sizes, dtype=int)
    isos_of_theta = np.zeros(sweep.sizes, dtype=int)
    exitstatus_of_theta = np.zeros(sweep.sizes, dtype=bool)
    end_time_of_theta = np.zeros(sweep.sizes, dtype=float)

    # loop over all runs to construct unique_networks_dict and arrays from docstring
    unique_int = 0
    run_int = 0
    total_runs = np.prod(sweep.sizes)
    for run_id_list in np.ndindex(*sweep.sizes):

        A_run = results[run_id_list]['adjacency']
        num_cells = results[run_id_list]['num_cells']
        end_time = results[run_id_list]['end_time']

        # Print out progress
        if run_int % 100 == 0:
            print('progress: (%.1f percent)' % (100 * run_int / total_runs), run_int, run_id_list, '...')

        # exit status entry in results dict is new; handler for old approach
        try:
            exitstatus = results[run_id_list]['bool_completion']
        except KeyError:
            # True means the run completed; False means early exit
            exitstatus = np.abs(results[run_id_list]['end_time'] - sweep.base_cellgraph.t1) <= 1e-8

        # Uniqueness case A - have we seen a graph of this size yet? if no it's unique
        if num_cells not in unique_networks_dict.keys():
            network_id = 'M%s_v0' % num_cells
            bio_int = get_isomorphism_int(A_run)
            unique_networks_dict[num_cells] = {
                network_id:
                     {'adjacency': A_run,
                      'runs': [run_id_list],
                      'iso': 0,
                      'unique_int': unique_int,
                      'bio_int': bio_int}
            }
            unique_int += 1
        else:
            # Uniqueness case B
            # Need now to compare the adjacency matrix to those observed before, to see if it is unique
            # - necessary condition for uniqueness is unique Degree matrix, D -- if D is unique, then A is unique
            is_iso = False
            for k, v in unique_networks_dict[num_cells].items():
                network_id = k
                is_iso, iso_swaps = check_tree_isomorphism(A_run, v['adjacency'])
                # Uniqueness case B1 - if is_iso, then we just add the run to the appropriate entry of the dictionary
                if is_iso:
                    unique_networks_dict[num_cells][k]['runs'] += [run_id_list]
                    break

            # Uniqueness case B2 - this is a new graph of size num_cells -- add new entry to the dictionary
            if not is_iso:
                nunique = len(unique_networks_dict[num_cells].keys())
                network_id = 'M%s_v%d' % (num_cells, nunique)
                bio_int = get_isomorphism_int(A_run)
                unique_networks_dict[num_cells][network_id] = \
                    {'adjacency': A_run,
                     'runs': [run_id_list],
                     'iso': nunique,
                     'unique_int': unique_int,
                     'bio_int': bio_int}
                unique_int += 1

        # Fill in primary output arrays (see docstring)
        num_cells_of_theta[run_id_list] = num_cells
        unique_network_id_of_theta[run_id_list] = unique_networks_dict[num_cells][network_id]['unique_int']
        isos_of_theta[run_id_list] = unique_networks_dict[num_cells][network_id]['bio_int']
        exitstatus_of_theta[run_id_list] = exitstatus
        end_time_of_theta[run_id_list] = end_time

        run_int += 1

    results_of_theta = {
        'num_cells': num_cells_of_theta,
        'unique_network_id': unique_network_id_of_theta,
        'isos': isos_of_theta,
        'exitstatus': exitstatus_of_theta,
        'end_time': end_time_of_theta,
    }

    if dir_save is not None:
        print('Digest complete. Saving results to file...')
        # 1) save unique_networks_dict as .pkl
        fpath_uniques_pkl = dir_save + os.sep + 'unique_networks_dict.pkl'
        with open(fpath_uniques_pkl, 'wb') as pickle_file:
            pickle.dump(unique_networks_dict, pickle_file)
        # 2) save results_of_theta in single .npz
        fpath_results_npz = dir_save + os.sep + 'results_of_theta.npz'
        np.savez_compressed(fpath_results_npz,
                            num_cells=num_cells_of_theta,
                            unique_network_id=unique_network_id_of_theta,
                            isos=isos_of_theta,
                            exitstatus=exitstatus_of_theta,
                            end_time=end_time_of_theta)

    return unique_networks_dict, results_of_theta


def draw_uniques(unique_networks_dict, outdir):
    """
    Draws all unique adjacency matrices to outdir
    """
    for num_cells in sorted(unique_networks_dict.keys()):
        for iso in sorted(unique_networks_dict[num_cells].keys()):
            label1_str = iso
            label2_bio_int = unique_networks_dict[num_cells][iso]['bio_int']
            label3_bio_str = NAMES_IMPORTANT_GRAPHS_ID[label2_bio_int]
            A_run = unique_networks_dict[num_cells][iso]['adjacency']
            num_runs = len(unique_networks_dict[num_cells][iso]['runs'])

            degree = np.diag(np.sum(A_run, axis=1))
            title = 'id=%s_bio=%s_nruns=%d' % (label1_str, label3_bio_str, num_runs)
            fpath = outdir + os.sep + 'sweepgraphs_%s' % title
            draw_from_adjacency(A_run, node_color=np.diag(degree), labels=None, cmap='Pastel1',
                                title=title, spring_seed=None, fpath=fpath, ftype='.jpg')
    return None


def visualize_sweep_data_1D(sweep, unique_networks_dict, arr_num_cells, arr_unique_network_id, arr_isos, outdir):
    """
    Args:
        sweep - instance of SweepCellGraph
        unique_networks_dict, arr_num_cells, arr_unique_network_id, arr_isos -- come directly from digest_sweep()
    """
    assert len(arr_num_cells.shape) == 1
    assert arr_num_cells.shape == arr_unique_network_id.shape
    assert arr_num_cells.shape == arr_isos.shape
    # Note: the below assert can be relaxed once we generalize this plot line, i.e. the i[0] part:
    #   specific_runs = [i[0] for i in unique_networks_dict[num_cells][iso]['runs']]
    assert sweep.k_vary == 1

    param_values = sweep.params_values[0]
    param_name = sweep.params_name[0]
    param_variety = sweep.params_variety[0]

    # Part 0) In case param_values are vectors corresponding to diffusion_arg triplets
    if not np.isscalar(param_values[0]) and param_name == 'diffusion_arg':
        param_values, idx_diffusion_sweep = handler_param_diffusion_arg(param_values)

    # Part 1) Create fancier M(theta) plot using unique graphs as diff colors
    kwargs = dict(
        markersize=6,
        markeredgecolor='k',
        markeredgewidth=0.4,
    )
    plt.figure(figsize=(4, 4), dpi=600)  # (4,4) used for 100 diffusion 1D
    plt.plot(param_values, arr_num_cells, '--o', **kwargs)

    # Part 1B) plot separate points on top for the special runs with graph isomorphisms
    for num_cells in sorted(unique_networks_dict.keys()):
        for iso in sorted(unique_networks_dict[num_cells].keys()):
            specific_runs = [i[0] for i in unique_networks_dict[num_cells][iso]['runs']]
            plt.plot(param_values[specific_runs], arr_num_cells[specific_runs], 'o', **kwargs)
            plt.plot()
    plt.title(r'$M(\theta)$ for $\theta=$%s' % param_name)
    plt.ylabel('num_cells')
    plt.xlabel('%s' % param_name)

    fpath = outdir + os.sep + 'num_cells_1d_vary_%s.png' % (param_name)
    plt.savefig(fpath)
    plt.show()
    # save numcells vs pvary data to file also
    arr_out = np.array([param_values, arr_num_cells]).T
    np.savetxt(outdir + os.sep + 'num_cells_1d_vary_%s.csv' % (param_name), arr_out, delimiter=',', fmt='%.5f')

    return None


if __name__ == '__main__':

    # load selected sweep
    dir_sweep_selected = 'runs' + os.sep + '1d_dzstretch_custom'  # point to a 1D sweep generated using run_sweep.py
    fpath_sweep_pkl = dir_sweep_selected + os.sep + 'sweep.pkl'
    assert os.path.exists(fpath_sweep_pkl)

    # analyze the sweep results
    sweep_cellgraph, unique_networks_dict, results_of_theta = wrapper_load_or_digest_sweep(dir_sweep_selected)

    arr_num_cells = results_of_theta['num_cells']
    arr_unique_network_id = results_of_theta['unique_network_id']
    arr_isos = results_of_theta['isos']
    arr_exitstatus = results_of_theta['exitstatus']
    arr_end_time = results_of_theta['end_time']

    # visualize the extracted arrays
    outdir = "output" + os.sep + "vis_sweep"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    assert sweep_cellgraph.k_vary == 1  # sweeps with k_vary == 2 or 3 are supported in notebooks/visualize_sweep.ipynb
    visualize_sweep_data_1D(sweep_cellgraph, unique_networks_dict, arr_num_cells, arr_unique_network_id, arr_isos, outdir)
    draw_uniques(unique_networks_dict, outdir)
