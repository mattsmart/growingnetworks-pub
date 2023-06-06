import datetime
import numpy as np
import os
import pickle
import time

from utils_io import run_subdir_setup
from settings import SWEEP_VARIETY_VALID, IS_RUNNING_ON_CLUSTER, DIR_RUNS
from run_cellgraph import create_cellgraph, mod_cellgraph_ode_params

"""
class SweepCellGraph structure

self.sweep_label     - name of the subdir within 'runs' dir which identifies the sweep
self.dir_sweep       - [inferred] directory in which all runs are stored as the params are varied
self.base_kwargs     - core kwargs for CllGraph generation which will be tweaked to create phase diagram
self.params_name     - ordered k-list of param names which are to be varied
self.params_values   - ordered k-list of array like for each param
self.params_variety  - ordered k-list with elements in ['meta_cellgraph', 'sc_ode']
self.k_vary          - [inferred] int k >= 1; number of parameters which are to be varied
self.sizes           - [inferred] k-list with elements len(params_values[j])
self.total_runs      - [inferred] int = prod_j sizes[j]) 
self.results_dict    - main output object; dict of the form 
    - k_tuple ----> output_dict dictionary (for a single cellgraph trajectory)
    - e.g. if there are two+ parameters being swept, it is: 
      (i, j, ...): output_dict (see below)

Important notes:
    - output_dict currently has the form:
        {'num_cells': int,
         'adjacency': num_cells x num_cells array,
         'division_events': array (d x 3)   - for each division event, append row: [mother_idx, daughter_idx, time_idx],
         'cell_stats':      array (M x 3)   - stores cell metadata: [n_div, time_idx_last_div, time_idx_birth]}
    - each run creates a directory uniquely named i_j_... in self.dir_sweep
"""


class SweepCellGraph():

    def __init__(self,
                 sweep_label=None,
                 base_cellgraph_kwargs=None,
                 params_name=None,
                 params_values=None,
                 params_variety=None,
                 solver_kwargs=None,
                 dir_root=DIR_RUNS):
        self.sweep_label = sweep_label
        self.base_kwargs = base_cellgraph_kwargs
        self.solver_kwargs = solver_kwargs
        self.params_name = params_name
        self.params_values = params_values
        self.params_variety = params_variety
        self.results_dict = {}  # see docstring for contents

        # asserts
        k = len(self.params_name)
        assert k == len(params_values)
        assert k == len(params_variety)
        assert all(a in SWEEP_VARIETY_VALID for a in params_variety)

        path_run_subfolder = dir_root + os.sep + self.sweep_label
        if os.path.exists(path_run_subfolder):
            print("Warning init SweepCellGraph -- path_run_subfolder exists")
            assert not os.path.exists(path_run_subfolder + os.sep + 'sweep.pkl')
        else:
            os.mkdir(path_run_subfolder)
        self.dir_sweep = path_run_subfolder
        self.dir_root = dir_root

        # create base cellgraph for sweep speedups
        self.base_cellgraph = create_cellgraph(**self.base_kwargs)

        # set inferred attributes
        self.k_vary = k
        self.sizes = [len(params_values[j]) for j in range(k)]
        self.total_runs = np.prod(self.sizes)
        return

    def basic_run(self, cellgraph, save_cellgraph_files=True, plotting=False,
                  subgraphIsoCheck=False,
                  trajectory_max_cells=None):
        """
        Given a cellgraph, performs some basic operations (e.g. a trajectory) and outputs a 'results' array

        Args:
            if save_cellgraph_files: save the results of each cellgraph run,
                                     (else, just collect their output results and only save to sweep.pkl at the end)
        Returns:
            output_results - 1D array of size k
        """
        assert cellgraph.hdf5_mode

        # Write initial CellGraph info to file
        if cellgraph.verbose:
            cellgraph.print_state(msg='call in SweepCellGraph.basic_run()')
        if save_cellgraph_files:
            flagWriteState = True
            cellgraph.write_metadata()
            if flagWriteState and not IS_RUNNING_ON_CLUSTER:
                cellgraph.write_state(fmod='init')
            if plotting:
                cellgraph.plot_graph(fmod='init')
        else:
            flagWriteState = False  # this disables writing to file (or into an .h5) during wrapper graph trajectory

        # From the initialized graph (after all divisions above), simulate graph trajectory
        if cellgraph.verbose:
            print('\nExample trajectory for the graph...')
        event_detected, cellgraph = cellgraph.wrapper_graph_trajectory(
            plotting=plotting, subgraphIsoCheck=subgraphIsoCheck, trajectory_max_cells=trajectory_max_cells,
            flagWriteState=flagWriteState, **self.solver_kwargs)
        if cellgraph.verbose:
            print("\n in main: num cells after wrapper trajectory =", cellgraph.num_cells)

        # Plot the timeseries for each cell
        if save_cellgraph_files and plotting:
            cellgraph.plot_graph(fmod='final')
            if cellgraph.sc_dim_ode > 1:
                cellgraph.plot_xy_separate(fmod='final')
            cellgraph.plotly_traj(fmod='final', show=False, write=True)
            cellgraph.plot_state_unified(arrange_vertical=True, fmod='final')

        # Save class state as pickle object
        if save_cellgraph_files:
            cellgraph.pickle_save('classdump.pkl')

        # Check if cellgraph run completed
        bool_completion = (cellgraph.times_history[-1] == cellgraph.t1)

        # HSN edited here --
        # in case of incomplete run due to subgraphIsoCheck,
        # adding end_time to show that simulation didn't run to completion.
        output_results = {
            'num_cells': cellgraph.num_cells,
            'adjacency': cellgraph.adjacency,
            'division_events': cellgraph.division_events,
            'cell_stats': cellgraph.cell_stats,
            'end_time': cellgraph.times_history[-1],
            'bool_completion': bool_completion,  # True means the run completed; False means early exit
        }
        return output_results

    def sweep(self, save_cellgraph_files=True, subgraphIsoCheck=False, trajectory_max_cells=None):
        """
        Args:
            if save_cellgraph_files: save the results of each cellgraph run,
                                     (else, just collect their output results and only save to sweep.pkl at the end)
            if subgraphIsoCheck:     end graph trajectory when it is not isomorphic to a subgraph of any target
            if trajectory_max_cells is not None: end graph trajectory early if it reaches <int> size

        Optimization for file structure on cluster:
             instead of laying out directories in one pile as subfolders    0_0, 0_1, 0_2, ... k1_k2
             nest them like so:     p3/p2/p1
            Note -- the convention of reversed order above is to play nice with merge_sweep.py copy/move options
        """
        time_start = time.time()

        def build_io_dict(run_id_list):
            if self.k_vary == 1:
                dir_minus_two = self.dir_root
                dir_minus_one = self.sweep_label
                dir_final = str(run_id_list[0])
            elif self.k_vary == 2:
                dir_minus_two = self.dir_root + os.sep + self.sweep_label
                dir_minus_one = str(run_id_list[1])
                dir_final = str(run_id_list[0])
            elif self.k_vary == 3:
                dir_minus_two = self.dir_root + os.sep + self.sweep_label + os.sep + str(run_id_list[2])
                dir_minus_one = str(run_id_list[1])
                dir_final = str(run_id_list[0])
            else:
                cap = (os.sep).join([str(i) for i in run_id_list[2:][::-1]])
                dir_minus_two = self.dir_root + os.sep + self.sweep_label + os.sep + cap
                dir_minus_one = str(run_id_list[1])
                dir_final = str(run_id_list[0])

            io_dict = run_subdir_setup(
                dir_runs=dir_minus_two,
                run_subfolder=dir_minus_one,
                timedir_override=dir_final, minimal_mode=True)
            return io_dict

        run_int = 0
        for run_id_list in np.ndindex(*self.sizes):
            if run_int % 100 == 0:
                print('progress: (%.1f percent)' % (100 * run_int / self.total_runs), run_int, run_id_list, '...')

            # run_id_list = self.convert_run_int_to_run_id_list(run_int)
            # 1) Prepare io_dict - this is unique to each run
            if save_cellgraph_files:
                io_dict = build_io_dict(run_id_list)
            else:
                io_dict = False  # we use io_dict = False as 'no file' mode in CellGraph class

            # 2) create modified form of the base_cellgraph -- approach depends on self.fast_mod_mode
            mods_params_ode = self.base_kwargs.get('mods_params_ode', {})
            modified_cellgraph_kwargs = self.base_kwargs.copy()
            modified_cellgraph_kwargs['io_dict'] = io_dict
            for j in range(self.k_vary):
                pname = self.params_name[j]
                pvariety = self.params_variety[j]
                pval = self.params_values[j][run_id_list[j]]
                if pvariety == 'sc_ode':
                    mods_params_ode[pname] = pval
                else:
                    assert pvariety == 'meta_cellgraph'
                    modified_cellgraph_kwargs[pname] = pval
            modified_cellgraph_kwargs['mods_params_ode'] = mods_params_ode
            modified_cellgraph = create_cellgraph(**modified_cellgraph_kwargs)

            # 3) Perform the run
            output_results = self.basic_run(modified_cellgraph,
                                            save_cellgraph_files=save_cellgraph_files,
                                            subgraphIsoCheck=subgraphIsoCheck,
                                            trajectory_max_cells=trajectory_max_cells)

            # 4) Extract relevant output
            self.results_dict[run_id_list] = output_results
            run_int += 1

        time_end = time.time()
        time_elapsed = time_end - time_start
        print('\nExecution time in d-H:M:S:', datetime.timedelta(seconds=time_elapsed), ", #_threads:", os.cpu_count())
        print('Sweep done. Saving pickle file...')
        self.pickle_save()
        return

    def printer(self):
        print('self.sweep_label -', self.sweep_label)
        try:
            print('self.dir_sweep -', self.dir_sweep)
        except AttributeError:
            print('self.sweep_dir -', self.sweep_dir)
        print('self.k_vary -', self.k_vary)
        print('self.sizes -', self.sizes)
        print('self.total_runs -', self.total_runs)
        print('Parameters in sweep:')
        for idx in range(self.k_vary):
            pname = self.params_name[idx]
            pvar = self.params_variety[idx]
            pv = self.params_values[idx]
            if np.isscalar(pv):
                print('\tname: %s, variety: %s, npts: %d, low: %.4f, high: %.4f' % (pname, pvar, len(pv), pv[0], pv[-1]))
            else:
                print('\tname: %s, variety: %s, npts: %d, low: %s, high: %s' % (pname, pvar, len(pv), pv[0], pv[-1]))

    def pickle_save(self, fname='sweep.pkl'):
        fpath = self.dir_sweep + os.sep + fname
        with open(fpath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        return
