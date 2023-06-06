import numpy as np
import os
import time

from class_cellgraph import CellGraph
from utils_io import run_subdir_setup
from preset_solver import PRESET_SOLVER
from preset_cellgraph import PRESET_CELLGRAPH
from settings import IS_RUNNING_ON_CLUSTER

"""
To run a cellgraph trajectory, need
1) the cellgraph kwargs (can get from preset_cellgraph.py)
2) the solver kwargs (can get from preset_solver.py) -- currently hard-coded to be used with solve_ivp() by default
3) an io_dict instance, which can be tacked on to cellgraph_kwargs before passing to create_cellgraph()
"""


def create_cellgraph(style_ode=None,
                     style_detection=None,
                     style_division=None,
                     style_diffusion=None,
                     num_cells=None,
                     diffusion_arg=None,
                     diffusion_rate=None,
                     alpha_sharing=None,
                     beta_sharing=None,
                     t0=None,
                     t1=None,
                     init_cond=None,
                     timeintervalrun=None,
                     state_history=None,
                     io_dict=False,
                     walltimes=None,
                     verbosity=0,
                     mods_params_ode={}):
    """
    Instantiates the CellGraph and modifies ODE params if needed
    """
    if diffusion_rate is not None:
        assert diffusion_arg is None
        diffusion_arg = diffusion_rate
    if beta_sharing is None and style_division == 'plus_minus_delta_ndiv_bam':
        assert alpha_sharing is not None
        beta_sharing = 0.5 * alpha_sharing

    cellgraph = CellGraph(
        num_cells=num_cells,
        style_ode=style_ode,
        style_detection=style_detection,
        style_division=style_division,
        style_diffusion=style_diffusion,
        state_history=state_history,
        diffusion_arg=diffusion_arg,
        alpha_sharing=alpha_sharing,
        beta_sharing=beta_sharing,
        t0=t0,
        t1=t1,
        init_cond=init_cond,
        timeintervalrun=timeintervalrun,
        io_dict=io_dict,
        walltimes=walltimes,
        verbosity=verbosity)

    # Post-processing (for deep attributes) as in class_cellgraph.py main() call
    for k, v in mods_params_ode.items():
        cellgraph.sc_template.params_ode[k] = v

    return cellgraph


def mod_cellgraph_ode_params(base_cellgraph, mods_params_ode):
    """
    Args:
        base_cellgraph    - an instance of CellGraph
        mods_params_ode   - dict of form {single cell params ode name: new_attribute_value}
    Returns:
        new CellGraph with all attributes same as base except those specified in attribute_mods
    Currently, unused in favor of simply recreating CellGraph each loop (this would be faster, just more bug risk)
    """
    for k, v in mods_params_ode.items():
        base_cellgraph.sc_template.params_ode[k] = v
    return base_cellgraph


if __name__ == '__main__':

    flag_preset = True
    subgraphIsoCheck = False  # if True - terminate loop when the graph can no longer reach a known structure
    flag_plotly_traj = False  # very slow currently; can be useful for debugging/trajectory inspection

    if flag_preset:

        cellgraph_preset_choice = 'drosophila_3quarter_pint'
        io_dict = run_subdir_setup(run_subfolder='cellgraph')
        solver_kwargs = PRESET_SOLVER['solve_ivp_radau_strict']['kwargs']
        solver_kwargs['vectorized'] = True

        cellgraph_preset = PRESET_CELLGRAPH[cellgraph_preset_choice]
        cellgraph_preset['io_dict'] = io_dict
        cellgraph_preset['verbosity'] = 1
        cellgraph = create_cellgraph(**cellgraph_preset)

    else:
        # High-level initialization & graph settings
        style_ode = 'PWL3_swap'                      # styles: ['PWL2', 'PWL3', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
        style_detection = 'manual_crossings_1d_mid'  # styles: ['ignore', 'scipy_peaks', 'manual_crossings_1d_mid', 'manual_crossings_1d_hl', 'manual_crossings_2d']
        style_division = 'plus_minus_delta_bam'      # styles: check style_division_valid in settings.py
        style_diffusion = 'xy'                       # styles: ['all', 'xy']
        M = 1
        alpha_sharing = 0.2
        beta_sharing = 0.0
        diffusion_arg = 0
        verbosity = 0  # in 0, 1, 2 (highest)

        # Main-loop-specific settings
        add_init_cells = 0

        # Initialization modifications for different cases
        if style_ode == 'PWL2':
            state_history = np.array([[100, 100]]).T     # None or array of shape (NM x times)
        elif style_ode == 'PWL3_swap':
            state_history = np.array([[0, 0, 0]]).T      # None or array of shape (NM x times)
        else:
            state_history = None

        # Specify time interval which is separate from solver kwargs (used in graph_trajectory explicitly)
        t0 = 0
        t1 = 200
        timeintervalrun = 1
        # Prepare io_dict
        io_dict = run_subdir_setup(run_subfolder='cellgraph')

        # Instantiate the graph and modify ode params if needed
        cellgraph = CellGraph(
            num_cells=M,
            style_ode=style_ode,
            style_detection=style_detection,
            style_division=style_division,
            style_diffusion=style_diffusion,
            state_history=state_history,
            alpha_sharing=alpha_sharing,
            beta_sharing=beta_sharing,
            diffusion_arg=diffusion_arg,
            t0=t0,
            t1=t1,
            timeintervalrun=timeintervalrun,
            io_dict=io_dict,
            verbosity=verbosity)
        if cellgraph.style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
            cellgraph.sc_template.params_ode['epsilon'] = 1e-2
            cellgraph.sc_template.params_ode['pulse_vel'] = 0.2

        # Add some cells through manual divisions (two different modes - linear or random) to augment initialization
        for idx in range(add_init_cells):
            dividing_idx = np.random.randint(0, cellgraph.num_cells)
            print("Division event (idx, div idx):", idx, dividing_idx)
            cellgraph = cellgraph.division_event(idx, 0)  # Mode 1 - linear division idx
            # Output plot & print
            #cellgraph.plot_graph()
            cellgraph.print_state()
            print()

        # Setup solver kwargs for the graph trajectory wrapper
        solver_kwargs = {}                   # assume passing to solve_ivp for now
        solver_kwargs['method'] = 'Radau'
        solver_kwargs['t_eval'] = None       # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
        solver_kwargs['max_step'] = np.Inf   # for testing, try 1e-1 or 1e-2
        solver_kwargs['atol'] = 1e-8
        solver_kwargs['rtol'] = 1e-4

    # Write initial CellGraph info to file
    cellgraph.print_state(msg='initialized in run_cellgraph.py main')
    cellgraph.write_metadata()
    cellgraph.write_state(fmod='init')
    if not IS_RUNNING_ON_CLUSTER:
        cellgraph.plot_graph(fmod='init')

    # From the initialized graph (after all divisions above), simulate graph trajectory
    print('\nExample trajectory for the graph...')
    start_time = time.time()
    event_detected, cellgraph = cellgraph.wrapper_graph_trajectory(subgraphIsoCheck=subgraphIsoCheck, **solver_kwargs)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n in main: num cells after wrapper trajectory =", cellgraph.num_cells)

    # Plot the timeseries for each cell
    cellgraph.plot_state_unified(arrange_vertical=True, fmod='final')
    if not IS_RUNNING_ON_CLUSTER:
        cellgraph.plot_graph(fmod='final')
        cellgraph.plot_graph(fmod='final', spring_seed=0, by_degree=True, by_ndiv=False, by_last_div=False, by_age=False)
        cellgraph.plot_graph(fmod='final', gviz_prog='dot', by_degree=True, by_ndiv=False, by_last_div=False, by_age=False)
        cellgraph.plot_graph(fmod='final', gviz_prog='circo', by_degree=True, by_ndiv=False, by_last_div=False, by_age=False)
        cellgraph.plot_graph(fmod='final', gviz_prog='twopi', by_degree=True, by_ndiv=False, by_last_div=False, by_age=False)
        cellgraph.plot_xyz_state_for_specific_cell(plot_cell_index=0, decorate=True)
        cellgraph.plot_xyz_state_for_specific_cell(plot_cell_index=1, decorate=True)
    if cellgraph.sc_dim_ode > 1:
        cellgraph.plot_xy_separate(fmod='final')
    if flag_plotly_traj:
        cellgraph.plotly_traj(fmod='final', show=True, write=True)
    # Plot walltimes between division events
    cellgraph.plot_walltimes(fmod='final')

    # Save class state as pickle object
    cellgraph.pickle_save('classdump.pkl')
