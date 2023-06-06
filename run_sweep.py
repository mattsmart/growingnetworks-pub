import utils_multiprocessing  # must be imported before numpy
import argparse
import numpy as np

from class_sweep_cellgraph import SweepCellGraph
from preset_sweep import PRESET_SWEEP, SWEEP_SOLVER, diffusion_arg_sweep_helper
from settings import ALPHA_SHARING, BETA_SHARING, DIR_RUNS

if __name__ == '__main__':
    flag_preset = True  # use preset defined in SWEEP_PRESETS, or prepare own custom sweep runave_ce

    if flag_preset:
        # see https://docs.python.org/3/howto/argparse.html for how to submit these "a" and "b" and "t" arguments from command line
        # alternatively you could just change the default arguments in this function and submit the run without "a" or "t" inputs
        parser = argparse.ArgumentParser()
        parser.add_argument('-a', '--alpha_sharing', type=float, default=ALPHA_SHARING,
                            help='the degree of sharing between mother and daughter (history-independent)')
        parser.add_argument('-b', '--beta_sharing', type=float, default=BETA_SHARING,
                            help='the degree of sharing between mother and daughter (history-dependent)')
        parser.add_argument('-t', '--t_pulse_switch', type=float, default=175.,
                            help='time at which z(t) production switches off')
        parser.add_argument('-d', '--dir_root', type=str, default=DIR_RUNS,
                            help='directory to store sweep (default: DIR_RUNS)')
        args = parser.parse_args()

        # only necessary arg from argparse
        dir_root_param = args.dir_root

        # 0) choose a preset
        # - testing presets (small)
        preset_choice = '3d_alpha_beta_t_pulse_testing'
        # - larger presets
        sweep_preset = PRESET_SWEEP[preset_choice]
        sweep_style_ode = sweep_preset['base_cellgraph_kwargs']['style_ode']

        # 1) fill in kwargs from argparse
        # arg 1 - alpha_sharing
        if 'alpha_sharing' in sweep_preset['params_name']:
            pos = sweep_preset['params_name'].index('alpha_sharing')
            alpha_sharing_param = sweep_preset['params_values'][pos][0]  # take first element in linspace
            alpha_sharing_str = 'Vary'
        else:
            alpha_sharing_param = args.alpha_sharing
            alpha_sharing_str = str(alpha_sharing_param)
        sweep_preset['base_cellgraph_kwargs']['alpha_sharing'] = alpha_sharing_param

        # arg 2 - beta_sharing - care that it can be used
        if 'beta_sharing' in sweep_preset['params_name']:
            pos = sweep_preset['params_name'].index('beta_sharing')
            beta_sharing_param = sweep_preset['params_values'][pos][0]  # take first element in linspace
            beta_sharing_str = 'Vary'
        else:
            beta_sharing_param = args.beta_sharing
            beta_sharing_str = str(beta_sharing_param)
        sweep_preset['base_cellgraph_kwargs']['beta_sharing'] = beta_sharing_param

        # arg 3 - time at which z(t) production switches - care for cases
        if 't_pulse_switch' in sweep_preset['params_name']:
            pos = sweep_preset['params_name'].index('t_pulse_switch')
            t_pulse_switch_param = sweep_preset['params_values'][pos][-1]  # take last (largest) element in linspace
            t_pulse_switch_str = 'Vary'
            # care: "t1" - integration time of graph - is determined by t_pulse_switch
            print('Warning: doing a sweep over t_pulse_switch can be inefficient, as t1 = 2.5 * max(t_pulse linspace)')
        else:
            t_pulse_switch_param = args.t_pulse_switch
            sweep_preset['base_cellgraph_kwargs']['t_pulse_switch'] = t_pulse_switch_param
            t_pulse_switch_str = str(t_pulse_switch_param)
        sweep_preset['base_cellgraph_kwargs']['mods_params_ode']['t_pulse_switch'] = t_pulse_switch_param

        assert sweep_style_ode in ['PWL3_swap', 'PWL3_zstepdecay']
        if sweep_style_ode == 'PWL3_swap':
            # care: "t1" - integration time of graph - is determined by t_pulse_switch
            sweep_preset['base_cellgraph_kwargs']['t1'] = 2.5 * t_pulse_switch_param
        else:
            sweep_preset['base_cellgraph_kwargs']['mods_params_ode']['dz_stretch'] = t_pulse_switch_param
            assert 2 * t_pulse_switch_param <= sweep_preset['base_cellgraph_kwargs']['t1']

        # 2) extra settings to enforce
        sweep_preset['base_cellgraph_kwargs']['timeintervalrun'] = 1.0

        # 3) generate label suffix for the sweep
        this_sweep_label = sweep_preset['sweep_label']
        this_sweep_label = this_sweep_label + '_alpha_sharing_' + alpha_sharing_str + '_beta_sharing_' + beta_sharing_str + '_t_pulse_' + t_pulse_switch_str
        sweep_preset['sweep_label'] = this_sweep_label

        # 4) final checks
        assert sweep_preset['solver_kwargs']['atol'] == 1e-8
        assert sweep_preset['solver_kwargs']['rtol'] == 1e-4

        # 5) specify root dir for the sweep
        sweep_preset['dir_root'] = dir_root_param

        sweep_cellgraph = SweepCellGraph(**sweep_preset)

    else:
        sweep_label = '1d_dzstretch_custom'
        params_name = ['dz_stretch']
        params_variety = ['sc_ode']  # must be in ['meta_cellgraph', 'sc_ode']
        params_values = [
            np.linspace(5.0, 148.5, 10)  # (5.0, 8.5, 10)
        ]

        # Initialize the base CellGraph which will be varied during the sweep
        # A) High-level initialization & graph settings
        style_ode = 'PWL3_zstepdecay'                  # styles: ['PWL3_zstepdecay', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
        style_detection = 'manual_crossings_1d_mid'    # styles: ['ignore', 'scipy_peaks', 'manual_crossings_1d_mid', 'manual_crossings_1d_hl']
        style_division = 'plus_minus_delta_ndiv_bam'   # see division styles in settings.py
        style_diffusion = 'xy'                         # styles: ['all', 'xy']
        num_cells = 1
        alpha_sharing = 0.02
        beta_sharing = 0.0
        diffusion_arg = 0.0

        # B) Initialization modifications for different cases
        if style_ode == 'PWL2':
            state_history = np.array([[100, 100]]).T     # None or array of shape (NM x times)
        elif style_ode == 'PWL3_swap':
            state_history = np.array([[0, 0, 0]]).T  # None or array of shape (NM x times)
        else:
            state_history = None

        # C) Specify time interval which is separate from solver kwargs (used in graph_trajectory explicitly)
        t0 = 0                 # None or float
        t1 = 450               # None or float
        timeintervalrun = 1    # None or float

        # D) Setup solver kwargs for the graph trajectory wrapper
        solver_kwargs = SWEEP_SOLVER
        base_kwargs = dict(
            style_ode=style_ode,
            style_detection=style_detection,
            style_division=style_division,
            style_diffusion=style_diffusion,
            num_cells=num_cells,
            alpha_sharing=alpha_sharing,
            beta_sharing=beta_sharing,
            diffusion_arg=diffusion_arg,
            t0=t0,
            t1=t1,
            timeintervalrun=timeintervalrun,
            state_history=state_history,
            verbosity=0)

        # Initialize the sweep object
        sweep_cellgraph = SweepCellGraph(
            sweep_label=sweep_label,
            base_cellgraph_kwargs=base_kwargs,
            params_name=params_name,
            params_values=params_values,
            params_variety=params_variety,
            solver_kwargs=solver_kwargs)

    # Perform the sweep: 3 important settings
    #   - save_cellgraph_files: if False, will only give one big file -- sweep.pkl
    #   - subgraphIsoCheck:     if True, end graph trajectory when it is not isomorphic to a subgraph of any target
    #   - trajectory_max_cells: default None; if int, end graph trajectory when it reaches corresponding size
    sweep_cellgraph.sweep(save_cellgraph_files=True, subgraphIsoCheck=False, trajectory_max_cells=65)
