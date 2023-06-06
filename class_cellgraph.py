import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

from class_hdf5handler import HDF5Handler
from class_singlecell import SingleCell
from dynamics_detect_cycles import detection_args_given_style
from dynamics_graph import graph_ode_system_vectorized, graph_ode_system
from dynamics_vectorfields import set_ode_attributes, ode_integration_defaults
from utils_io import run_subdir_setup
from utils_networkx import draw_from_adjacency
from utils_networkx import check_tree_isomorphism_with_insect
from settings import *

"""
The collection of coupled cells is represented by
- an adjacency matrix defining cell-cell connections
- a corresponding array of state variable timeseries (for each cell)
- and other attributes listed below

Attributes:
- self.num_cells       - integer         - denoted by "M" = number of cells in the system
- self.sc_dim_ode      - integer         - denoted by "N" = number of dynamic variables tracked in a single cell
- self.graph_dim_ode   - integer         - sc_dim_ode * self.num_cells
- self.adjacency       - array (M x M)   - cell-cell adjacency matrix
- self.diffusion       - array (N)       - rate of diffusion may be distinct for each of the N internal sc variables
- self.alpha_sharing   - float           - history-independent fraction of unequal sharing (defined in "Division rules overview" below)
- self.beta_sharing    - float           - history-dependent fraction of unequal sharing (defined in "Division rules overview" below)
- self.labels          - list of strings - unique name for each node on the graph e.g. 'cell_%d'
- self.sc_template     - SingleCell      - instance of custom class which exposes dx/dt=f(x) (where x is one cell)
- self.style_ode       - string          - determines single cell ODE
- self.style_dynamics  - string          - determines how the coupled ODE is integrated (e.g. scipy solve_ivp())
- self.style_division  - string          - determines how graph state changes when cells divide
- self.style_detection - string          - determines how division events are detected
- self.state_history   - array (NM x t)  - state history of the graph
- self.times_history   - array (t)       - timepoints on which graph state was integrated
- self.t0              - float           - initial timepoint maintained through recursive traj; t1 can be flexible
- self.t1              - float           - final timepoint maintained through recursive traj
- self.timeintervalrun - float           - step size for graph integration; see wrapper_graph_trajectory() method
- self.division_events - array (d x 3)   - for each division event, append row: [mother_idx, daughter_idx, time_idx] 
- self.cell_stats      - array (M x 3)   - stores cell metadata: [n_div, time_idx_last_div, time_idx_birth]
- self.io_dict         - dict            - [see utils_io.py] 
                                              if None  -  io_dict = run_subdir_setup(run_subfolder='cellgraph')
                                              if False -  io_dict = False (do not generate new directories)
- self.walltimes       - list of floats  - collect and store clocktime at division events during run

Utility methods:
- self.state_to_stacked(x):    converts array x from shape [N x M] to [NM] 
- self.state_to_rectangle(x):  converts array x from shape [NM] to [N x M]

Issues:
- state_init and state history may need to be reinitialized following a division event, unless we do zero or NaN fill 

Division rules overview: 

if self.style_division == 'copy':
    # copy whole state
    # self.alpha_sharing and self.beta_sharing not used
elif self.style_division == 'partition_equal':
    # divide whole state by half
    # self.alpha_sharing and self.beta_sharing not used
elif self.style_division == 'partition_ndiv_all':
    # for all variables
    # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
    # for mother, r_keep = 0.5 - delta(mother_ndiv)
    # for daughter, (1-r_keep)
elif self.style_division == 'partition_ndiv_bam':
    # for just the z variable
    # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)    
    # for mother, r_keep = 0.5 - delta(mother_ndiv)
    # for daughter, (1-r_keep)
elif self.style_division == 'plus_minus_delta_bam':
    # self.beta_sharing not used
    # for just the z variable
    # for mother, r_keep = 1 - self.alpha_sharing
    # for daughter, r_keep = 1 + self.alpha_sharing
elif self.style_division == 'plus_minus_delta_ndiv_bam':
    # for just the z variable
    # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
    # for mother, r_keep = 1 - delta(mother_ndiv)
    # for daughter, r_keep = 1 + delta(mother_ndiv)
else:
    # self.beta_sharing not used
    # assert self.style_division == 'partition_bam_fixedfrac'
    # for just the z variable
    # for mother, r_keep = 0.5 - self.alpha_sharing
    # for daughter, (1-r_keep)

"""


class CellGraph():

    def __init__(
            self,
            num_cells=1,
            adjacency=None,
            labels=None,
            sc_template=None,
            style_ode=None,
            style_dynamics=None,
            style_detection=None,
            style_division=None,
            style_diffusion=None,
            state_history=None,
            times_history=None,
            diffusion_arg=None,
            alpha_sharing=None,
            beta_sharing=None,
            t0=None,
            t1=None,
            init_cond=None,
            timeintervalrun=None,
            division_events=None,
            cell_stats=None,
            io_dict=None,
            walltimes=None,
            hdf5_mode=HDF5_MODE,
            verbosity=2):
        self.num_cells = num_cells
        self.adjacency = adjacency
        self.labels = labels
        self.sc_template = sc_template
        self.style_ode = style_ode
        self.style_dynamics = style_dynamics
        self.style_detection = style_detection
        self.style_division = style_division
        self.style_diffusion = style_diffusion
        self.diffusion_arg = diffusion_arg
        self.alpha_sharing = alpha_sharing
        self.beta_sharing = beta_sharing
        self.io_dict = io_dict
        self.hdf5_mode = hdf5_mode
        self.hdf5_handler = None
        self.verbosity = verbosity

        # set verbosity level
        self.verbose = False
        self.verbose_deep = False
        if verbosity > 0:
            self.verbose = True       # can pass to external functions; use verbosity through vprint() internally
        if verbosity > 1:
            self.verbose_deep = True  # can pass to external functions; use verbosity through vprint() internally

        # dynamics tracking variables
        self.state_history = state_history          # array NM x t - graph state
        self.times_history = times_history          # array t      - times array
        self.t0 = t0                                # float; defaults to set_integration_defaults t0
        self.t1 = t1                                # float; defaults to set_integration_defaults t1
        self.init_cond = init_cond                  # array N x 1; defaults to set_integration_defaults init_cond
        self.timeintervalrun = timeintervalrun      # float or None; see wrapper_graph_trajectory() for cases
        self.division_events = division_events      # array d x 3  - stores info on division events (details above)
        self.cell_stats = cell_stats                # array M x 3  - stores cell metadata (details above)
        self.walltimes = walltimes                  # list d+1 x 3  - stores walltime at divisions (details above)

        if adjacency is None:
            self.adjacency = np.zeros((self.num_cells, self.num_cells))
        if labels is None:
            self.labels = ['c%d' % c for c in range(0, self.num_cells)]
        if style_ode is None:
            self.style_ode = STYLE_ODE
        if style_dynamics is None:
            self.style_dynamics = STYLE_DYNAMICS
        if style_detection is None:
            self.style_detection = STYLE_DETECTION
        if style_division is None:
            self.style_division = STYLE_DIVISION
        if style_diffusion is None:
            self.style_diffusion = STYLE_DIFFUSION
        if diffusion_arg is None:
            self.diffusion_arg = DIFFUSION_ARG
        if alpha_sharing is None:
            assert self.style_division in ['copy', 'partition_equal']  # i.e. the parameter-less division rules
        if beta_sharing is None:
            assert self.style_division in ['copy', 'partition_equal', 'plus_minus_delta_bam', 'partition_bam_fixedfrac']
        if io_dict is None:
            self.io_dict = run_subdir_setup(run_subfolder='cellgraph')
        elif not io_dict:
            self.io_dict = False  # do not generate a directory; used for base model before param sweep
        else:
            assert isinstance(io_dict, dict)

        # set file output handler as hdf5 custom class, or None
        if self.hdf5_mode and isinstance(self.io_dict, dict):
            hdf5_path = self.io_dict['dir_base'] + os.sep + 'classdata.h5'
            self.hdf5_handler = HDF5Handler(hdf5_path, io_dict=self.io_dict, verbose=HDF5_VERBOSE)
        else:
            self.hdf5_handler = None

        # initialize single cell template which exposes dx/dt=f(x) for internal gene regulation components of network
        if sc_template is None:
            self.sc_template = SingleCell(style_ode=self.style_ode, label='')
        assert self.sc_template.style_ode == self.style_ode

        # construct graph matrices based on adjacency
        self.degree = np.diag(np.sum(self.adjacency, axis=1))
        self.laplacian = self.degree - self.adjacency

        sc_dim_ode, sc_dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.graph_dim_ode = sc_dim_ode * self.num_cells
        self.sc_dim_ode = sc_dim_ode

        # graph diffusion settings
        # 1) create vector self.diffusion (internal diffusion variables can have own diff. rates)
        if isinstance(self.diffusion_arg, (list, tuple, np.ndarray)):
            assert len(self.diffusion_arg) == sc_dim_ode
            self.diffusion = np.array(self.diffusion_arg, dtype=float)
        else:
            assert np.isscalar(self.diffusion_arg)
            self.diffusion = np.array([self.diffusion_arg for _ in range(sc_dim_ode)], dtype=float)
        # 2) if 'xy' style, force all elements beyond the first two (x and y) to zero
        if self.style_diffusion == 'xy':
            assert variables_short[0] == 'Cyc_act'
            assert variables_short[1] == 'Cyc_tot'
            self.diffusion[2:] = 0.0

        sc_t0, sc_t1, sc_num_steps, sc_init_cond = ode_integration_defaults(self.style_ode)
        if self.t0 is None:
            self.t0 = sc_t0
        if self.t1 is None:
            self.t1 = sc_t1
        if self.init_cond is None:
            self.init_cond = sc_init_cond

        if state_history is None:
            # Approach 1: use default init cond from single cell ode code
            state_init = np.tile(self.init_cond, self.num_cells)
            # Approach 2: just zeros
            #state_init = np.zeros(self.graph_dim_ode)
            # Approach 3: random
            #state_init = 10 * np.random.rand(self.graph_dim_ode)
            self.state_history = np.zeros((self.graph_dim_ode, 1))
            self.state_history[:, 0] = state_init

        if self.times_history is None:
            self.times_history = np.array([self.t0])
        if self.division_events is None:
            self.division_events = np.zeros((0, 3), dtype=int)
        if self.cell_stats is None:
            self.cell_stats = np.zeros((self.num_cells, 3), dtype=int)
        if self.walltimes is None:
            self.walltimes = [time.time()]

        assert self.num_cells > 0
        assert self.adjacency.shape == (self.num_cells, self.num_cells)
        assert len(self.labels) == self.num_cells
        assert self.state_history.shape[0] == self.graph_dim_ode
        assert self.style_ode in STYLE_ODE_VALID
        assert self.style_dynamics in STYLE_DYNAMICS_VALID
        assert self.style_detection in STYLE_DETECTION_VALID
        assert self.style_division in STYLE_DIVISION_VALID
        assert self.style_diffusion in STYLE_DIFFUSION_VALID
        assert all([c >= 0.0 for c in self.diffusion])
        assert self.division_events.shape[1] == 3
        assert self.cell_stats.shape[0] == self.num_cells and self.cell_stats.shape[1] == 3

    def vprint(self, vlevel, s, *args):
        """
        Given a string s and a verbosity label vlevel
        print(s) if self.verbosity is at vlevel or higher
        """
        if self.verbosity >= vlevel:
            print(s, *args)

    def cell_last_division_time_idx(self, cell_idx):
        """
        Wrapper to get time index of last division event for given cell_idx
        Note: This represents the time it acted as a daughter cell or mother cell in a division event
        """
        return int(self.cell_stats[cell_idx, 1])

    def cell_birth_time_idx(self, cell_idx):
        return int(self.cell_stats[cell_idx, 2])

    def time_indices_where_acted_as_mother(self, cell_choice):
        """
        Returns array like of time indices where cell_choice acted as the mother cell during a division event
        """
        event_acted_as_mother = np.argwhere(self.division_events[:, 0] == cell_choice)
        if len(event_acted_as_mother) > 0:
            events_idx = self.division_events[event_acted_as_mother, 2]
        else:
            events_idx = []
        return events_idx

    def n_divisions(self):
        return int(self.division_events.shape[0])

    def division_event(self, idx_dividing_cell, idx_timepoint):
        """
        Returns a new instance of the CellGraph with updated state variables (as a result of adding one cell)

        Division rules overview: 

        if self.style_division == 'copy':
            # copy whole state
            # self.alpha_sharing and self.beta_sharing not used
        elif self.style_division == 'partition_equal':
            # divide whole state by half
            # self.alpha_sharing and self.beta_sharing not used
        elif self.style_division == 'partition_ndiv_all':
            # for all variables
            # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            # for mother, r_keep = 0.5 - delta(mother_ndiv)
            # for daughter, (1-r_keep)
        elif self.style_division == 'partition_ndiv_bam':
            # for just the z variable
            # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)    
            # for mother, r_keep = 0.5 - delta(mother_ndiv)
            # for daughter, (1-r_keep)
        elif self.style_division == 'plus_minus_delta_bam':
            # self.beta_sharing not used
            # for just the z variable
            # for mother, r_keep = 1 - self.alpha_sharing
            # for daughter, r_keep = 1 + self.alpha_sharing
        elif self.style_division == 'plus_minus_delta_ndiv_bam':
            # for just the z variable
            # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            # for mother, r_keep = 1 - delta(mother_ndiv)
            # for daughter, r_keep = 1 + delta(mother_ndiv)
        else:
            # self.beta_sharing not used
            # assert self.style_division == 'partition_bam_fixedfrac'
            # for just the z variable
            # for mother, r_keep = 0.5 - self.alpha_sharing
            # for daughter, (1-r_keep)

        """

        def partition_ndiv(mother_ndiv, current_mother_state, partition_ndiv_indices):
            calculated_delta = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            # for mother, r_keep = 0.5 - delta(mother_ndiv)
            # for daughter, (1-r_keep)
            r_keep = 0.5 - calculated_delta
            post_mother_state = np.copy(current_mother_state)
            post_daughter_state = np.copy(current_mother_state)
            post_mother_state[partition_ndiv_indices] = post_mother_state[partition_ndiv_indices] * r_keep
            post_daughter_state[partition_ndiv_indices] = post_daughter_state[partition_ndiv_indices] * (1 - r_keep)
            return post_mother_state, post_daughter_state

        def partition_fixed_frac(current_mother_state, partition_indices):
            r_keep = 0.5 - self.alpha_sharing
            post_mother_state = np.copy(current_mother_state)
            post_daughter_state = np.copy(current_mother_state)
            post_mother_state[partition_indices] = post_mother_state[partition_indices] * r_keep
            post_daughter_state[partition_indices] = post_daughter_state[partition_indices] * (1 - r_keep)
            return post_mother_state, post_daughter_state

        def plus_minus_delta_func(current_mother_state, partition_indices):
            post_mother_state = np.copy(current_mother_state)
            post_daughter_state = np.copy(current_mother_state)
            r_keep = 1 - self.alpha_sharing
            post_mother_state[partition_indices] = post_mother_state[partition_indices] * r_keep
            r_keep = 1 + self.alpha_sharing
            post_daughter_state[partition_indices] = post_daughter_state[partition_indices] * r_keep
            return post_mother_state, post_daughter_state

        def plus_minus_delta_ndiv_func(mother_ndiv, current_mother_state, partition_ndiv_indices):
            post_mother_state = np.copy(current_mother_state)
            post_daughter_state = np.copy(current_mother_state)
            # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            # for mother, r_keep = 1 - delta(mother_ndiv)
            # for daughter, r_keep = 1 + delta(mother_ndiv)
            calculated_delta = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
            r_keep = 1 - calculated_delta
            post_mother_state[partition_ndiv_indices] = post_mother_state[partition_ndiv_indices] * r_keep
            r_keep = 1 + calculated_delta
            post_daughter_state[partition_ndiv_indices] = post_daughter_state[partition_ndiv_indices] * r_keep
            return post_mother_state, post_daughter_state

        def split_mother_and_daughter_state():
            current_graph_state = self.state_history[:, -1]
            mother_idx_low = self.sc_dim_ode * idx_dividing_cell
            mother_idx_high = self.sc_dim_ode * (idx_dividing_cell + 1)
            current_mother_state = current_graph_state[mother_idx_low : mother_idx_high]
            mother_ndiv = self.cell_stats[idx_dividing_cell, 0]

            if self.style_division == 'copy':
                # copy whole state
                # self.alpha_sharing and self.beta_sharing not used
                post_mother_state = current_mother_state
                post_daughter_state = current_mother_state
            elif self.style_division == 'partition_equal':
                # divide whole state by half
                # self.alpha_sharing and self.beta_sharing not used
                post_mother_state = current_mother_state / 2.0
                post_daughter_state = current_mother_state / 2.0
            elif self.style_division == 'partition_ndiv_all':
                # for all variables
                # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
                # for mother, r_keep = 0.5 - delta(mother_ndiv)
                # for daughter, (1-r_keep)
                assert self.alpha_sharing is not None
                assert self.beta_sharing is not None
                # the condition depends on whether beta is negative or non-negative, thus the sign function
                assert (self.alpha_sharing - 0.5) <= (1 - np.sign(self.beta_sharing)) * self.beta_sharing / 2
                assert (self.alpha_sharing + 0.5) >= (1 + np.sign(self.beta_sharing)) * self.beta_sharing / 2
                partition_ndiv_indices = [i for i in range(self.sc_dim_ode)]
                post_mother_state, post_daughter_state = \
                    partition_ndiv(mother_ndiv, current_mother_state, partition_ndiv_indices)
            elif self.style_division == 'partition_ndiv_bam':
                # for just the z variable
                # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
                # for mother, r_keep = 0.5 - delta(mother_ndiv)
                # for daughter, (1-r_keep)
                assert self.alpha_sharing is not None
                assert self.beta_sharing is not None
                # the condition depends on whether beta is negative or non-negative, thus the sign function
                assert (self.alpha_sharing - 0.5) <= (1 - np.sign(self.beta_sharing)) * self.beta_sharing / 2
                assert (self.alpha_sharing + 0.5) >= (1 + np.sign(self.beta_sharing)) * self.beta_sharing / 2
                assert self.sc_template.variables_short[2] == 'Bam'
                partition_ndiv_indices = [2]
                post_mother_state, post_daughter_state = \
                    partition_ndiv(mother_ndiv, current_mother_state, partition_ndiv_indices)
            elif self.style_division == 'plus_minus_delta_bam':
                # self.beta_sharing not used
                # for just the z variable
                # for mother, r_keep = 1 - self.alpha_sharing
                # for daughter, r_keep = 1 + self.alpha_sharing
                assert self.alpha_sharing is not None
                assert -1 <= self.alpha_sharing <= 1
                assert self.sc_template.variables_short[2] == 'Bam'
                partition_indices = [2]
                post_mother_state, post_daughter_state = \
                    plus_minus_delta_func(current_mother_state, partition_indices)
            elif self.style_division == 'plus_minus_delta_ndiv_bam':
                # for just the z variable
                # delta(mother_ndiv) = self.alpha_sharing - self.beta_sharing / (1 + mother_ndiv)
                # for mother, r_keep = 1 - delta(mother_ndiv)
                # for daughter, r_keep = 1 + delta(mother_ndiv)
                assert self.alpha_sharing is not None
                assert self.beta_sharing is not None
                # the condition depends on whether beta is negative or non-negative, thus the sign function
                assert (self.alpha_sharing - 1) <= (1 - np.sign(self.beta_sharing)) * self.beta_sharing / 2
                assert (self.alpha_sharing + 1) >= (1 + np.sign(self.beta_sharing)) * self.beta_sharing / 2
                assert self.sc_template.variables_short[2] == 'Bam'
                partition_ndiv_indices = [2]
                post_mother_state, post_daughter_state = \
                    plus_minus_delta_ndiv_func(mother_ndiv, current_mother_state, partition_ndiv_indices)
            else:
                # self.beta_sharing not used
                # for just the z variable
                # for mother, r_keep = 1/2 - self.alpha_sharing
                # for daughter, (1-r_keep)
                assert self.alpha_sharing is not None
                assert -0.5 <= self.alpha_sharing <= 0.5
                assert self.style_division == 'partition_bam_fixedfrac'
                assert self.sc_template.variables_short[2] == 'Bam'
                partition_indices = [2]
                post_mother_state, post_daughter_state = \
                    partition_fixed_frac(current_mother_state, partition_indices)

            return post_mother_state, post_daughter_state, mother_idx_low, mother_idx_high

        idx_daughter_cell = self.num_cells  # e.g. if there is 1 cell that divides, idx mother = 0, idx daughter = 1
        updated_num_cells = self.num_cells + 1
        updated_graph_dim_ode = int(self.sc_dim_ode * updated_num_cells)

        updated_labels = self.labels + ['c%d' % (updated_num_cells - 1)]
        # generate update adjacency matrix
        updated_adjacency = np.zeros((updated_num_cells, updated_num_cells))
        # fill A[0:M-1, 0:M-1] entries with old adjacency matrix
        updated_adjacency[0:updated_num_cells - 1, 0:updated_num_cells - 1] = self.adjacency
        # add new row/column with index corresponding k to the generating cell
        # i.e. for input i = idx_dividing_cell, set A[i, k] = 1 and # set A[k, i] = 1
        updated_adjacency[idx_dividing_cell, -1] = 1
        updated_adjacency[-1, idx_dividing_cell] = 1

        # update new initial state based on extra cell and dispersion of maternal variables
        updated_state_history = np.zeros((updated_graph_dim_ode, len(self.times_history)))
        self.vprint(2, "division_event() CHECK %s %s %s %s" % (updated_state_history.shape, updated_state_history.shape, self.state_history.shape, self.times_history.shape))
        updated_state_history[0:self.graph_dim_ode, :] = self.state_history
        post_mother_state, post_daughter_state, mlow, mhigh = split_mother_and_daughter_state()
        self.vprint(2, "division_event() %d %d %d" % (updated_graph_dim_ode, self.graph_dim_ode, len(post_daughter_state)))
        self.vprint(2, "division_event() %s %s %d %d" % (post_mother_state, post_daughter_state, mlow, mhigh))
        updated_state_history[mlow:mhigh, -1] = post_mother_state
        updated_state_history[self.graph_dim_ode:, -1] = post_daughter_state

        # update division event array: (d x 3)   - see details in docstring
        n_divisions = self.n_divisions() + 1
        updated_division_events = np.zeros((n_divisions, 3), dtype=int)
        updated_division_events[:-1, :] = self.division_events
        updated_division_events[-1, :] = np.array([idx_dividing_cell, idx_daughter_cell, idx_timepoint])

        # update cell stats array: (M x 2)   - see details in docstring
        updated_cell_stats = np.zeros((updated_num_cells, 3), dtype=int)
        updated_cell_stats[:-1, :] = self.cell_stats
        # cell stats for new daughter cell: has no divisions, last division event is *now*, is born *now*
        updated_cell_stats[-1, :] = np.array([0, idx_timepoint, idx_timepoint])  # update cell stats for daughter cell
        updated_cell_stats[idx_dividing_cell, 0] += 1              # for mother cell - update stat: n_div
        updated_cell_stats[idx_dividing_cell, 1] = idx_timepoint   # for mother cell - update stat: time_last_div

        # update walltimes: list of floats, length (d+1), that tracks walltime at division events (benchmarking)
        self.walltimes.append(time.time())

        new_cellgraph = CellGraph(
            style_ode=self.style_ode,
            style_dynamics=self.style_dynamics,
            style_detection=self.style_detection,
            style_division=self.style_division,
            style_diffusion=self.style_diffusion,
            io_dict=self.io_dict,
            sc_template=self.sc_template,
            num_cells=updated_num_cells,
            adjacency=updated_adjacency,
            labels=updated_labels,
            state_history=updated_state_history,
            times_history=self.times_history,
            diffusion_arg=self.diffusion_arg,
            alpha_sharing=self.alpha_sharing,
            beta_sharing=self.beta_sharing,
            t0=self.t0,
            t1=self.t1,
            timeintervalrun=self.timeintervalrun,
            division_events=updated_division_events,
            cell_stats=updated_cell_stats,
            walltimes=self.walltimes,
            verbosity=self.verbosity
        )
        return new_cellgraph

    def graph_trajectory_TOYFLOW(self, init_cond=None, t0=None, t1=None, **solver_kwargs):
        single_cell = None
        assert self.style_ode == 'toy_flow'

        def graph_ode_system(t_scalar, xvec, single_cell):
            dxvec_dt = -1 * self.diffusion * np.dot(self.laplacian, xvec)
            return dxvec_dt

        fn = graph_ode_system
        time_interval = [t0, t1]

        if 'vectorized' not in solver_kwargs.keys():
            solver_kwargs['vectorized'] = True
        sol = solve_ivp(fn, time_interval, init_cond, args=(single_cell,), **solver_kwargs)
        r = np.transpose(sol.y)
        times = sol.t
        return r, times

    def detect_oscillations_graph_trajectory(self, times, traj):
        """
        Given a graph trajectory, detect oscillation events (if any).
        Detection method is based on self.style_detection
            - if self.style_detection == 'ignore': function short circuits, return False event and None for the rest
            - if self.style_detection == 'scipy_peaks': uses detect_oscillations_scipy() on each cell traj since its last division
        Args:
            times            - full history of times (i.e. pass copy of self.times_history)
            traj             - full history of state evolution (i.e. pass copy of self.state_history)
        Returns:
            event_detected   - bool
            event_cell       - None; or cell index where the event occurred first
            event_idx        - None; or time index of the event
            event_time       - None; or actual time (i.e. t[t_index]) of the event
            truncated_times  - None; or times[0:event_idx]
            truncated_traj   - None; or traj[0:event_idx, ...]
        """

        event_detected = False
        first_event_cell = None
        first_event_idx = None
        first_event_time = 1e11
        truncated_times = None
        truncated_traj = None

        if self.style_detection == 'ignore':
            pass
        else:
            detect_fn, detect_kwargs = detection_args_given_style(
                self.style_detection, self.sc_template, verbose=self.verbose_deep)
            self.vprint(2, "\n=========== Detection call outer print ===========")
            self.vprint(2, "\tdetect_fn %s" % detect_fn)
            self.vprint(2, "\tdetect_kwargs %s" % detect_kwargs)
            traj_rectangle = self.state_to_rectangle(traj)
            for idx in range(self.num_cells):
                # for each cell, check the trajectory since its last division to look for oscillation event
                self.vprint(2, 'detect_osc().. step 4 - accessing cell last division time (idx, t_absolute) tuple for cell idx %d' % (idx))
                time_idx_last_div_specific_cell = self.cell_last_division_time_idx(idx)
                self.vprint(2, 'detect_osc().. step 3')
                slice_traj = traj_rectangle[:, idx, time_idx_last_div_specific_cell:]
                self.vprint(2, 'detect_osc().. step 2')
                slice_times = times[time_idx_last_div_specific_cell:]
                self.vprint(2, 'detect_osc().. step 1')
                self.vprint(2, 'detect_osc().. before enter detect_fn()... %d, %s, %s' % (time_idx_last_div_specific_cell, times[0:3], times[-3:]))
                detect_args = (slice_times, slice_traj)
                num_oscillations, events_idx, events_times, duration_cycles = detect_fn(*detect_args, **detect_kwargs)

                if num_oscillations > 0:
                    self.vprint(1, 'EVENT: %d oscillations detected for cell %d' % (num_oscillations, idx))
                    event_detected = True
                    if events_times[0] == first_event_time:
                        self.vprint(1, 'Note - two cells are dividing at the same time, we are picking '
                                       'the one with the earlier index to divide... why not have all divide?')
                        continue

                    # This is the earliest division event identified so far -- record it (and possibly return it)
                    # since we gave sliced array, need to shift the index of the times to match our full self.times_history
                    if events_times[0] < first_event_time:
                        first_event_time = events_times[0]                                 # no need to shift (absolute t)
                        first_event_idx = events_idx[0] + time_idx_last_div_specific_cell  # need to shift because of slice
                        first_event_cell = idx
                else:
                    self.vprint(2, 'No oscillation detected for cell %d' % idx)

        if event_detected:
            buffer = 1
            self.vprint(2, 'TODO remove buffer event detected')
            truncated_times = times[0:(first_event_idx + buffer)]
            truncated_traj = traj[:, 0:(first_event_idx + buffer)]

        return event_detected, first_event_cell, first_event_idx, truncated_times, truncated_traj

    def wrapper_graph_trajectory(self, t0=None, t1=None, timeintervalrun=None, plotting=False,
                                 subgraphIsoCheck=False, trajectory_max_cells=None,
                                 flagWriteState=True, **solver_kwargs):
        """
        Iteratively runs graph_trajectory() to integrate from time t0 to t1, expanding the graph at each cell division

        If timeintervalrun is None: exit when no division events detected in truncating loop
        - step 1: compute full graph trajectory on [t_start, t_end]
        - step 2: inspect the trajectory for division events
            - if none are detected, exit the function
            - if one or more are detected, go to step 3
        - step 3: cut off trajectory at, and set t0_shifted to, the earliest division event -- t_last_div
        - step 4: expand the graph by one cell
        - step 5: redo steps 1 & 2 but replace t_start by t_last_div

        Else: exit when no division events or t1_shifted == t_end
        - step 1: compute full graph trajectory on [t_start, timeintervalrun]
        - step 2: inspect the trajectory for division events
            - if none are detected, compute next slice of trajectory [t_last, t_last + timeintervalrun]
            - if one or more are detected, go to step 3
        - step 3: cut off trajectory at, and set t0_shifted to, the earliest division event -- t_last_div
        - step 4: expand the graph by one cell
        - step 5: redo steps 1 & 2 but integrate on [t_last_div, t1_shifted] (which might be very small window)

        trajectory_max_cells [None]: exit the graph trajectory loop whenever a certain network size has been reached
        """
        if t0 is None:
            t0 = self.t0
        if t1 is None:
            t1 = self.t1
        if timeintervalrun is None:
            timeintervalrun = self.timeintervalrun
        if trajectory_max_cells is not None:
            assert isinstance(trajectory_max_cells, int)
        else:
            trajectory_max_cells = int(1e9)

        # loop initialization
        continue_trajectory = True  # main wrapper loop flag
        division_detected = None    # key sub-flag (will be set on first loop)
        division_counter = 0

        t0_shifted = t0
        # specify a time interval over which to run the code in search of an event -- change by HAYDEN
        if self.timeintervalrun is None:
            t1_shifted = t1
        else:
            assert 0 <= timeintervalrun <= (t1 - t0)
            t1_shifted = t0 + timeintervalrun

        cellgraph = self  # care: all self.foo should be cellgraph.foo below

        self.vprint(1, "\nwrapper_graph_trajectory(): Input number of cells", cellgraph.num_cells)
        self.vprint(1, "wrapper_graph_trajectory(): Input times shape", cellgraph.times_history.shape)
        self.vprint(1, "wrapper_graph_trajectory(): Input state shape", cellgraph.state_history.shape)
        self.vprint(1, "wrapper_graph_trajectory(): Input cell_stats shape", cellgraph.cell_stats.shape)

        while continue_trajectory:

            # Step (1) - Integrate relevant slice
            sub_interval = [t0_shifted, t1_shifted]
            self.vprint(2, "\nwrapper_graph_trajectory(): division_counter %d, time interval" % division_counter, sub_interval)

            # Step (2) - Inspect each cell's trajectory for division events
            division_detected, cellgraph = cellgraph.graph_trajectory(time_interval=sub_interval, **solver_kwargs)
            self.vprint(2, "wrapper_graph_trajectory(), subsequent line: cellgraph.cell_stats.shape", cellgraph.cell_stats.shape)

            # If a division event is observed, several immediate things need to be done
            if division_detected:
                self.vprint(1, "wrapper_graph_trajectory(): ===========Division event detected===========")
                time_last_division_idx = cellgraph.division_events[-1, 2]  # time index of most recent division
                t0_shifted = cellgraph.times_history[time_last_division_idx]  # move the start time of the integration
                division_counter += 1

                self.vprint(1, "\nDivision event in wrapper_graph_trajectory():")
                self.vprint(2, "wrapper_graph_trajectory(), before printer: cellgraph.cell_stats.shape",
                            cellgraph.cell_stats.shape)
                if self.verbose:
                    cellgraph.print_state(msg='call in wrapper_graph_trajectory()')
                if flagWriteState:
                    cellgraph.write_state(fmod='iter%d' % division_counter)
                if plotting:
                    cellgraph.plot_graph(fmod='iter%d' % division_counter)

            # Two central loop cases from the function docstring
            if timeintervalrun is None:
                # in this case, only continue wrapper if division is detected
                continue_trajectory = division_detected
            else:
                # in this case, continue wrapper even when division is NOT detected, if t1_shifted < max time
                if not division_detected:
                    if abs(t1_shifted - t1) < 1e-9:
                        continue_trajectory = False
                    else:
                        t0_shifted = cellgraph.times_history[-1]
                        # if solve_ivp subtly fails with: "Required step size is less than spacing between numbers.", this assert catches that
                        assert cellgraph.times_history[-1] == t1_shifted
                        t1_shifted = t0_shifted + timeintervalrun
                        if t1_shifted > t1:
                            t1_shifted = t1
                        continue_trajectory = True

            # Short-circuit option: check if the graph continues to be building towards one of the known insect cysts
            if division_detected:
                # CHANGE BY HAYDEN - wasp, ant, bee, moth, lacewing, parthenogeneticus, parasiticwasp
                this_cell_adjacency = cellgraph.adjacency
                # COULD ADD OTHER STRUCTURES -- also could specify at input of wrapper graph trajectory
                if subgraphIsoCheck:
                    is_isomorphic = check_tree_isomorphism_with_insect(this_cell_adjacency, 'wasp', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'ant', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'bee', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'alternativelacewing1', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'alternativelacewing2', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'parthenogeneticus', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'fulvicephalus1', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'fulvicephalus2', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, 'fulvicephalus3', True)
                    is_isomorphic = is_isomorphic or check_tree_isomorphism_with_insect(this_cell_adjacency, '16cellcyst_linear', True)
                else:
                    is_isomorphic = True
                if not is_isomorphic:
                    self.vprint(1, "wrapper_graph_trajectory(), short-circuit exit due to NOT is_isomorphic")
                    continue_trajectory = False
                    # break
                if cellgraph.num_cells >= trajectory_max_cells:
                    self.vprint(1, "wrapper_graph_trajectory(), short-circuit exit due to trajectory_max_cells")
                    continue_trajectory = False

            # Extra: for bug-checking / inspection
            if self.verbose_deep:
                print("wrapper_graph_trajectory(): ===========CORE LOOP PRINT========")
                for idx in range(5):
                    print(cellgraph.num_cells, idx, cellgraph.times_history[idx],
                          '\n\t', cellgraph.state_history[:, idx])
                print('...')
                for idx in range(5):
                    shiftidx = len(cellgraph.times_history) - 5 + idx
                    print(cellgraph.num_cells, shiftidx, cellgraph.times_history[shiftidx],
                          '\n\t', cellgraph.state_history[:, shiftidx])
                print("wrapper_graph_trajectory(): ===========CORE LOOP END========")

        self.vprint(1, "wrapper_graph_trajectory(): Output number of cells", cellgraph.num_cells)
        self.vprint(1, "wrapper_graph_trajectory(): Output times shape", cellgraph.times_history.shape)
        self.vprint(1, "wrapper_graph_trajectory(): Output state shape", cellgraph.state_history.shape)
        self.vprint(1, "wrapper_graph_trajectory(): Output cell_stats shape", cellgraph.cell_stats.shape)

        return division_detected, cellgraph

    def graph_trajectory(self, time_interval, init_cond=None, **solver_kwargs):
        """
        In principle, can simulate starting from the current state of the graph to some arbitrary timepoint,
        However, we'd like to "pause" after the first cell completes a cycle (call this time "t_div").
          - Issue #1: how to detect this given a timeseries of state changes for the whole graph.
          - Issue #2: what if no limit cycles are observed (answer - we keep simulating until quiescence).
          - Immediately at t_div, we record the graph state, pause it, create a new instance of the graph with M + 1
            cells, and decide how to split the material between the dividing cell and its daughter cell.

        Suppose the ODE for a single cell with state x_1 in R^N is given by
            d{x_1}/dt = F({x_1})
        Then the ODE for M coupled cells with a diffusive coupling of all N variables is
            dX/dt = F(X) - LX
        Where
            X = [x_1, x_2, ..., x_M]^T a stacked vector of length NM representing the state of all cells
        """
        if init_cond is None:
            init_cond = self.state_history[:, -1]

        if 't_eval' in solver_kwargs.keys():
            if solver_kwargs['t_eval'] is not None:
                print("In graph trajectory -- overriding time_interval based on solver_kwargs['t_eval']")
                print("\tInput time_interval:", time_interval)
                print("\tsolver_kwargs['t_eval']:", solver_kwargs['t_eval'])
                time_interval[0] = min(time_interval[0], solver_kwargs['t_eval'][0])
                time_interval[1] = max(time_interval[1], solver_kwargs['t_eval'][-1])
                print("\tModified time_interval:", time_interval)

        # Vectorization handler
        if 'vectorized' not in solver_kwargs.keys():
            solver_kwargs['vectorized'] = False

        if solver_kwargs['vectorized']:
            fn = graph_ode_system_vectorized
        else:
            fn = graph_ode_system

        assert self.style_dynamics == 'solve_ivp'
        sol = solve_ivp(fn, time_interval, init_cond, args=(self.sc_template, self), **solver_kwargs)
        r = sol.y
        times = sol.t

        self.vprint(2, "graph_trajectory() CHECK solve_ivp output: r.shape t.shape, t[0], self.times_history[0] ---", r.shape, times.shape, times[0], self.times_history[0])
        self.vprint(2, "graph_trajectory() CHECK HISTORY before np.concatenate()", self.state_history.shape, self.times_history.shape)
        state_history_extended = np.concatenate((self.state_history, r[:, 1:]), axis=1)
        times_history_extended = np.concatenate((self.times_history, times[1:]), axis=0)
        self.vprint(2, "graph_trajectory() CHECK HISTORY after np.concatenate()", state_history_extended.shape, times_history_extended.shape)

        # Module: Oscillation detection (no event detected if self.style_detection is 'ignore')
        event_detected, mother_cell, event_time_idx, times_history_truncated, state_history_truncated = \
            self.detect_oscillations_graph_trajectory(times_history_extended, state_history_extended)
        if event_detected:
            state_history_extended = state_history_truncated
            times_history_extended = times_history_truncated
        self.vprint(2, "Updating state, time history in graph_trajectory()")
        self.state_history = state_history_extended
        self.times_history = times_history_extended
        self.vprint(2, "\tnew dimensions time", self.times_history.shape)
        self.vprint(2, "\tnew dimensions state", self.state_history.shape)
        self.vprint(2, "graph_trajectory() CHECK HISTORY before div event...", self.state_history.shape, self.times_history.shape)
        self.vprint(2, "Note the reported time event idx is", event_time_idx)

        if event_detected:
            new_cellgraph = self.division_event(mother_cell, event_time_idx)
        else:
            new_cellgraph = self

        self.vprint(2, "graph_trajectory(), final line: new_cellgraph.cell_stats.shape", new_cellgraph.cell_stats.shape)
        return event_detected, new_cellgraph

    def print_state(self, msg=''):
        print("CellGraph print_state()", msg)
        print("\tself.adjacency\n", self.adjacency)
        print("\tself.labels -", self.labels)
        print("\tself.num_cells, self.graph_dim_ode, self.sc_dim_ode -", self.num_cells, self.graph_dim_ode, self.sc_dim_ode)
        print("\tself.style_ode -", self.style_ode)
        print("\tself.style_dynamics -", self.style_dynamics)
        print("\tself.style_detection -", self.style_detection)
        print("\tself.style_division -", self.style_division)
        print("\tself.style_diffusion -", self.style_diffusion)
        print("\tself.diffusion_arg -", self.diffusion_arg)
        print("\tself.alpha_sharing -", self.alpha_sharing)
        print("\tself.beta_sharing -", self.beta_sharing)
        print("\tself.diffusion -", self.diffusion)
        print("\tself.state_history.shape, self.time_history.shape -", self.state_history.shape, self.times_history.shape)
        print("\tself.t0", self.t0)
        print("\tself.t1", self.t1)
        print("\tself.init_cond", self.init_cond)
        print("\tself.timeintervalrun", self.timeintervalrun)
        print("\ttimepoints: t0, t1: -", self.times_history[0], self.times_history[-1])
        print("\tself.division_events.shape: -", self.division_events.shape)
        print("\tself.cell_stats.shape: -", self.cell_stats.shape)
        print("\tself.walltimes length: -", len(self.walltimes))
        print("\t\ttotal time so far: -", self.walltimes[-1] - self.walltimes[0])
        print("\tCurrent [state] and [ndiv, time last division, time birth]:")
        X = self.state_to_rectangle(self.state_history)
        for cell in range(self.num_cells):
            print('\t\tCell #%d' % cell, X[:, cell, -1].flatten(), 'stats:', self.cell_stats[cell, :])
        return

    def plot_graph(self, fmod=None, title='CellGraph', by_ndiv=True, by_degree=True, by_last_div=True, by_age=True,
                   spring_seed=None, gviz_prog='dot', figsize=(4,4)):
        fpath = self.io_dict['dir_plots'] + os.sep + 'networkx'
        if fmod is not None:
            title = title + ' ' + fmod
            fpath = self.io_dict['dir_plots'] + os.sep + 'networkx_%s' % fmod
        if by_ndiv:
            tvar = title + ' (number of divisions)'
            fpathvar = fpath + '_nDiv'
            n_divisions = self.cell_stats[:, 0]
            labels = {idx: r'$%d$' % (val) for idx, val in enumerate(n_divisions)}
            draw_from_adjacency(self.adjacency, title=tvar, node_color=n_divisions, labels=labels, cmap='Pastel1',
                                fpath=fpathvar, draw_division=self.division_events, figsize=figsize,
                                spring_seed=spring_seed, gviz_prog=gviz_prog)
        if by_degree:
            tvar = title + ' (degree)'
            fpathvar = fpath + '_Degree'
            degree_vec = np.diag(self.degree)
            labels = None
            draw_from_adjacency(self.adjacency, title=tvar, node_color=degree_vec, labels=labels, cmap='Pastel1',
                                fpath=fpathvar, draw_division=self.division_events, figsize=figsize,
                                spring_seed=spring_seed, gviz_prog=gviz_prog)
        if by_last_div:
            tvar = title + ' (time of last division)'
            fpathvar = fpath + '_tLastEvent'
            t_last_div = self.cell_stats[:, 1]
            t_last_div_abs = self.times_history[t_last_div]
            labels = {idx: r'$%.1f$' % (val) for idx, val in enumerate(t_last_div_abs)}
            draw_from_adjacency(self.adjacency, title=tvar, node_color=t_last_div_abs, labels=labels, cmap='GnBu',
                                fpath=fpathvar, figsize=figsize,
                                spring_seed=spring_seed, gviz_prog=gviz_prog)
        if by_age:
            tvar = title + ' (time of birth)'
            fpathvar = fpath + '_tBirth'
            birthdays = self.cell_stats[:, 2]
            birthdays_abs = self.times_history[birthdays]
            labels = {idx: r'$%.1f$' % (val) for idx, val in enumerate(birthdays_abs)}
            draw_from_adjacency(self.adjacency, title=tvar, node_color=birthdays_abs, labels=labels, cmap='GnBu',
                                fpath=fpathvar, figsize=figsize,
                                spring_seed=spring_seed, gviz_prog=gviz_prog)
        plt.close()
        return

    def plot_state_unified(self, fmod='', arrange_vertical=True, decorate=True):

        if arrange_vertical:
            # subplots as M x 1 grid
            ncols = 1
            nrows = self.num_cells
            sharex = True
        else:
            # subplots as k x 4 grid
            assert self.num_cells <= 16  # for now
            ncols = 4
            nrows = 1 + (self.num_cells - 1) // ncols
            sharex = False

        fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8), constrained_layout=True, squeeze=False, sharex=sharex)
        state_tensor = self.state_to_rectangle(self.state_history)
        times = self.times_history
        birthdays = self.cell_stats[:, 2]

        for idx in range(self.num_cells):
            if arrange_vertical:
                i, j = idx, 0
            else:
                i = idx // 4
                j = idx % 4

            r = np.transpose(state_tensor[:, idx, :])
            # start at different points for each cell (based on "birthday")
            init_idx = birthdays[idx]
            # peform plotting
            axarr[i, j].plot(
                times[init_idx:], r[init_idx:, :], label=[self.sc_template.variables_short[i] for i in range(self.sc_dim_ode)])
            # set labels for axes
            axarr[i, j].set_ylabel(r'$x_{%d}$' % idx)
            if not arrange_vertical or i == self.num_cells - 1:
                axarr[i, j].set_xlabel(r'$t$')

            if decorate:
                # plot points in phase space where division event occurred
                events_idx = self.time_indices_where_acted_as_mother(idx)
                for event_idx in events_idx:
                    axarr[i,j].axvline(times[event_idx], linestyle='--', c='gray')

        plt.legend()
        plt.suptitle('plot_state_each_cell() - %s' % fmod)

        fpath = self.io_dict['dir_plots'] + os.sep + 'states_all'
        if fmod is not None:
            fpath += '_' + fmod
        plt.savefig(fpath + '.pdf')
        plt.close()
        return

    def plot_xyz_state_for_specific_cell(self, plot_cell_index=0, decorate=True, fpath=None):
        assert plot_cell_index in range(self.num_cells)

        plt.rcParams.update({'font.size': 24})

        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        state_tensor = self.state_to_rectangle(self.state_history)
        times = self.times_history
        birthdays = self.cell_stats[:, 2]

        idx = plot_cell_index

        r = np.transpose(state_tensor[:, idx, :])
        # start at different points for each cell (based on "birthday")
        init_idx = birthdays[idx]
        # perform plotting
        plt.plot(times[init_idx:], r[init_idx:, 0],
                         label=r'$x$', linewidth=2.0, color="#006ebbff")
        plt.plot(times[init_idx:], r[init_idx:, 1],
                         label=r'$y$', linewidth=2.0, color="red")
        plt.plot(times[init_idx:], r[init_idx:, 2],
                         label=r'$z$ pulse', linewidth=2.0, color="#76a932ff")
        # set labels for axes
        ax.set_ylabel(r'$x_{%d}$' % idx)
        ax.set_xlabel(r'$t$')

        if decorate:
            # plot points in phase space where division event occurred
            events_idx = self.time_indices_where_acted_as_mother(idx)
            for event_idx in events_idx:
                ax.axvline(times[event_idx], linestyle='--', c='gray')

        plt.xlim(0, times[-1])
        plt.legend()

        if fpath is None:
            fmod = str(plot_cell_index)
            fpath = self.io_dict['dir_plots'] + os.sep + 'states_individual'
            fpath += '_' + fmod

        plt.savefig(fpath + '.pdf')
        plt.close()
        return

    def plot_xy_separate(self, quiver=True, decorate=True, fmod=''):
        state_tensor = self.state_to_rectangle(self.state_history)
        times = self.times_history
        birthdays = self.cell_stats[:, 2]

        fixed_cmap = plt.get_cmap('Spectral')

        for idx in range(self.num_cells):

            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            r = np.transpose(state_tensor[:, idx, :])
            # start at different points for each cell (based on "birthday")
            init_idx = birthdays[idx]
            times_slice = times[init_idx:]
            r_slice = r[init_idx:, :]

            # perform plotting
            alpha = 0.5
            sc = plt.scatter(r_slice[:, 0], r_slice[:, 1], c=times_slice, cmap=fixed_cmap, alpha=alpha)  # draw scatter points
            plt.plot(r_slice[:, 0], r_slice[:, 1], '-k', linewidth=0.5, alpha=alpha)  # draw connections between points

            if quiver:
                weighted_start = 0.5  # default: 0
                weighted_shrink = 0.5 # default: 1
                x0 = (r_slice[:-1, 0] + weighted_start * r_slice[1:, 0]) / (1 + weighted_start)
                y0 = (r_slice[:-1, 1] + weighted_start * r_slice[1:, 1]) / (1 + weighted_start)
                u0 = weighted_shrink * (r_slice[1:, 0] - r_slice[:-1, 0])
                v0 = weighted_shrink * (r_slice[1:, 1] - r_slice[:-1, 1])
                plt.quiver(x0, y0, u0, v0, angles='xy', scale=1, scale_units='xy', color='k', alpha=alpha)

            if decorate:
                if self.style_ode in ['PWL3_swap', 'PWL3_zstepdecay']:
                    pp = self.sc_template.params_ode
                    xlow = 0.5 * pp['a1']
                    xhigh = 0.5 * (pp['a1'] + pp['a2'])
                    plt.axvline(xlow, linestyle='--', c='gray')
                    plt.axvline(xhigh, linestyle='--', c='gray')
                    ylow = pp['a1']
                    yhigh = (pp['a1'] - pp['a2'])
                    plt.axhline(ylow, linestyle='-.', c='gray')
                    plt.axhline(yhigh, linestyle='-.', c='gray')

                # plot points in phase space where division event occurred
                events_idx = self.time_indices_where_acted_as_mother(idx)
                star_kwargs = dict(
                    marker='*', edgecolors='k', linewidths=0.5, zorder=10,
                )
                if len(events_idx) > 0:
                    xdiv = state_tensor[0, idx, events_idx]
                    ydiv = state_tensor[1, idx, events_idx]
                    plt.scatter(xdiv, ydiv, c='gold', **star_kwargs)  # immediately after division (xvec partitioned)
                    xdiv = state_tensor[0, idx, events_idx - 1]
                    ydiv = state_tensor[1, idx, events_idx - 1]
                    plt.scatter(xdiv, ydiv, c='green', **star_kwargs)  # immediately before division

            # set labels for axes
            ax.set_xlabel(r'$x_{0}$')
            ax.set_ylabel(r'$x_{1}$')
            cbar = plt.colorbar(sc)
            plt.clim(times[0], times[-1])
            cbar.set_label(r'$t$')

            plt.title('Cell %d trajectory - %s' % (idx, fmod))

            fpath = self.io_dict['dir_plots'] + os.sep + 'traj_cell_%d' % idx
            if fmod is not None:
                fpath += '_' + fmod
            plt.savefig(fpath + '.pdf')
            print(fpath + '.pdf')
            plt.close()
        return

    def plotly_traj(self, fmod=None, write=True, show=True):
        times = self.times_history
        state = self.state_history
        state_tensor = self.state_to_rectangle(state)
        birthdays = self.cell_stats[:, 2]

        column_names = ['cell', 'time_index', 'time'] + ['x%d' % i for i in range(self.sc_dim_ode)]
        df = pd.DataFrame(columns=column_names)
        i = 0
        for cell in range(self.num_cells):
            init_idx = birthdays[cell]
            looptot = len(times) - init_idx
            for idx, t in enumerate(range(init_idx, len(times))):
                row = ['cell%d' % cell,
                       t,
                       times[t]]
                row = row + [state_tensor[i, cell, t] for i in range(self.sc_dim_ode)]
                df.loc[idx + i] = row
            i += looptot

        cmap_list = px.colors.qualitative.Plotly
        fig = make_subplots(rows=self.sc_dim_ode,
                            cols=1,
                            x_title=r'$t$')

        for i in range(self.sc_dim_ode):
            if i == 0:
                showlegend = True
            else:
                showlegend = False
            for cell in range(self.num_cells):
                cell_color = cmap_list[cell % len(cmap_list)]
                fig.append_trace(
                    go.Scatter(
                        x=df[df['cell'] == 'cell%d' % cell]['time'],
                        y=df[df['cell'] == 'cell%d' % cell]['x%d' % i],
                        mode='lines+markers',
                        name='c%d' % (cell),
                        line=dict(color=cell_color),
                        marker_color=cell_color,
                        legendgroup=cell,
                        showlegend=showlegend,
                    ),
                    row=i + 1, col=1)
            fig.update_yaxes(title_text=self.sc_template.variables_short[i], row=i + 1, col=1)

        fig.update_layout(title_text="Cell state trajectories for each variable")

        fpath = self.io_dict['dir_plots'] + os.sep + 'plotly_traj'
        if fmod is not None:
            fpath += '_%s' % fmod
        if write:
            fig.write_html(fpath + '.html', include_mathjax='cdn')
        if show:
            fig.show()
        plt.close()

    def plot_walltimes(self, fmod=''):
        walltime_at_div = np.array(self.walltimes) - self.walltimes[0]
        walltime_between_div = walltime_at_div[1:] - walltime_at_div[:-1]
        x_axis = np.arange(len(walltime_at_div))  # indexes number of divisions

        fig, axarr = plt.subplots(ncols=1, nrows=2, figsize=(8, 8), constrained_layout=True, squeeze=False, sharex=True)
        # Upper plot
        axarr[0, 0].plot(x_axis, walltime_at_div, '--o')
        axarr[0, 0].grid(True, linestyle='--')
        axarr[0, 0].set_ylabel('walltime total (s)')
        # Lower plot
        axarr[1, 0].plot(x_axis[1:], walltime_between_div, '--o')
        axarr[1, 0].grid(True, linestyle='--')
        axarr[1, 0].set_ylabel('walltime between divisions (s)')
        axarr[1, 0].set_xlabel('division #')

        fpath = self.io_dict['dir_plots'] + os.sep + 'walltimes'
        if fmod is not None:
            fpath += '_%s' % fmod
        plt.savefig(fpath + '.pdf')
        print(fpath + '.pdf')
        plt.close()

    def state_to_stacked(self, x):
        """
        converts array x from shape [N x M] to [NM]
        converts array x from shape
         - [N x M]      to  [NM]
         - [N x M x t]  to  [NM x t]

        E.g.: suppose 2 cells each with 2 components
              the first two components belong to cell one, the next two to cell two
              in: [1,2,3,4]   out: [[1, 3],
                                    [2, 4]]
        """
        d = len(x.shape)
        assert d in [2, 3]
        assert x.shape[0:2] == (self.sc_dim_ode, self.num_cells)
        if d == 2:
            out = x.reshape(self.graph_dim_ode, order='F')
        else:
            out = x.reshape((self.graph_dim_ode, -1), order='F')
        return out

    def state_to_rectangle(self, x):
        """
        converts array x from shape
         - [NM]      to  [N x M]
         - [NM x t]  to  [N x M x t]
        """
        d = len(x.shape)
        assert d in [1, 2]
        assert x.shape[0] == self.graph_dim_ode
        if d == 1:
            out = x.reshape((self.sc_dim_ode, self.num_cells), order='F')
        else:
            out = x.reshape((self.sc_dim_ode, self.num_cells, -1), order='F')
        return out

    def write_metadata(self):
        """
        Generates k files:
        - io_dict['runinfo'] - stores metadata on the specific run
        """
        # First file: basedir + os.sep + run_info.txt
        filepath = self.io_dict['runinfo']
        with open(filepath, "a", newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # module choices
            writer.writerow(['style_ode', self.style_ode])
            writer.writerow(['style_dynamics', self.style_dynamics])
            writer.writerow(['style_detection', self.style_detection])
            writer.writerow(['style_division', self.style_division])
            writer.writerow(['style_diffusion', self.style_diffusion])
            # dimensionality
            writer.writerow(['num_cells', self.num_cells])
            writer.writerow(['sc_dim_ode', self.graph_dim_ode])
            writer.writerow(['graph_dim_ode', self.sc_dim_ode])
            # coupling settings
            writer.writerow(['alpha_sharing', self.alpha_sharing])
            writer.writerow(['beta_sharing', self.beta_sharing])
            writer.writerow(['diffusion_arg', self.diffusion_arg])
            writer.writerow(['diffusion', self.diffusion])
            # integration settings
            writer.writerow(['t0', self.t0])
            writer.writerow(['t1', self.t1])
            writer.writerow(['init_cond', self.init_cond])
            writer.writerow(['timeintervalrun', self.timeintervalrun])
            # initialization of each cell
            X = self.state_to_rectangle(self.state_history)
            for cell in range(self.num_cells):
                writer.writerow(['cell_%d' % cell, X[:, cell, -1].flatten()])
            # any single cell dynamics params
        self.sc_template.write_ode_params(filepath)

        return filepath

    def write_state(self, fmod=None, filetype='.txt'):
        """Output current CellGraph state within an HDF5 file OR to separate text files

        Choice depends on hdf5_mode attribute:
        - self.hdf5_mode == True  - virtual HDF5 subfolders via HDF5Handler() custom class
        - self.hdf5_mode == False - text files via numpy

        Args
        - fmod: generally conveys the division index
        - filetype: usually '' or '.txt'
        """
        if self.hdf5_mode:
            filetype = ''
            suffix = fmod
        else:
            if fmod is None:
                suffix = filetype
            else:
                suffix = '_' + fmod + filetype

        def dir_format(s, s_end=suffix):
            if self.hdf5_mode:
                # e.g. 'states/cell_stats/iter17' or 'states/times_history'
                if s_end == '':
                    sout = 'states/%s' % s
                else:
                    sout = 'states/%s/%s' % (s, s_end)
            else:
                # e.g. 'RELATIVE_PATH\\states\\cell_stats_iter17.txt'
                sout = self.io_dict['dir_states'] + os.sep + s + s_end
            return sout

        if self.hdf5_mode:
            h5 = self.hdf5_handler
            # OVERWRITE these files as 'states/name' - they always contain explicit history
            h5.write_dataset(dir_format('labels', s_end=''), np.array(self.labels).astype('S'), rm=True)
            h5.write_dataset(dir_format('times_history', s_end=''), self.times_history, rm=True)
            h5.write_dataset(dir_format('state_history', s_end=''), self.state_history, rm=True)
            h5.write_dataset(dir_format('division_events', s_end=''), self.division_events, rm=True)
            h5.write_dataset(dir_format('walltimes', s_end=''), np.array(self.walltimes), rm=True)
            # ADD these files as 'states/name/iter24' (division idx) - can't infer history for these
            h5.write_dataset(dir_format('adjacency'), self.adjacency)
            h5.write_dataset(dir_format('cell_stats'), self.cell_stats)

        else:
            np.savetxt(dir_format('labels'), self.labels, fmt="%s")
            np.savetxt(dir_format('adjacency'), self.adjacency, fmt="%.4f")
            np.savetxt(dir_format('times_history'), self.times_history, fmt="%.4f")
            np.savetxt(dir_format('state_history'), self.state_history, fmt="%.4f")
            np.savetxt(dir_format('cell_stats'), self.cell_stats, fmt="%d")
            np.savetxt(dir_format('division_events'), self.division_events, fmt="%d")
            np.savetxt(dir_format('walltimes'), self.walltimes, fmt="%.2f")
        return

    def pickle_save(self, fname='classdump.pkl'):
        fpath = self.io_dict['dir_base'] + os.sep + fname
        with open(fpath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        return
