"""
Sets project-wide defaults 
"""

import os
import sys
import platform


# If code is running on Windows or Mac, set IS_RUNNING_ON_CLUSTER = False
IS_RUNNING_ON_CLUSTER = True
if platform.system() in ['Windows', 'Darwin']:
    IS_RUNNING_ON_CLUSTER = False

# IO: Default directories
OSCILLATOR_NETWORKS = os.path.dirname(__file__)
if IS_RUNNING_ON_CLUSTER:
    username = os.path.split(os.environ['HOME'])[-1]
    DIR_RUNS = '/mnt/ceph/users/%s/runs' % username
else:
    DIR_RUNS = 'runs'      # store timestamped runs here
DIR_OUTPUT = 'output'  # misc output like simple plots
if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

# Append project to path
sys.path.append(OSCILLATOR_NETWORKS)

# HDF5 settings which control CellGraph run output
HDF5_MODE = True          # if True, store state info at each division in an HDF5 file instead of many .txt
HDF5_VERBOSE = False      # attribute of HDF5Handler() class
HDF5_COMPRESS = "lzf"     # compress datasets (slight speed hit but big disk savings); can set to None
HDF5_COMPRESS_LVL = None  # lzf: must be None | gzip: compress lvl int 0 to 9; details here https://docs.h5py.org/en/stable/high/dataset.html
# Cluster issue:
#  BlockingIOError: [Errno 11] Unable to open file
#    (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"  # workaround for locking issue on cluster

# PLOTTING
PLOT_XLABEL = r'Cyc$_{act}$'
PLOT_YLABEL = r'Cyc$_{tot}$'

# DEFAULTS: module 0 - Integrating the ODE trajectory
STYLE_DYNAMICS = 'solve_ivp'
STYLE_DYNAMICS_VALID = ['solve_ivp', 'numba_lsoda', 'diffeqpy', 'libcall', 'rk4', 'euler']

# DEFAULTS: module 1 - Single cell dynamical system
STYLE_ODE = 'PWL3_swap'
STYLE_ODE_VALID = ['Yang2013', 'bpj2017', 'PWL3_bpj2017',
                   'PWL2', 'PWL3', 'PWL3_swap', 'PWL3_zstepdecay',
                   'PWL4_auto_wz', 'PWL4_auto_ww', 'PWL4_auto_linear',
                   'toy_flow', 'toy_clock']

# DEFAULTS: module 2 - Oscillation detection
STYLE_DETECTION = 'manual_crossings_1d_mid'
STYLE_DETECTION_VALID = ['ignore', 'scipy_peaks', 'manual_crossings_1d_mid', 'manual_crossings_1d_hl', 'manual_crossings_2d']

# DEFAULTS: module 3 - Coupled cell graph dynamical system - division rules
STYLE_DIVISION = 'plus_minus_delta_ndiv_bam'
STYLE_DIVISION_VALID = ['copy', 'partition_equal', 'partition_ndiv_all', 'partition_ndiv_bam',
                        'partition_bam_fixedfrac', 'plus_minus_delta_bam', 'plus_minus_delta_ndiv_bam']


# DEFAULTS: module 4 - Coupled cell graph dynamical system - diffusion rules
DIFFUSION_ARG = 1.0  # default diffusion_arg attribute of cellgraph; induces the diffusion vector attribute
STYLE_DIFFUSION = 'xy'
STYLE_DIFFUSION_VALID = ['all', 'xy']

# Set default alpha sharing parameter to none -- force user to specify.
ALPHA_SHARING = None
BETA_SHARING = None

# Parameter sweep settings
SWEEP_VARIETY_VALID = ['meta_cellgraph', 'sc_ode']
