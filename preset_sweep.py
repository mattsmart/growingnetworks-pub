import copy
import numpy as np

from preset_cellgraph import PRESET_CELLGRAPH
from preset_solver import PRESET_SOLVER


# small utility function for tiling diffusion_arg sweeps
def diffusion_arg_sweep_helper(base_diffusion_vec, idx_to_sweep, linspace_tuple):
	assert len(linspace_tuple) == 3
	base_diffusion_vec = np.array(base_diffusion_vec, dtype=float)
	# 1) diffusion grid should have shape (N x linspace)
	tile_block_shape = (linspace_tuple[2], 1)
	diffusion_grid = np.tile(base_diffusion_vec, tile_block_shape)
	# 2) now set the appropriate elements to the linspace sweep setting
	diffusion_grid[:, idx_to_sweep] = np.linspace(*linspace_tuple)
	return diffusion_grid


# changed solver from default variant to 'strict' variant
SWEEP_SOLVER = PRESET_SOLVER['solve_ivp_radau_strict']['kwargs']
SWEEP_BASE_CELLGRAPH = PRESET_CELLGRAPH['PWL3_swap_copy']
SWEEP_BASE_CELLGRAPH_ZSTEPDECAY = PRESET_CELLGRAPH['PWL3_zstepdecay_default']
SWEEP_BASE_CELLGRAPH_ZDECAY = PRESET_CELLGRAPH['PWL3_zdecay_default']


PRESET_SWEEP = {}

# 1d vel --- scan vel of t_pulse_switch
PRESET_SWEEP['t_pulse_switch_copy'] = dict(
	sweep_label='sweep_preset_t_pulse_switch',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		't_pulse_switch'
	],
	params_values=[
		np.linspace(1, 200, 101)
	],
	params_variety=[
		'sc_ode'
	],
	solver_kwargs=SWEEP_SOLVER
)


PRESET_SWEEP['2d_vel_t_pulse_switch_copy'] = dict(
	sweep_label='sweep_preset_2d_vel_t_pulse_switch',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'pulse_vel',
		't_pulse_switch'
	],
	params_values=[
		np.linspace(0.01, 0.3, 21),
		np.linspace(1, 200, 21)
	],
	params_variety=[
		'sc_ode',
		'sc_ode'
	],
	solver_kwargs=SWEEP_SOLVER
)

# 1d vel --- scan vel of z pulse
PRESET_SWEEP['1d_vel_copy'] = dict(
	sweep_label='sweep_preset_1d_vel',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'pulse_vel'
	],
	params_values=[
		np.linspace(0.0, 0.050, 80)
	],
	params_variety=[
		'sc_ode'
	],
	solver_kwargs=SWEEP_SOLVER
)

# 1d vel --- scan vel of z pulse
PRESET_SWEEP['1d_vel_copy_explore'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_copy_explore']['params_values'] = [np.linspace(0.04, 0.22, 200)]
PRESET_SWEEP['1d_vel_copy_explore']['base_cellgraph_kwargs']['mods_params_ode']['t_pulse_switch'] = 75.0
PRESET_SWEEP['1d_vel_copy_explore']['base_cellgraph_kwargs']['mods_params_ode']['epsilon'] = 1e-6


# Variant of '1d_vel_copy'
PRESET_SWEEP['1d_vel_ndiv_bam'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']

# Variant of '1d_vel_copy'
PRESET_SWEEP['1d_vel_ndiv_all'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_ndiv_all']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']

# Variant of '1d_vel_copy'
PRESET_SWEEP['1d_vel_bam_fixedfrac'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_bam_fixedfrac']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']

# Variant of '1d_vel_copy'
PRESET_SWEEP['1d_vel_plus_minus_delta_bam'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_plus_minus_delta_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']

# Variant of '1d_vel_copy'
PRESET_SWEEP['1d_vel_plus_minus_delta_ndiv_bam'] = copy.deepcopy(PRESET_SWEEP['1d_vel_copy'])
PRESET_SWEEP['1d_vel_plus_minus_delta_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam']

# scan diffusion values
PRESET_SWEEP['1d_diffusion_copy'] = dict(
	sweep_label='sweep_preset_1d_diffusion',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'diffusion_arg'
	],
	params_values=[
		np.linspace(0.44, 0.46, 11)
	],
	params_variety=[
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_ndiv_bam'] = copy.deepcopy(PRESET_SWEEP['1d_diffusion_copy'])
PRESET_SWEEP['1d_diffusion_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_ndiv_all'] = copy.deepcopy(PRESET_SWEEP['1d_diffusion_copy'])
PRESET_SWEEP['1d_diffusion_ndiv_all']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_bam_fixedfrac'] = copy.deepcopy(PRESET_SWEEP['1d_diffusion_copy'])
PRESET_SWEEP['1d_diffusion_bam_fixedfrac']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_plus_minus_delta_bam'] = copy.deepcopy(PRESET_SWEEP['1d_diffusion_copy'])
PRESET_SWEEP['1d_diffusion_plus_minus_delta_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_plus_minus_delta_ndiv_bam'] = copy.deepcopy(PRESET_SWEEP['1d_diffusion_copy'])
PRESET_SWEEP['1d_diffusion_plus_minus_delta_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam']

# 2d vel and diffusion
PRESET_SWEEP['2d_pulse_vel_diffusion'] = dict(
	sweep_label='sweep_preset_2d_pulse_vel_diffusion',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam'],
	params_name=[
		'pulse_vel',
		'diffusion_arg'
	],
	params_values=[
		np.linspace(0.02, 0.15, 80),
		np.linspace(0.00, 8.5, 200)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

"""
- intended for use with:
	(A) alpha_sharing = -0.004 or -0.005, beta = 0.0
	(B) alpha_sharing = 0.0,              beta = 0.004 or 0.005
- note also that 64 cells (but not bee) hit for {c=[0.4698492 and 0.5125628], v=0.02997342, alpha=0, beta=0.004}
	- pulse velocity extents: 0.028 to 0.032
	- diffusion extents: 0 to 0.6
"""
PRESET_SWEEP['2d_pulse_vel_diffusion_beehunt_A'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_diffusion'])
PRESET_SWEEP['2d_pulse_vel_diffusion_beehunt_A']['params_values'] = [
	np.linspace(0.028, 0.032, 40),  # pulse vel (zoomed)
	np.linspace(0.000, 1.000, 40)   # diffusion (zoomed)
]

PRESET_SWEEP['2d_pulse_vel_diffusion_bam_fixedfrac'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_diffusion'])
PRESET_SWEEP['2d_pulse_vel_diffusion_bam_fixedfrac']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']

PRESET_SWEEP['2d_pulse_vel_diffusion_plus_minus_delta_bam'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_diffusion'])
PRESET_SWEEP['2d_pulse_vel_diffusion_plus_minus_delta_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']

# 2d vel and alpha_sharing
PRESET_SWEEP['2d_pulse_vel_alpha_sharing_ndiv_bam'] = dict(
	sweep_label='sweep_preset_2d_pulse_vel_alpha_sharing',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'],
	params_name=[
		'pulse_vel',
		'alpha_sharing'
	],
	params_values=[
		np.linspace(0.0380, 0.0384, 5),
		np.linspace(0.0135, 0.0145, 10)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

PRESET_SWEEP['2d_pulse_vel_alpha_sharing_bam_fixedfrac'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_alpha_sharing_ndiv_bam'])
PRESET_SWEEP['2d_pulse_vel_alpha_sharing_bam_fixedfrac']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']

PRESET_SWEEP['2d_pulse_vel_alpha_sharing_plus_minus_delta_bam'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_alpha_sharing_ndiv_bam'])
PRESET_SWEEP['2d_pulse_vel_alpha_sharing_plus_minus_delta_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']

PRESET_SWEEP['2d_pulse_vel_alpha_sharing_plus_minus_delta_ndiv_bam'] = copy.deepcopy(PRESET_SWEEP['2d_pulse_vel_alpha_sharing_ndiv_bam'])
PRESET_SWEEP['2d_pulse_vel_alpha_sharing_plus_minus_delta_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam']


# 3d vel and diffusion and alpha_sharing
PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_ndiv_bam'] = dict(
	sweep_label='sweep_preset_3d_pulse_vel_diffusion_alpha_sharing',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'],
	params_name=[
		'pulse_vel',
		'diffusion_arg',
		'alpha_sharing'
	],
	params_values=[
		np.linspace(0.01, 0.20, 20),
		np.linspace(0.01, 40, 40),
		np.linspace(-0.10, 0.10, 6)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_bam_fixedfrac'] = copy.deepcopy(PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_ndiv_bam'])
PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_bam_fixedfrac']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']

PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_plus_minus_delta_bam'] = copy.deepcopy(PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_ndiv_bam'])
PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_sharing_plus_minus_delta_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']


# Example - 1D diffusion_arg scan using new vector mode
# scan diffusion_arg vector values
# - uses helper fn -- diffusion_arg_sweep_helper(base vector, index_for_sweep, linspace_args)
# enforce 'all' mode for diffusion style
PRESET_SWEEP['1d_diffusion_copy'] = dict(
	sweep_label='sweep_preset_1d_diffusionComplex',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'diffusion_arg'
	],
	params_values=[
		diffusion_arg_sweep_helper([1.0, 2.0, 0.0], 2, (0.44, 0.46, 11))
	],
	params_variety=[
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)
PRESET_SWEEP['1d_diffusion_copy']['base_cellgraph_kwargs']['style_diffusion'] = 'all'


"""
Below: presets for new single cell ODE and z(t) style - "PWL3_zstepdecay" (indicated using sweep label prefix)
"""
# 2d vel and diffusion
# intended for use with third dimension using disbatch on alpha
PRESET_SWEEP['zstepdecay_2d_dzstretch_diffusion_ndiv_bam'] = dict(
	sweep_label='sweep_preset_zstepdecay_2d_dzstretch_diffusion',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH_ZSTEPDECAY,
	params_name=[
		'dz_stretch',
		'diffusion_arg'
	],
	params_values=[
		np.linspace(5.0, 8.0, 40),
		np.linspace(0.0, 10.0, 40)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)


# 2d vel and diffusion
# intended for use with third dimension using disbatch on alpha
PRESET_SWEEP['zdecay_2d_dz_eta_diffusion'] = dict(
	sweep_label='sweep_preset_zdecay_2d_dz_eta_diffusion',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH_ZDECAY,
	params_name=[
		'dz_eta',
		'diffusion_arg'
	],
	params_values=[
		np.linspace(0.001, 0.025, 80),
		np.linspace(0.0, 20.0, 200)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

# 2d vel and diffusion
# intended for testing cluster submission and benchmarking
PRESET_SWEEP['zstepdecay_2d_testing'] = dict(
	sweep_label='test_sweep_preset_zstepdecay_2d_dzstretch_diffusion',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH_ZSTEPDECAY,
	params_name=[
		'dz_stretch',
		'diffusion_arg'
	],
	params_values=[
		np.linspace(5.0, 8.0, 2),
		np.linspace(0.0, 10.0, 2)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

# 3d vel, diffusion, alpha (asymmetry) - small sweep for local testing
PRESET_SWEEP['3d_pulse_vel_diffusion_alpha_testing'] = dict(
	sweep_label='sweep_preset_3d_pulse_vel_diffusion_asymmetry_testing',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam'],
	params_name=[
		'pulse_vel',
		'diffusion_arg',
		'alpha_sharing'
	],
	params_values=[
		np.linspace(0.02, 0.15, 2),
		np.linspace(0.00, 8.5, 2),
		np.linspace(0.01, 0.02, 3)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)


# 2d vel, t_pulse - small sweep for local testing
PRESET_SWEEP['2d_pulse_vel_duration_testing'] = dict(
	sweep_label='sweep_preset_2d_pulse_vel_duration_testing',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam'],
	params_name=[
		'pulse_vel',
		't_pulse_switch'
	],
	params_values=[
		np.linspace(0.12, 0.15, 4),
		np.linspace(150, 200, 3)
	],
	params_variety=[
		'sc_ode',
		'sc_ode',
	],
	solver_kwargs=SWEEP_SOLVER
)

# 3d alpha, beta, t_pulse - small sweep for local testing
PRESET_SWEEP['3d_alpha_beta_t_pulse_testing'] = dict(
	sweep_label='sweep_preset_3d_alpha_beta_t_pulse_testing',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam'],
	params_name=[
		'alpha_sharing',
		'beta_sharing',
		't_pulse_switch'
	],
	params_values=[
		np.linspace(-0.03, -0.01, 4),
		np.linspace(0.01, 0.03, 3),
		np.linspace(150, 200, 3)
	],
	params_variety=[
		'meta_cellgraph',
		'meta_cellgraph',
		'sc_ode'
	],
	solver_kwargs=SWEEP_SOLVER
)
