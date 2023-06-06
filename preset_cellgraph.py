import copy
import numpy as np


PRESET_CELLGRAPH = {
    'PWL3_swap_copy': dict(
        num_cells=1,
        style_ode='PWL3_swap',
        style_detection='manual_crossings_1d_mid',
        style_division='copy',
        style_diffusion='xy',
        diffusion_arg=0,
        alpha_sharing=0.1,
        beta_sharing=0.0,
        t0=0,
        t1=600,
        state_history=np.array([[0, 0, 0]]).T,
        verbosity=0,
        mods_params_ode={}
    )
}

# Variant of 'PWL3_swap_copy'
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'] = copy.deepcopy(PRESET_CELLGRAPH['PWL3_swap_copy'])
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']['style_division'] = 'partition_ndiv_bam'

PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all'] = copy.deepcopy(PRESET_CELLGRAPH['PWL3_swap_copy'])
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']['style_division'] = 'partition_ndiv_all'

PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac'] = copy.deepcopy(PRESET_CELLGRAPH['PWL3_swap_copy'])
PRESET_CELLGRAPH['PWL3_swap_partition_bam_fixedfrac']['style_division'] = 'partition_bam_fixedfrac'

PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam'] = copy.deepcopy(PRESET_CELLGRAPH['PWL3_swap_copy'])
PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_bam']['style_division'] = 'plus_minus_delta_bam'

PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam'] = copy.deepcopy(PRESET_CELLGRAPH['PWL3_swap_copy'])
PRESET_CELLGRAPH['PWL3_swap_plus_minus_delta_ndiv_bam']['style_division'] = 'plus_minus_delta_ndiv_bam'


PRESET_CELLGRAPH['waterbear_fig4'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[8.0, 8.0, 0.0],
    alpha_sharing=0.2,
    beta_sharing=0.1,
    t0=0,
    t1=450,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.036,
        t_pulse_switch=150,
    )
)

PRESET_CELLGRAPH['waterbear_fig4_alt'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[4, 4, 0.0],
    alpha_sharing=0.2,
    beta_sharing=0.1,
    t0=0,
    t1=250,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.064,
        t_pulse_switch=82,
    )
)

PRESET_CELLGRAPH['argentine_ant'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[0.30, 0.30, 0.0],
    alpha_sharing=0.008,
    beta_sharing=0.004,
    t0=0,
    t1=400,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.0408,
        t_pulse_switch=150,
    )
)

PRESET_CELLGRAPH['drosophila_5quarter_pint'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[6.1, 6.1, 0.0],
    alpha_sharing=0.016,
    beta_sharing=0.0,
    t0=0,
    t1=250,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.06608,
        t_pulse_switch=82,
    )
)


PRESET_CELLGRAPH['drosophila_3quarter_pint'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[6.1, 6.1, 0],
    alpha_sharing=0.023,
    beta_sharing=0.0,
    t0=0,
    t1=250,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.0832,
        t_pulse_switch=82,
    )
)

PRESET_CELLGRAPH['drosophila_oneshort'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[5.3, 5.3, 0],
    alpha_sharing=-0.02,
    beta_sharing=0.0,
    t0=0,
    t1=250,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.0824,
        t_pulse_switch=82,
    )
)

PRESET_CELLGRAPH['nasonia_vitripennis'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[0.151, 0.151, 0.0],
    alpha_sharing=0.026,
    beta_sharing=0.0,
    t0=0,
    t1=250,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.06128, #0.0613,  #0.06128
        t_pulse_switch=82,
    )
)

PRESET_CELLGRAPH['almost_bee_A'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[0.42, 0.42, 0.0],
    alpha_sharing=-0.005,
    beta_sharing=0.0,
    t0=0,
    t1=400,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.02865823,
        t_pulse_switch=175.0,
    )
)

PRESET_CELLGRAPH['lacewing1'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[2.8, 2.8, 0.0],
    alpha_sharing=0.010,
    beta_sharing=0.0,
    t0=0,
    t1=400,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.09,
        t_pulse_switch=175.0,
    )
)

PRESET_CELLGRAPH['lacewing1_shortpulse'] = copy.deepcopy(PRESET_CELLGRAPH['lacewing1'])
PRESET_CELLGRAPH['lacewing1_shortpulse']['mods_params_ode']['t_pulse_switch'] = 60
PRESET_CELLGRAPH['lacewing1_shortpulse']['t1'] = 130

PRESET_CELLGRAPH['lacewing2'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[2.9, 2.9, 0.0],
    alpha_sharing=-0.010,
    beta_sharing=0.0,
    t0=0,
    t1=400,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.044,
        t_pulse_switch=175.0,
    )
)

PRESET_CELLGRAPH['lacewing_R3'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[2.92, 2.92, 0.0],
    alpha_sharing=0.013,
    beta_sharing=0.0,
    t0=0,
    t1=400,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.0584,
        t_pulse_switch=82.0,
    )
)


"""
Above: settings for PWL3_swap (triangular z(t) pulse)
Below: settings for PWL3_zstepdecay (heaviside rate of increase until t1, with constant exponential decay)
"""

PRESET_CELLGRAPH['PWL3_zstepdecay_default'] = dict(
        num_cells=1,
        style_ode='PWL3_zstepdecay',
        style_detection='manual_crossings_1d_mid',
        style_division='plus_minus_delta_ndiv_bam',
        style_diffusion='xy',
        diffusion_arg=0,
        alpha_sharing=0.1,
        beta_sharing=0.0,
        t0=0,
        t1=450,
        timeintervalrun=1.0,
        verbosity=0,
        mods_params_ode=dict(
            a1=8,
            a2=2,
            epsilon=0.01,
            gamma=0.01,
            dz_stretch=6.0,
            dz_t_heaviside=150.0,
            dz_eta=0.01,
        )
    )

PRESET_CELLGRAPH['PWL3_zdecay_default'] = dict(
        num_cells=1,
        style_ode='PWL3_zstepdecay',
        style_detection='manual_crossings_1d_mid',
        style_division='plus_minus_delta_ndiv_bam',
        style_diffusion='xy',
        diffusion_arg=0,
        alpha_sharing=0.03,
        beta_sharing=0.0,
        t0=0,
        t1=450,
        init_cond=[0, 0, 6.0],
        timeintervalrun=1.0,
        verbosity=0,
        mods_params_ode=dict(
            a1=8,
            a2=2,
            epsilon=0.01,
            gamma=0.01,
            dz_stretch=0.0,
            dz_t_heaviside=0.0,
            dz_eta=0.010,
        )
    )

PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search1'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[8.0, 8.0, 0.0],
    alpha_sharing=0.2,
    beta_sharing=0.1,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=6.1,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search2'] = copy.deepcopy(PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search1'])
PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search2']['mods_params_ode']['dz_eta'] = 0.05
PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search2']['mods_params_ode']['dz_stretch'] = 5.1

PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search3'] = copy.deepcopy(PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search2'])
PRESET_CELLGRAPH['waterbear_fig4_zstepdecay_search3']['mods_params_ode']['dz_stretch'] = 4.25

"""
Below: search for 4star or 4line for PWL3_swap and PWL3_zstepdecay
"""
# for pulse_vel 0.03, alpha=-0.2 is 7star; alpha=-0.3 is 5star; alpha=-0.4 is 4star
PRESET_CELLGRAPH['4star_fig4'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=0.0,
    alpha_sharing=-0.4,
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.03,
        t_pulse_switch=150,
    )
)

PRESET_CELLGRAPH['4line_fig4'] = dict(
    num_cells=1,
    style_ode='PWL3_swap',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=0.0,
    alpha_sharing=0.4,
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        pulse_vel=0.03,
        t_pulse_switch=150,
    )
)

PRESET_CELLGRAPH['4star_fig4_zstepdecay_variantA'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=1.0,
    alpha_sharing=0.2,  # tried -0.2 to start
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=8.5,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

PRESET_CELLGRAPH['4star_fig4_zstepdecay_search'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=0.4,
    alpha_sharing=0.03,  # tried -0.2 to start
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=7.5,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

PRESET_CELLGRAPH['4line_fig4_zstepdecay_search'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[0.0, 0.0, 0.0],
    alpha_sharing=0.2,
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=8,  # 5.4 gives 2cell; 5.5 gives 3cell, 6.0 gives 3cell with high diffusion but 11 with c=0,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

"""
Below: Some additional interesting structures 
- no diffusion in the three below
- no history dependence (beta = 0)
"""

PRESET_CELLGRAPH['13cell_bilinear_chain'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=[0.0, 0.0, 0.0],
    alpha_sharing=0.8,
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=8,  # 5.4 gives 2cell; 5.5 gives 3cell, 6.0 gives 3cell with high diffusion but 11 with c=0,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

PRESET_CELLGRAPH['19cell_trilinear_chain'] = copy.deepcopy(PRESET_CELLGRAPH['13cell_bilinear_chain'])
PRESET_CELLGRAPH['19cell_trilinear_chain']['alpha_sharing'] = 0.4

PRESET_CELLGRAPH['22cell_quadlinear_asymm_chain'] = copy.deepcopy(PRESET_CELLGRAPH['13cell_bilinear_chain'])
PRESET_CELLGRAPH['22cell_quadlinear_asymm_chain']['alpha_sharing'] = 0.2


PRESET_CELLGRAPH['Kstar_zstepdecay_unlimited_divisions'] = dict(
    num_cells=1,
    style_ode='PWL3_zstepdecay',
    style_detection='manual_crossings_1d_mid',
    style_division='plus_minus_delta_ndiv_bam',
    style_diffusion='xy',
    diffusion_arg=0.0,
    alpha_sharing=-0.2,
    beta_sharing=0.0,
    t0=0,
    t1=450,
    timeintervalrun=1.0,
    state_history=np.array([[0, 0, 0]]).T,
    verbosity=0,
    mods_params_ode=dict(
        a1=8,
        a2=2,
        epsilon=0.01,
        gamma=0.01,
        dz_stretch=6.0,
        dz_t_heaviside=150.0,
        dz_eta=0.01,
    )
)

PRESET_CELLGRAPH['Kline_zstepdecay_unlimited_divisions'] = copy.deepcopy(PRESET_CELLGRAPH['Kstar_zstepdecay_unlimited_divisions'])
PRESET_CELLGRAPH['Kline_zstepdecay_unlimited_divisions']['alpha_sharing'] = 0.2
