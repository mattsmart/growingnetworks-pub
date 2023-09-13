import numpy as np
from numba import jit

from settings import STYLE_ODE_VALID


def set_ode_attributes(style_ode):
    dim_ode = 3  # dimension of ODE system
    dim_misc = 2  # dimension of misc. variables (e.g. fusome content)
    variables_short = {0: 'Cyc_act',
                       1: 'Cyc_tot',
                       2: 'Bam',
                       3: 'n_div',
                       4: 'fusome'}
    variables_long = {0: 'Cyclin active',
                      1: 'Cyclin total',
                      2: 'Modulator, e.g. Bam',
                      3: 'Number of Divisions',
                      4: 'Fusome content'}
    if style_ode == 'Yang2013':
        # currently, all methods share the same attributes above
        pass
    elif style_ode == 'bpj2017':
        dim_ode = 2
        variables_short = {0: 'Cyc_act',
                           1: 'Cyc_tot',
                           2: 'n_div',
                           3: 'fusome'}
        variables_long = {0: 'Cyclin active',
                          1: 'Cyclin total',
                          2: 'Number of Divisions',
                          3: 'Fusome content'}
    if style_ode == 'PWL3_bpj2017':
        # currently, all methods share the same attributes above
        pass
    elif style_ode in ['PWL3', 'PWL3_swap', 'PWL3_zstepdecay']:
        # currently, all methods share the same attributes above
        pass
    elif style_ode == 'PWL2':
        # currently, all methods share the same attributes above
        dim_ode = 2
        variables_short = {0: 'Cyc_act',
                           1: 'Cyc_tot',
                           2: 'n_div',
                           3: 'fusome'}
        variables_long = {0: 'Cyclin active',
                          1: 'Cyclin total',
                          2: 'Number of Divisions',
                          3: 'Fusome content'}
    elif style_ode == 'PWL4':
        # currently, all methods share the same attributes above
        dim_ode = 2
        variables_short = {0: 'Cyc_act',
                           1: 'Cyc_tot',
                           2: 'n_div',
                           3: 'fusome'}
        variables_long = {0: 'Cyclin active',
                          1: 'Cyclin total',
                          2: 'Number of Divisions',
                          3: 'Fusome content'}
    elif style_ode in ['PWL4_auto_ww', 'PWL4_auto_wz', 'PWL4_auto_linear']:
        # currently, all methods share the same attributes above
        dim_ode = 4
        variables_short = {0: 'Cyc_act',
                           1: 'Cyc_tot',
                           2: 'Bam',
                           3: 'Bam_controller',
                           4: 'n_div',
                           5: 'fusome'}
        variables_long = {0: 'Cyclin active',
                          1: 'Cyclin total',
                          2: 'Modulator, e.g. Bam',
                          3: 'Bam controller',
                          4: 'Number of Divisions',
                          5: 'Fusome content'}
    elif style_ode == 'toy_flow':
        dim_ode = 1  # dimension of ODE system
        dim_misc = 0  # dimension of misc. variables (e.g. fusome content)
        variables_short = {0: 'vol'}
        variables_long = {0: 'Water Volume'}
    elif style_ode == 'toy_clock':
        dim_ode = 1  # dimension of ODE system
        dim_misc = 0
        variables_short = {0: 'x'}
        variables_long = {0: 'x=sin(w*t)'}
    else:
        print("Warning: style_ode %s is not supported by set_ode_attributes()" % style_ode)
        print("Supported odes include:", STYLE_ODE_VALID)
    assert len(variables_short.keys()) == len(variables_long.keys())
    assert len(variables_short.keys()) == (dim_ode + dim_misc)
    return dim_ode, dim_misc, variables_short, variables_long


def set_ode_params(style_ode):
    if style_ode == 'Yang2013':
        # reference is Yang2013 Table S1
        p = {
            'k_synth': 1,  # nM / min
            'a_deg': 0.01,  # min^-1
            'b_deg': 0.04,  # min^-1
            'EC50_deg': 32,  # nM
            'n_deg': 17,  # unitless
            'a_Cdc25': 0.16,  # min^-1
            'b_Cdc25': 0.80,  # min^-1
            'EC50_Cdc25': 35,  # nM
            'n_Cdc25': 11,  # unitless
            'a_Wee1': 0.08,  # min^-1
            'b_Wee1': 0.40,  # min^-1
            'EC50_Wee1': 30,  # nM
            'n_Wee1': 3.5,  # unitless
        }
        # add any extra parameters that are separate from Yang2013
        p['Bam_activity'] = 1  # as indicated in SmallCellCluster review draft p7
        p['Bam_deg'] = 0  # degradation rate; arbitrary, try 0 or 1e-2 to 1e-4
    elif style_ode == 'bpj2017':
        # reference is bpj2017 Table 3
        # "The Design Space of the Embryonic Cell Cycle Oscillator"
        p = {
            'a1': 7,  # like time^-1
            'a2': 0.67,  # like time^-1
            'mu': 0.4,  # like time^-1
            'epsilon': 0.001,  # like time
            'n1': 10,    # unitless
            'n2': 10,  # unitless
            'gamma1': 0.2,  # concentration
            'gamma2': 0.2,  # concentration
        }
        # add any extra parameters that are separate from Yang2013
        p['Bam_activity'] = 1  # as indicated in SmallCellCluster review draft p7
        p['Bam_deg'] = 0  # degradation rate; arbitrary, try 0 or 1e-2 to 1e-4
    elif style_ode == 'PWL3_bpj2017':
        p = {
            'a1': 5,                 # for g(x)
            'a2': 1,                 # for g(x)
            'b1': 10,                # for h(x)
            'c1': 0,                # for h(x)
            'gamma': 1e-2,           # for h(x)
            'epsilon': 1e-2,         # speed scale for fast variable Cyc_act
            'pulse_vel': 0.3,        # z pulse - rate of inhibitor accumulation
            'I_initial': 0,          # z pulse - initial inhibitor (e.g. Bam) concentration
            't_pulse_switch': 25.0   # z pulse - treating inhibitor timeseries as pulse with a negative slope from t=T to t=2T
        }
        # add any extra parameters that are separate from Yang2013
        p['Bam_activity'] = 1  # as indicated in SmallCellCluster review draft p7
        p['Bam_deg'] = 0  # degradation rate; arbitrary, try 0 or 1e-2 to 1e-4
    elif style_ode in ['PWL2', 'PWL3']:
        """ Notes from Hayden slide 12:
        - ((1−𝛾))/2𝜀𝛾 is duration that green intersects red between extrema of red
        - the free params are 𝑎, 𝐼, 𝜀𝛾/(1+𝛾), b
        - Maybe specify conditions on 𝛾
        """
        p = {
            'epsilon': 1e-1,        # speed scale for fast variable Cyc_act
            'a1': 2,                # defines the corners of PWL function for x
            'a2': 1,                # defines the corners of PWL function for x
            'b': 2,                 # defines the y-intercept of the dy/dt=0 nullcline, y(x) = 1/gamma * (-x + b)
            'gamma': 1e-1,          # degradation of Cyc_tot
            'pulse_vel': 0.3,       # rate of inhibitor accumulation
            'I_initial': 0,         # initial inhibitor (e.g. Bam) concentration
            't_pulse_switch': 25.0  # treating inhibitor timeseries as pulse with a negative slope from t=T to t=2T
        }
        assert 0 < p['epsilon'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['pulse_vel']
    elif style_ode == 'PWL3_swap':
        p = {
            'epsilon': 1e-2,        # speed scale for fast variable Cyc_act
            'a1': 1.0,              # defines the corners of PWL function for x -- this must be 1.0 to enforce normalization in paper
            'a2': 0.25,             # defines the corners of PWL function for x -- this is "a" in the paper
            'gamma': 1e-2,          # degradation of Cyc_tot
            'pulse_vel': 0.2,       # rate of inhibitor accumulation            -- -- this is "v" in the paper
            't_pulse_switch': 25.0  # treating inhibitor timeseries as pulse with a negative slope from t=T to t=2T
        }
        assert 0 < p['epsilon'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['pulse_vel']
        assert p['a1'] == 1.0
    elif style_ode == 'PWL3_zstepdecay':
        p = {
            'epsilon': 1e-2,         # speed scale for fast variable Cyc_act
            'a1': 8,                 # defines the corners of PWL function for x
            'a2': 2,                 # defines the corners of PWL function for x
            'gamma': 1e-2,           # degradation of Cyc_tot
            'dz_stretch': 6.0,       # scaling parameter for z(t)
            'dz_eta': 0.01,          # shape parameter for z(t); low eta is shark fin, high eta is stretched heaviside
            'dz_t_heaviside': 150.0  # time at witch the step input ends
        }
        assert 0 < p['epsilon'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['dz_stretch']
        assert 0 < p['dz_eta']
    elif style_ode in ['PWL4_auto_wz', 'PWL4_auto_ww']:
        p = {
            'epsilon': 1e-1,        # speed scale for fast variable Cyc_act
            'a1': 4,                # defines the corners of PWL function for x
            'a2': 2,                # defines the corners of PWL function for x
            'gamma': 1e-1,          # degradation of Cyc_tot
            'delta_w': 0.1,         # defines degradation rate of bam controller, w(t)
            'w_threshold': 0.5,     # defines threshold at which w(t) produces (above w1) or destroys (below w1) Bam
        }
        assert 0 < p['epsilon'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['delta_w']
        assert 0 <= p['w_threshold'] <= 1.0  # should be below the init cond of w(t) - generally 1.0, but above 0.0
    elif style_ode == 'PWL4_auto_linear':
        p = {
            'epsilon': 1e-1,        # speed scale for fast variable Cyc_act
            'a1': 2,                # defines the corners of PWL function for x
            'a2': 1,                # defines the corners of PWL function for x
            'gamma': 1e-1,          # degradation of Cyc_tot
            'pulse_vel': 1,         # rate of inhibitor accumulation (via dzdt += epsilon * w(t))
            'delta_w': 0.1,         # defines degradation rate of bam controller, w(t)
            'w_threshold': 0,       # [Not needed] defines threshold at which w(t) produces (above w1) or destroys (below w1) Bam
            'b_Bam': 0,             # [Not needed] constant production of Bam
        }
        assert 0 < p['epsilon'] < 1
        assert 0 < p['gamma']
        assert 0 <= p['delta_w']
        assert 0 <= p['w_threshold'] <= 1.0  # should be below the init cond of w(t) - generally 1.0, but above 0.0
    elif style_ode == 'toy_flow':
        p = {}
    elif style_ode == 'toy_clock':
        p = {'w': 2*np.pi}
    else:
        print("Warning: style_ode %s is not supported by get_params_ODE()" % style_ode)
        print("Supported odes include:", STYLE_ODE_VALID)
        p = {}
    return p


def set_ode_vectorfield(style_ode, params, init_cond, **ode_kwargs):
    """
    Returns
        dxdt, which is the output of vector field function call
    """
    if style_ode == 'Yang2013':
        dxdt = vectorfield_Yang2013(init_cond, params, ode_kwargs.get('t', 0),
                                    z=ode_kwargs.get('z', 0))
    elif style_ode == 'bpj2017':
        dxdt = vectorfield_bpj2017(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL3_bpj2017':
        dxdt = vectorfield_PWL3_bpj2017(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL2':
        dxdt = vectorfield_PWL2(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL3':
        dxdt = vectorfield_PWL3(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL3_swap':
        dxdt = vectorfield_PWL3_swap(init_cond,
                                     ode_kwargs.get('t', 0),
                                     params['pulse_vel'],
                                     params['t_pulse_switch'],
                                     params['a1'],
                                     params['a2'],
                                     params['epsilon'],
                                     params['gamma'])
    elif style_ode == 'PWL3_zstepdecay':
        dxdt = vectorfield_PWL3_zstepdecay(init_cond,
                                           ode_kwargs.get('t', 0),
                                           params['dz_stretch'],
                                           params['dz_t_heaviside'],
                                           params['dz_eta'],
                                           params['a1'],
                                           params['a2'],
                                           params['epsilon'],
                                           params['gamma'])
    elif style_ode == 'PWL4_auto_ww':
        dxdt = vectorfield_PWL4_autonomous_ww(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL4_auto_wz':
        dxdt = vectorfield_PWL4_autonomous_wz(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'PWL4_auto_linear':
        dxdt = vectorfield_PWL4_autonomous_linear(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'toy_flow':
        dxdt = vectorfield_toy_flow(init_cond, params, ode_kwargs.get('t', 0))
    elif style_ode == 'toy_clock':
        dxdt = vectorfield_toy_clock(init_cond, params, ode_kwargs.get('t', 0))
    else:
        print("Warning: style_ode %s is not supported by set_ode_vectorfield()" % style_ode)
        print("Supported odes include:", STYLE_ODE_VALID)
        dxdt = None
    return dxdt


def pointer_ode_vectorfield(style_ode):
    """
    Returns
        dxdt, which is the output of vector field function call
    """
    if style_ode == 'Yang2013':
        fn = vectorfield_Yang2013
    elif style_ode == 'bpj2017':
        fn = vectorfield_bpj2017
    elif style_ode == 'PWL3_bpj2017':
        fn = vectorfield_PWL3_bpj2017
    elif style_ode == 'PWL2':
        fn = vectorfield_PWL2
    elif style_ode == 'PWL3':
        fn = vectorfield_PWL3
    elif style_ode == 'PWL3_swap':
        fn = vectorfield_PWL3_swap
    elif style_ode == 'PWL3_zstepdecay':
        fn = vectorfield_PWL3_zstepdecay
    elif style_ode == 'PWL4_auto_ww':
        fn = vectorfield_PWL4_autonomous_ww
    elif style_ode == 'PWL4_auto_wz':
        fn = vectorfield_PWL4_autonomous_wz
    elif style_ode == 'PWL4_auto_linear':
        fn = vectorfield_PWL4_autonomous_linear
    elif style_ode == 'toy_flow':
        fn = vectorfield_toy_flow
    elif style_ode == 'toy_clock':
        fn = vectorfield_toy_clock
    else:
        print("Warning: style_ode %s is not supported by set_ode_vectorfield()" % style_ode)
        print("Supported odes include:", STYLE_ODE_VALID)
        fn = None
    return fn


def set_ode_jacobian(style_ode):
    """
    Returns
        jac, which is pointer to fn or None
    """
    if style_ode == 'PWL3_swap':
        jac = jacobian_PWL3_swap
    else:
        jac = None
    return jac


def ode_integration_defaults(style_ode):
    t0 = 0.0
    if style_ode == 'Yang2013':
        t1 = 800
        num_steps = 2000
        init_cond = [60.0, 0.0, 0.0]
    elif style_ode == 'bpj2017':
        t1 = 10
        num_steps = 2000
        init_cond = [60.0, 0.0, 0.0]
    elif style_ode == 'PWL3_bpj2017':
        t1 = 100
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0]
    elif style_ode == 'PWL2':
        t1 = 50
        num_steps = 2000
        init_cond = [1.0, 1.0]
    elif style_ode == 'PWL3':
        t1 = 50
        num_steps = 2000
        init_cond = [1.0, 1.0, 0.0]
    elif style_ode == 'PWL3_swap':
        t1 = 200
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0]
    elif style_ode == 'PWL3_zstepdecay':
        t1 = 200
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0]
    elif style_ode == 'PWL4_auto_ww':
        t1 = 50
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0, 1.0]   # fourth component is initial condition of w(t) aka key "w0" parameter
    elif style_ode == 'PWL4_auto_wz':
        t1 = 50
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0, 1.0]   # fourth component is initial condition of w(t) aka key "w0" parameter
    elif style_ode == 'PWL4_auto_linear':
        t1 = 100 #50
        num_steps = 2000
        init_cond = [0.0, 0.0, 0.0, 2.0]   # fourth component is initial condition of w(t) aka key "w0" parameter
    elif style_ode == 'toy_flow':
        t1 = 50
        num_steps = 2000
        init_cond = [100.0]
    elif style_ode == 'toy_clock':
        t1 = 3.4
        num_steps = 2000
        init_cond = [0.0]
    else:
        print("Warning: style_ode %s is not supported by ode_integration_defaults()" % style_ode)
        print("Supported odes include:", STYLE_ODE_VALID)
        t1 = None
        num_steps = None
        init_cond = None
    return t0, t1, num_steps, init_cond


def vectorfield_toy_flow(init_cond, params, t):
    return [0]


def vectorfield_toy_clock(init_cond, params, t):
    w = params['w']
    dxdt = w * np.cos(w * t)
    return [dxdt]


def vectorfield_Yang2013(init_cond, params, t, z=0):
    """
    Args:
        params - dictionary of ODE parameters used by Yang2013
        x - array-like
        y - array-like
        z - array-like
    Returns:
        array like of shape [x, y] or [x, y, z] depending on two_dim flag
    """
    p = params
    x, y, _ = init_cond  # TODO note z is passed through init_cond but is unused; use static "external" z for now

    # "f(x)" factor of the review - degradation
    # TODO care if x = 0 -- add case?
    r_degradation = (p['EC50_deg'] / x) ** p['n_deg']
    degradation = p['a_deg'] + p['b_deg'] / (1 + r_degradation)
    degradation_scaled = degradation / (1 + z / p['Bam_activity'])  # as in p7 of SmallCellCluster Review draft

    # "g(x)" factor of the review - activation by Cdc25
    # TODO care if x = 0 -- add case?
    r_activation = (p['EC50_Cdc25'] / x) ** p['n_Cdc25']
    activation = p['a_Cdc25'] + p['b_Cdc25'] / (1 + r_activation)

    # "k_i" factor of the review - de-activation by Wee1
    r_deactivation_inv = (x / p['EC50_Wee1']) ** p['n_Wee1']
    deactivation = p['a_Wee1'] + p['b_Wee1'] / (1 + r_deactivation_inv)

    dxdt = p['k_synth'] - degradation_scaled * x + activation * (y - x) - deactivation * x
    dydt = p['k_synth'] - degradation_scaled * y
    #dzdt = np.zeros_like(dxdt)
    dzdt = -p['Bam_deg'] * z

    out = [dxdt, dydt, dzdt]
    return out


def vectorfield_bpj2017(init_cond, params, t):
    p = params
    x, y = init_cond  # TODO note z is passed through init_cond but is unused; use static "external" z for now

    ratio_pow_1 = (p['gamma1'] / x) ** p['n1']
    ratio_pow_2 = (p['gamma2'] / x) ** p['n2']

    A = 1 + p['a1'] * (1 / (ratio_pow_1 + 1))
    Q = (p['mu'] + p['a2'] * (1 / (ratio_pow_2 + 1))) * (y-x) - x

    dxdt = 1 - A*x + 1/(p['epsilon']) * Q
    dydt = 1 - A*y

    out = [dxdt, dydt]
    return out


def vectorfield_PWL3_bpj2017(init_cond, params, t):
    x, y, z = init_cond

    derivative_I_of_t = PWL_derivative_I_of_t_pulse(z, t, params['pulse_vel'], params['t_pulse_switch'])
    g_of_x = PWL_g_of_x(x, params['a1'], params['a2'])
    h_of_x_minus_z = PWL_h_of_x(x-z, params['b1'], params['c1'], params['gamma'])

    dxdt = 1/params['epsilon'] * (y - g_of_x)
    dydt = h_of_x_minus_z - y
    dzdt = derivative_I_of_t * np.ones_like(dxdt)  # second factor for vectorization support

    out = [dxdt, dydt, dzdt]
    return out


def PWL_g_of_x_SCALAR(params, x):
    """
    Currently unused; see vectorized variant PWL_g_of_x()
    - Like PWL_g_of_x but assumes a2 = 1
    - Not vectorized
    """
    a = params['a']
    if x < (a/2):
        g = -2*x
    elif x <= ((1+a)/2):
        g = 2 * (x - a)
    else:
        g = 2 * (1 - x)
    return g


# default decorator is: @jit(nopython=True)
#@jit
@jit(nopython=True)
#@jit(nopython=True, fastmath=True)
def PWL_g_of_x(x, a1, a2):
    # Note: generally assume a2 = 1
    g1 = np.where(x < a1/2,
                  2 * x,
                  0)
    g2 = np.where(((a1/2) <= x) & (x < ((a2 + a1)/2)),
                  2 * (-x + a1),
                  0)
    g3 = np.where(x >= ((a2 + a1)/2),
                  2 * (-a2 + x),
                  0)
    g = g1 + g2 + g3
    return g


@jit(nopython=True)
def PWL_h_of_x(x, b1, c1, gamma):
    h1 = np.where(x < c1, b1, 0)
    c2 = c1 + gamma * b1
    h2 = np.where(
        (c1 <= x) & (x < c1 + c2),
        b1 + (c1-x) / gamma, 0)
    #h3 = np.where(x >= ((d+a)/2), -d + x, 0)  # don't need h3 case because output always zero
    h = h1 + h2  # + h3                        # don't need h3 case because output always zero
    return h


def PWL_g_of_x_derivative(x, a1, a2):
    g1 = np.where(x < a1/2, 2, 0)
    g2 = np.where(
        ((a1/2) <= x) & (x < ((a1+a2)/2)),
        -2, 0)
    g3 = np.where(x >= ((a1+a2)/2), 2, 0)
    g = g1 + g2 + g3
    return g


def PWL_I_of_t_pulse(t, I_initial, vel, t_half):
    """
    Generates a triangular pulse rising at t=0 with switch at t = params['t_pulse_switch']
      when t > 2 * params['t_pulse_switch'], there is no further change
    """
    if t < t_half:
        I = I_initial + vel * t
    elif t < 2 * t_half:
        I = I_initial - vel * t + 2 * vel * t_half
    else:
        I = I_initial
    assert I >= 0
    assert t >= 0
    return I


# Currently, @jit(nopython=True) gives ~20% speedup in total runtime (all vectorization moved here)
@jit(nopython=True)
def PWL_derivative_I_of_t_pulse(z, t, vel, t_half, eps=1e-6):
    """
    Generates a triangular pulse rising at t=0 with switch at t = params['t_pulse_switch']
      when t > 2 * params['t_pulse_switch'], there is no further change
    """
    assert t >= 0
    if t < t_half:
        dIdt = vel * np.ones_like(z)
    elif t < 2 * t_half:
        dIdt = np.where(z <= eps, 0, -vel)
    else:
        dIdt = np.zeros_like(z)
    return dIdt


# Currently, @jit(nopython=True) gives ~20% speedup in total runtime (all vectorization moved here)
@jit(nopython=True)
def dzdt_step_with_decay(z, t, param_dz_stretch, param_dz_t_heaviside, param_dz_eta):
    """
    param_stretch:      below = v
    param_t_heaviside:  below = t1
    param_eta:          below = eta

    Functional form:
        eta^(-1) dz/dt = v h(t) - z
            - h(t) = u(t) - u(t-t1) is a difference of step functions u(t-a), giving a pulse ending at t1

    See: mathematica/z_of_t_forms.nb to see how three parameters control things
    - eta ~0.05   gives "shark fin" (sensitive to changes in eta here); height ~<= param_stretch (aka pulse_vel)
    - eta >~ 1.0  gives step from 0 to t1 with height ~= param_stretch (aka pulse_vel)
    - note the exact functional form of z(t) is known, so we can anticipate the oscillatory entrance/exit etc
    """
    assert t >= 0

    # Form #1 (check speed)
    if t < param_dz_t_heaviside:
        dzdt = param_dz_stretch * np.ones_like(z)
    else:
        dzdt = np.zeros_like(z)

    dzdt = dzdt - z
    dzdt = param_dz_eta * dzdt

    return dzdt


def vectorfield_PWL2(init_cond, params, t):
    """
    Originally from slide 12 of Hayden ppt
    - Change #1: here the variables are relabelled (based on Jan 18 discussion)
        x = -1 * v
        y = w
        - note x only degrades in the intermediate regime of PWL g(x) now, because of the relabelling
    - Change #2:
        Need to shift the y-nullcline to the right in order to have positive stable states
        Use new parameter "b" which is basal rate of y production
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y]
    """
    x, y = init_cond  # TODO note z is passed through init_cond but is unused; use static "external" z for now

    I_of_t = PWL_I_of_t_pulse(t, params['I_initial'], params['pulse_vel'], params['t_pulse_switch'])
    g_of_x = PWL_g_of_x(x, params['a1'], params['a2'])

    dxdt = 1/params['epsilon'] * (y - g_of_x - I_of_t)
    dydt = params['b'] - x - params['gamma'] * y

    out = [dxdt, dydt]
    return out


def vectorfield_PWL3(init_cond, params, t):
    """
    3-dim variant of PWL2 where the modulator I(t), here z, has own differential equation
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y, z]
    """
    x, y, z = init_cond

    derivative_I_of_t = PWL_derivative_I_of_t_pulse(z, t, params['pulse_vel'], params['t_pulse_switch'])
    g_of_x = PWL_g_of_x(x, params['a1'], params['a2'])

    dxdt = 1/params['epsilon'] * (y - g_of_x - z)
    dydt = params['b'] - x - params['gamma'] * y
    dzdt = derivative_I_of_t

    #dzdt = -p['Bam_deg'] * z

    out = [dxdt, dydt, dzdt]
    return out


# Aug 1, 2022 - @jit(nopython=True) decorator helps now (need to pass params explicitly though, no dict allowed)
@jit(nopython=True)
def vectorfield_PWL3_swap(init_cond, t, pulse_vel, pulse_t_switch, a1, a2, epsilon, gamma):
    """
    3-dim variant of PWL2 where the modulator I(t) now affects the y nullcline (slides its y intercept)
    Args:
        x - array-like
        y - array-like
        z - array-like
        t - time corresponding to integration variable (non-autonomous system)
        params - dictionary of ODE parameters used by piecewise linear ODE system
    Returns:
        array like of shape [x, y, z]

    Notes for vectorization in scipy ODE solver:
    - if vectorized, init_cond will be array of shape (N,1) or (N,N) -- the second one happens only occasionally
    - if NOT vectorized, init_cond will be array of shape (N)
    """
    # Handler: vectorization cases
    # TODO keep in mind this "PWL3_swap" is just one of many dx/dt = f(x) "right-hand-side" functions
    #  consider alternative A: code two separate f(x) for vectorized/not cases - faster?, but don't want to do that
    #  consider alternative B: move dimension handling higher up in scope, code all f(x) to assume vectorized - slower?
    if init_cond.ndim == 1:
        # Below is general format, but numba complains because of non-contiguous array (need to mandate higher in scope)
        # init_cond = init_cond.reshape(init_cond.shape[0], 1)  # numba doesn't like this
        init_cond_alt = np.expand_dims(init_cond, axis=1)       # convert init_cond to a Nx1 array (i.e. ndim == 2 now)
        x = init_cond_alt[0, :]
        y = init_cond_alt[1, :]
        z = init_cond_alt[2, :]
    else:
        x = init_cond[0, :]
        y = init_cond[1, :]
        z = init_cond[2, :]

    derivative_I_of_t = PWL_derivative_I_of_t_pulse(z, t, pulse_vel, pulse_t_switch)
    g_of_x = PWL_g_of_x(x, a1, a2)

    dxdt = 1/epsilon * (y - g_of_x)
    dydt = z - x - gamma * y
    dzdt = derivative_I_of_t  # note, already vectorized in the function call (and @jit)

    #dzdt = -p['Bam_deg'] * z

    out = [dxdt, dydt, dzdt]
    return out


def jacobian_PWL3_swap(t, init_cond, singlecell):
    params = singlecell.params_ode
    df1_dx = PWL_g_of_x_derivative(init_cond[0], params['a1'], params['a2'])
    df1_dy = 1/params['epsilon']
    df2_dx = -1.0
    df2_dy = -params['gamma']
    df2_dz = 1.0
    df3_dz = 0.0  # note it's zero because the "external pulse" is not a function of z
    jac = np.array([
        [df1_dx, df1_dy, 0],
        [df2_dx, df2_dy, df2_dz],
        [0,      0,      df3_dz]
    ])
    return jac


# Aug 1, 2022 - @jit(nopython=True) decorator helps now (need to pass params explicitly though, no dict allowed)
@jit(nopython=True)
def vectorfield_PWL3_zstepdecay(init_cond, t, dz_stretch, dz_t_heaviside, dz_eta, a1, a2, epsilon, gamma):
    """
    3-dim variant of PWL2 where the modulator I(t) now affects the y nullcline (slides its y intercept)
    Args:
        x - array-like
        y - array-like
        z - array-like
        t - time corresponding to integration variable (non-autonomous system)
        params - dictionary of ODE parameters used by piecewise linear ODE system
    Returns:
        array like of shape [x, y, z]

    Notes for vectorization in scipy ODE solver:
    - if vectorized, init_cond will be array of shape (N,1) or (N,N) -- the second one happens only occasionally
    - if NOT vectorized, init_cond will be array of shape (N)
    """
    # Handler: vectorization cases
    # TODO keep in mind this "PWL3_swap" is just one of many dx/dt = f(x) "right-hand-side" functions
    #  consider alternative A: code two separate f(x) for vectorized/not cases - faster?, but don't want to do that
    #  consider alternative B: move dimension handling higher up in scope, code all f(x) to assume vectorized - slower?
    if init_cond.ndim == 1:
        # Below is general format, but numba complains because of non-contiguous array (need to mandate higher in scope)
        # init_cond = init_cond.reshape(init_cond.shape[0], 1)  # numba doesn't like this
        init_cond_alt = np.expand_dims(init_cond, axis=1)       # convert init_cond to a Nx1 array (i.e. ndim == 2 now)
        x = init_cond_alt[0, :]
        y = init_cond_alt[1, :]
        z = init_cond_alt[2, :]
    else:
        x = init_cond[0, :]
        y = init_cond[1, :]
        z = init_cond[2, :]

    dzdt_output = dzdt_step_with_decay(z, t, dz_stretch, dz_t_heaviside, dz_eta)
    g_of_x = PWL_g_of_x(x, a1, a2)

    dxdt = 1/epsilon * (y - g_of_x)
    dydt = z - x - gamma * y
    dzdt = dzdt_output  # note, already vectorized in the function call (and @jit)

    out = [dxdt, dydt, dzdt]
    return out


# don't need for now
def PWL4_auto_helper(params, x, y, z, w):
    """
    Common steps used by PWL4 autonomous vectorfields (note auto means autonomous)
    - the "dzdt" equation is what differs between the various PWL4 autonomous vectorfields
    """
    g_of_x = PWL_g_of_x(x, params['a1'], params['a2'])
    dxdt = 1/params['epsilon'] * (y - g_of_x)
    dydt = z - x - params['gamma'] * y
    dwdt = -params['delta_w'] * w
    return dxdt, dydt, dwdt


# don't need for now
def vectorfield_PWL4_autonomous_ww(init_cond, params, t):
    """
    4-dim variant of PWL3_swap where the modulator Bam is now autonomous and controlled by 4th parameter w
    Notes:
        - dwdt can be integrated independently to get w(t)    -- analytically
        - dzdt can be integrated given w(t), givng z(t)       -- analytically
        - dxdt depends on x, y
        - dydt depends on x, y, z
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        w - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y, z, w]
    """
    x, y, z, w = init_cond
    dxdt, dydt, dwdt = PWL4_auto_helper(params, x, y, z, w)

    dzdt = w * (w - params['w_threshold'])

    out = [dxdt, dydt, dzdt, dwdt]
    return out


# don't need for now
def vectorfield_PWL4_autonomous_wz(init_cond, params, t):
    """
    4-dim variant of PWL3_swap where the modulator Bam is now autonomous and controlled by 4th parameter w
    Notes:
        - dwdt can be integrated independently to get w(t)    -- analytically
        - dzdt can be integrated given w(t), giving z(t)      -- analytically (see mathematica)
        - dxdt depends on x, y
        - dydt depends on x, y, z
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        w - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y, z, w]
    """
    x, y, z, w = init_cond
    dxdt, dydt, dwdt = PWL4_auto_helper(params, x, y, z, w)

    dzdt = z * (w - params['w_threshold'])

    out = [dxdt, dydt, dzdt, dwdt]

    return out


# don't need for now
def vectorfield_PWL4_autonomous_linear(init_cond, params, t):
    """
    4-dim variant of PWL3_swap where the modulator Bam is now autonomous and controlled by 4th parameter w
    Notes:
        - dwdt can be integrated independently to get w(t)    -- analytically
        - dzdt can be integrated given w(t), giving z(t)      -- analytically (see mathematica)
        - dxdt depends on x, y
        - dydt depends on x, y, z
    Args:
        params - dictionary of ODE parameters used by piecewise linear ODE system
        x - array-like
        y - array-like
        z - array-like
        w - array-like
        t - time corresponding to integration variable (non-autonomous system)
    Returns:
        array like of shape [x, y, z, w]
    """
    x, y, z, w = init_cond
    dxdt, dydt, dwdt = PWL4_auto_helper(params, x, y, z, w)

    dzdt = - z + params['b_Bam'] + params['pulse_vel'] * (w - params['w_threshold'])

    out = [dxdt, dydt, dzdt, dwdt]
    return out
