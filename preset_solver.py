PRESET_SOLVER = {}

vectorized = True  # for the scipy integrators, can pass vectorization flag for faster jacobian construction

PRESET_SOLVER['solve_ivp_radau_default'] = dict(
    label='solve_ivp_radau_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-6, rtol=1e-3, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_strictest'] = dict(
    label='solve_ivp_radau_strict',
    dynamics_method='solve_ivp',
    vectorized=vectorized,
    kwargs=dict(method='Radau', t_eval=None, atol=1e-14, rtol=1e-14, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_strict'] = dict(
    label='solve_ivp_radau_strict',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-8, rtol=1e-4, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_medium'] = dict(
    label='solve_ivp_radau_relaxed',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-6, rtol=1e-4, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_relaxed'] = dict(
    label='solve_ivp_radau_relaxed',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', t_eval=None, atol=1e-5, rtol=1e-2, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_radau_minstep'] = dict(
    label='solve_ivp_radau_minstep',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', min_step=1e-1, vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_BDF_default'] = dict(
    label='solve_ivp_BDF_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='BDF', vectorized=vectorized),
)

PRESET_SOLVER['solve_ivp_LSODA_default'] = dict(
    label='solve_ivp_LSODA_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='LSODA', vectorized=vectorized),
)

# TODO - not implemented
PRESET_SOLVER['diffeqpy_default'] = dict(
    label='diffeqpy_default',
    dynamics_method='diffeqpy',
    kwargs=dict(abstol=1e-8, reltol=1e-4),  # assumes RadauIIA5 solver for now
)

# TODO - not implemented
PRESET_SOLVER['numba_lsoda'] = dict(
    label='numba_lsoda',
    dynamics_method='numba_lsoda',
    kwargs=dict(atol=1e-8, rtol=1e-4),
)
