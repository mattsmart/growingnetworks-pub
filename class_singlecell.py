import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import quad, fixed_quad

# simulate_dynamics_general solves the ode's
from dynamics_detect_cycles import detect_oscillations_manual_1d
from dynamics_generic import simulate_dynamics_general
from dynamics_vectorfields import set_ode_params, ode_integration_defaults, set_ode_vectorfield, set_ode_attributes
from utils_io import run_subdir_setup
from settings import STYLE_DYNAMICS, STYLE_ODE, DIR_OUTPUT


class SingleCell():

    def __init__(self, init_cond_ode=None, style_ode=STYLE_ODE, params_ode=None, label=''):
        """
        For numeric cell labels (network growth), use label='%d' % idx, for instance
        """
        self.style_ode = style_ode
        dim_ode, dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.dim_ode = dim_ode            # dimension of ODE system
        self.dim_misc = dim_misc          # dimension of misc. variables (e.g. fusome content)
        self.num_variables = self.dim_ode + self.dim_misc
        # setup names for all dynamical variables
        self.variables_short = variables_short
        self.variables_long = variables_long
        if label != '':
            for idx in range(self.num_variables):
                self.variables_short[idx] += '_%s' % label
                self.variables_long[idx] += ' (Cell %s)' % label
        # make this flexible if other single cell ODEs are used
        self.state_ode = init_cond_ode
        self.params_ode = params_ode
        if self.params_ode is None:
            self.params_ode = set_ode_params(self.style_ode)

    def ode_system_vector(self, init_cond, t):
        ode_kwargs = {
            'z': init_cond[-1],
            't': t
        }
        dxdt = set_ode_vectorfield(self.style_ode, self.params_ode, init_cond, **ode_kwargs)
        return dxdt

    def trajectory(self, init_cond=None, t0=None, t1=None, num_steps=None, dynamics_method=STYLE_DYNAMICS,
                   flag_info=False, **solver_kwargs):
        """
        The num_steps kwarg will only act if we use a solvers which expect an array of times
        In general, we use adaptive solver and just specify t0, t1 (e.g. scipy Radau via solve_ivp())
        """
        # integration parameters
        T0, T1, NUM_STEPS, INIT_COND = ode_integration_defaults(self.style_ode)
        if init_cond is None:
            init_cond = self.state_ode
            if self.state_ode is None:
                init_cond = INIT_COND
        if t0 is None:
            t0 = T0
        if t1 is None:
            t1 = T1
        if num_steps is None:
            num_steps = NUM_STEPS
        times = np.linspace(t0, t1, num_steps + 1)
        if flag_info:
            print("ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0])
            print("Init Cond:", init_cond)
            self.printer()

        r, times = simulate_dynamics_general(init_cond, times, self, dynamics_method=dynamics_method, **solver_kwargs)
        if flag_info:
            print('Done trajectory\n')

        return r, times

    def printer(self):
        print('dim_ode:', self.dim_ode)
        print('dim_misc:', self.dim_misc)
        print('num_variables:', self.num_variables)
        print("State variables:")
        for idx in range(self.num_variables):
            print("\t %s: %s | %s" % (idx, self.variables_short[idx], self.variables_long[idx]))

    def write_ode_params(self, fpath):
        with open(fpath, "a", newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            keys = list(self.params_ode.keys())
            keys.sort()
            for k in keys:
                writer.writerow([k, self.params_ode[k]])

    def report_oscillations(self, times, traj, state_choice=0, verbose=False):
        # detect cycles for a singlecell trajectory and annotate them
        assert self.style_ode == 'PWL3_swap'
        assert state_choice in [0, 1]

        xlow = 0.5 * sc.params_ode['a1']
        xhigh = 0.5 * (sc.params_ode['a1'] + sc.params_ode['a2'])
        ylow = (sc.params_ode['a1'] - sc.params_ode['a2'])
        yhigh = sc.params_ode['a1']
        if state_choice == 0:
            ulow, uhigh = xlow, xhigh
        else:
            assert state_choice == 1
            ulow, uhigh = ylow, yhigh

        # manual detection of (potentially) more than one oscillation
        total_oscillations = 0
        total_events_idx = []
        total_events_times = []
        total_duration_cycles = []
        if verbose:
            print("Detecting oscillations...")
        osc_found = True                # init loop
        time_idx_previous_division = 0  # init loop
        while osc_found:
            # NOTE: currently, detect_oscillations_manual_1d will report the FIRST observed oscillation only,
            #   this is why we have this clunky code here wrapping it to handle case of multiple oscillations
            if verbose:
                print('checking for oscillation from t=%.2f [idx %d] onwards...'
                      % (times[time_idx_previous_division], time_idx_previous_division))
            num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_manual_1d(
                times[time_idx_previous_division:], traj[:, time_idx_previous_division:],
                xlow=ulow, xhigh=uhigh, use_mid=True, state_choice=state_choice, verbose=False, show=False)
            osc_found = num_oscillations > 0
            if osc_found:
                assert num_oscillations == 1  # breaks if we refactor detect_oscillations to return > 1 oscillation
                total_oscillations += num_oscillations
                total_events_idx += [a + time_idx_previous_division for a in events_idx]
                total_events_times += events_times
                total_duration_cycles += duration_cycles
                # track time index of last division
                time_idx_previous_division = total_events_idx[-1]
        if verbose:
            print("\t Number of oscillations = %d" % total_oscillations)
            print("\t events_idx", total_events_idx)
            print("\t events_times", total_events_times)
            print("\t duration_cycles", total_duration_cycles)
        return total_oscillations, total_events_idx, total_events_times, total_duration_cycles, ulow, uhigh


if __name__ == '__main__':
    scan_count_ndiv_fn_of_v = True

    style_ode = 'PWL3_swap'  # PWL3_zstepdecay, PWL3_swap, toy_clock
    sc = SingleCell(label='c1', style_ode=style_ode)
    if style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
        sc.params_ode['a1'] = 1
        sc.params_ode['a2'] = 0.25
        sc.params_ode['gamma'] = 1e-2
        sc.params_ode['epsilon'] = 1e-2  # try 1e-6
        sc.params_ode['pulse_vel'] = 0.0 #0.01  # 0 for constant z
        sc.params_ode['t_pulse_switch'] = 82.0  #75.0
    if style_ode in ['PWL4_auto_linear']:
        sc.params_ode['a1'] = 2
        sc.params_ode['a2'] = 1
    if style_ode in ['PWL3_zstepdecay']:
        sc.params_ode['epsilon'] = 1e-2
        sc.params_ode['a1'] = 1
        sc.params_ode['a2'] = 0.25
        sc.params_ode['gamma'] = 1e-2
        sc.params_ode['dz_stretch'] = 0.0
        sc.params_ode['dz_eta'] = 0.01
        sc.params_ode['dz_t_heaviside'] = 0.0

    init_cond = [0, 0, 0]
    #init_cond = [0, 0, 4.5]
    t0 = 0
    t1 = 200 #200 or 20

    solver_kwargs = {
        'method': 'Radau',
        'atol': 1e-8,
        'rtol': 1e-4,
        'dense_output': False,  # seems to have no effect
        't_eval': None}         # np.linspace(0, 100, 2000) or None
    r, times = sc.trajectory(flag_info=True, dynamics_method='solve_ivp',
                             init_cond=init_cond, t0=t0, t1=t1,
                             **solver_kwargs)
    print(r.shape)
    state_hist = r.T

    io_dict = run_subdir_setup(run_subfolder='singlecell')
    np.savetxt(io_dict['dir_base'] + os.sep + 'traj_times.txt', times)
    np.savetxt(io_dict['dir_base'] + os.sep + 'traj_x.txt', r)

    def PWL3swap_heuristic_period(z, m=2, delta=1e-4, z_rescale=False):
        if z_rescale:
            z = z / sc.params_ode['a1']

        a = sc.params_ode['a2'] / sc.params_ode['a1']
        gamma = sc.params_ode['gamma']
        epsilon = sc.params_ode['epsilon']
        z1 = 0.5 * (1 + gamma * m)
        z2 = 0.5 * (1 + a + gamma * m * (1-a))

        T_A = m/(2*z1) * np.log(1 + a * z1 / (z - z1))
        T_B = m/(2*z1) * np.log(1 + a * z1 / (z2 - z))
        T_C = 4 * epsilon / m * np.log(1 + a * m / (2 * delta))
        T_total = T_A + T_B + 0*T_C

        return T_total

    def PWL3swap_heuristic_freq_integral(v, m=2):
        """
        assumes v has already been "rescaled" (divide by a1)
        """
        a = sc.params_ode['a2'] / sc.params_ode['a1']
        gamma = sc.params_ode['gamma']
        z1 = 0.5 * (1 + gamma * m)
        z2 = 0.5 * (1 + a + gamma * m * (1-a))
        v_tp = v * sc.params_ode['t_pulse_switch']

        # setup integral
        intval = 0
        if v_tp >= z1:
            npt = int(5*1e5)
            zhigh = min(v_tp, z2)
            zarr = np.linspace(z1, zhigh, npt)
            dz = zarr[1] - zarr[0]
            for zval in zarr:
                intval += 1/PWL3swap_heuristic_period(zval, m=m) * dz

        return intval

    def PWL3swap_heuristic_ncycle(v, m=2, rescale_v=True):
        if rescale_v:
            v = v / sc.params_ode['a1']

        a = sc.params_ode['a2']/sc.params_ode['a1']
        gamma = sc.params_ode['gamma']
        z1 = 0.5 * (1 + gamma * m)
        z2 = 0.5 * (1 + a + gamma * m * (1-a))
        v_tp = float(v * sc.params_ode['t_pulse_switch'])

        # perform integral to get sval (as in Methods)
        def integrand(z):
            return 1 / PWL3swap_heuristic_period(z, m=m)
        sval = (1 / v) * quad(integrand, z1, min(v_tp, z2), maxiter=100)[0]

        if v_tp <= z1:
            n = 0
        elif z1 < v_tp <= z2:
            n = np.floor(2 * sval)
        else:
            n = 2 * np.floor(sval)
        return n

    # detect cycles for the singlecell trajectory and annotate them
    total_oscillations, total_events_idx, total_events_times, total_duration_cycles, ulow, uhigh \
        = sc.report_oscillations(times, state_hist, state_choice=0, verbose=True)

    # show period heuristic in special case of static z
    period_guess = None
    if sc.params_ode['pulse_vel'] == 0 and style_ode == 'PWL3_swap':
        period_guess = PWL3swap_heuristic_period(init_cond[-1], z_rescale=True)
        print("heuristic for limit cycle period:", period_guess)

    plt.plot(times, state_hist.T, label=[sc.variables_short[i] for i in range(sc.dim_ode)])
    plt.xlabel(r'$t$')
    for events_idx in total_events_idx:
        plt.axvline(times[events_idx], linestyle='--', c='gray')
        if period_guess is not None:
            plt.axvline(times[events_idx] + period_guess, linestyle='-.', c='purple')

    plt.axhline(0.5*(ulow + uhigh), linestyle='--', c='gray')
    plt.xlim(9, 20)
    plt.ylabel(r'state variable')
    plt.legend()
    plt.savefig(io_dict['dir_base'] + os.sep + 'traj_example.jpg')
    plt.show()

    # scan over pulse velocity to get number of divisions as function of v
    if scan_count_ndiv_fn_of_v:
        assert style_ode == 'PWL3_swap'
        velocities = np.concatenate((
            np.linspace(0/8.0, 0.0528/8.0, 50),
            np.linspace(0.0528/8.0, 0.06/8.0, 10),
            np.linspace(0.06/8.0, 0.24/8.0, 160)  #np.linspace(0.06, 0.25, 150)
        ))
        num_divisions = [0] * len(velocities)
        heuristic_fullcycles = [0] * len(velocities)
        print('\ncomputing n_div as function of pulse velocity...')
        for idx, v in enumerate(velocities):
            if idx % 20 == 0:
                print(idx, '...')
            sc.params_ode['pulse_vel'] = v
            r_scan, times_scan = sc.trajectory(flag_info=False, dynamics_method='solve_ivp',
                                     init_cond=init_cond, t0=t0, t1=t1,
                                     **solver_kwargs)
            total_oscillations, total_events_idx, total_events_times, total_duration_cycles, ulow, uhigh \
                = sc.report_oscillations(times_scan, r_scan.T, state_choice=0, verbose=False)
            num_divisions[idx] = total_oscillations
            heuristic_fullcycles[idx] = PWL3swap_heuristic_ncycle(v)
        np.savetxt(DIR_OUTPUT + os.sep + 'singlecell_ncycle_simdata.txt', num_divisions)
        np.savetxt(DIR_OUTPUT + os.sep + 'singlecell_ncycle_heuristic.txt', heuristic_fullcycles)
        np.savetxt(DIR_OUTPUT + os.sep + 'singlecell_ncycle_pulsev_unscaled.txt', velocities)
        np.savetxt(DIR_OUTPUT + os.sep + 'singlecell_ncycle_pulsev_scaled.txt', velocities/sc.params_ode['a1'])
        plt.plot(velocities, num_divisions, '--o', label='simulated data')
        plt.plot(velocities, heuristic_fullcycles, '-.x', label='heuristic')
        plt.legend()
        plt.ylabel(r'$n_{div}$')
        plt.xlabel('pulse velocity')
        plt.show()
