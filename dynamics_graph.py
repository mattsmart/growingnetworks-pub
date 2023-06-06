import numpy as np
from numba import jit


def f_of_x_single_cell(t_scalar, init_cond, single_cell):
    # Gene regulatory dynamics internal to one cell based on its state variables (dx/dt = f(x))
    dxdt = single_cell.ode_system_vector(init_cond, t_scalar)
    return dxdt


def graph_ode_system_vectorized(t_scalar, xvec, single_cell, cellgraph):
    """
    Intended to play the role of "foo(x)" for scipy solve_ivp(foo, vectorized=True), which solves dx/dt = foo(x)

    In vectorized=True mode, solve_ivp expects "foo" to handle xvec input of two possible shapes:
      - Case 1: xvec is numpy array with shape NM x 1  (general case, used for most timesteps)
      - Case 2: xvec is numpy array with shape NM x NM (special case, used when approximating the jacobian of foo)
      - (Aside: this is in contrast to vectorized=False, which assumes always xvec has shape NM)

    When vectorized=False, graph_ode_system() -- which is simplified version of this function -- should be used instead.

    Args:
        t_scalar          - (float) scalar representing the time coordinate of the ODE integration
        xvec              - (numpy arr, [NM x 1] or [NM x NM]) state variable of the ODE
        single_cell       - instance of custom class SingleCell
        cellgraph         - instance of custom class CellGraph
    Output:
        dxdt              - (numpy arr, [NM x 1] or [NM x NM]) the right-hand-side of the vector field dx/dt = foo(x)
    """
    xvec_matrix = cellgraph.state_to_rectangle(xvec)
    # Term 1: stores the single cell gene regulation (for each cell)
    #         [f(x_1) f(x_2) ... f(x_M)] as a stacked NM long 1D array
    batch_sz = xvec.shape[-1]  # for vectorized mode of solve_ivp
    term_1 = np.zeros((cellgraph.graph_dim_ode, batch_sz))
    # TODO maybe replace by map for speedup?
    for cell_idx in range(cellgraph.num_cells):
        a = cellgraph.sc_dim_ode * cell_idx
        b = cellgraph.sc_dim_ode * (cell_idx + 1)
        xvec_sc = xvec_matrix[:, cell_idx]
        # print(xvec_sc.shape)
        term_1[a:b, :] = f_of_x_single_cell(t_scalar, xvec_sc, single_cell)  # f_of_x has shape [N] or [N x N]

    # Term 2: computational approach B - framed as matrix multiplication for further parallelization
    # - note that the approach A code block above can be written in matrix form (sped up) as -Dvec * np.dot(X, L^T)
    # Step 1 - Compute X_times_LT
    # - recall xvec_matrix has shape [N x M x b], where b = 1 or NM (represents the batch index for the vectorization)
    # - we want to multiply by laplacian L^T, which is [M x M], using np.matmul
    # - to do this, first create a numpy "view" of xvec_matrix which shifts the last axis (batch index) to the front
    xvec_matrix_v = np.rollaxis(xvec_matrix, -1)                           # has shape [b x N x M] where b = 1 or NM
    # - now perform the matrix multiplication
    X_times_LT_v = np.matmul(xvec_matrix_v, cellgraph.laplacian.T)         # has shape [b x N x M]
    X_times_LT = np.rollaxis(X_times_LT_v, 0, start=X_times_LT_v.ndim)     # has shape [N x M x b]
    # Step 2 - compute term_2 by generalizing the approach from graph_ode_system()
    # - note cellgraph.diffusion                is a 1D array with shape [N]
    # - note cellgraph.diffusion[:, None, None] is a 3D array with shape [N x 1 x 1]
    # - this technique is known as "broadcasting": https://numpy.org/doc/stable/user/basics.broadcasting.html
    # - intuition:
    #     - X_times_LT is a 3D tensor of shape [N x M x b]
    #     - each "horizontal slice" corresponds to one of the N genes/proteins
    #     - the line below (efficiently) scales each slice by its corresponding diffusion coefficient
    D_times_X_times_LT = - cellgraph.diffusion[:, None, None] * X_times_LT  # has shape [N x M x b]
    # Step 3 - convert back to original shape convention (matching input array xvec)
    term_2 = cellgraph.state_to_stacked(D_times_X_times_LT)                 # has shape [NM x b]

    dxvec_dt = term_1 + term_2
    return dxvec_dt


def graph_ode_system(t_scalar, xvec, single_cell, cellgraph):
    """
    Non-vectorized implementation of graph_ode_system_vectorized()
    """
    xvec_matrix = cellgraph.state_to_rectangle(xvec)
    # Term 1: stores the single cell gene regulation (for each cell)
    #         [f(x_1) f(x_2) ... f(x_M)] as a stacked NM long 1D array
    term_1 = np.zeros(cellgraph.graph_dim_ode)
    # TODO can this be sped up? maybe map?
    for cell_idx in range(cellgraph.num_cells):
        a = cellgraph.sc_dim_ode * cell_idx
        b = cellgraph.sc_dim_ode * (cell_idx + 1)
        xvec_sc = xvec_matrix[:, cell_idx]
        term_1[a:b] = f_of_x_single_cell(t_scalar, xvec_sc, single_cell)

    # Term 2: stores the cell-cell coupling which is just laplacian diffusion -c * L * x
    # Note: we consider each reactant separately with own diffusion rate
    X_times_LT = np.matmul(xvec_matrix, cellgraph.laplacian.T)
    # Note: the following line is equivalent to -np.matmul(cellgraph.diffusion_diag_matrix, X_times_LT)
    # - multiplying by a diagonal matrix can be spedup via broadcasting to multiply by constant rows
    D_times_X_times_LT = - cellgraph.diffusion[:, None] * X_times_LT
    term_2 = cellgraph.state_to_stacked(D_times_X_times_LT)

    dxvec_dt = term_1 + term_2
    return dxvec_dt
