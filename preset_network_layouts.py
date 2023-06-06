
def preset_layout_lacewing(A, anti=True):
    """
    Given an array A, return pos -- dict mapping node_idx -> position (x, y)
    https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
    """
    n_lacewing = 12
    # step 1: build the full lacewing position layout - 12 cells
    # - method: pick positions on a 4x4 grid
    dx, dy = 1.0, 1.0
    xv = [x * dx for x in range(4)]
    yv = [y * dy for y in range(4)]

    # specify x,y limits to regularize plots of growing network
    dx_buffer, dy_buffer = 0.5 * dx, 0.5 * dy
    xlims = (xv[0] - dx_buffer, xv[-1] + dx_buffer)
    ylims = (yv[0] - dy_buffer, yv[-1] + dy_buffer)

    # initialize node position dictionary and place cell 0 somewhere in xy-plane
    pos_nodes = dict()
    pos_nodes[0] = (xv[2], yv[2])

    def add_node(duaghter_idx, mother_idx, direction):
        x0, y0 = pos_nodes[mother_idx]
        if direction == 'up':
            pos_nodes[duaghter_idx] = (x0, y0 + dy)
        if direction == 'down':
            pos_nodes[duaghter_idx] = (x0, y0 - dy)
        if direction == 'left':
            pos_nodes[duaghter_idx] = (x0 - dx, y0)
        if direction == 'right':
            pos_nodes[duaghter_idx] = (x0 + dx, y0)

    if anti:
        # first three divisions
        add_node(1, 0, 'left')  # cell 1 is "left" of cell 0
        add_node(2, 0, 'down')  # cell 2
        add_node(3, 1, 'down')  # cell 3
        # next four
        add_node(4, 3, 'down')  # cell 5
        add_node(5, 2, 'down')  # cell 4
        add_node(6, 1, 'up')  # cell 6
        add_node(7, 0, 'up')  # cell 7
        # last four
        add_node(8, 3, 'left')  # cell 8
        add_node(9, 4, 'left')  # cell 9
        add_node(10, 2, 'right')  # cell 10
        add_node(11, 5, 'right')  # cell 11
    else:
        # first three divisions
        add_node(1, 0, 'left')  # cell 1 is "left" of cell 0
        add_node(2, 1, 'down')  # cell 2
        add_node(3, 0, 'down')  # cell 3
        # next four
        add_node(4, 2, 'down')  # cell 4
        add_node(5, 1, 'up')  # cell 5
        add_node(6, 0, 'up')  # cell 6
        add_node(7, 3, 'down')  # cell 7
        add_node(8, 4, 'left')  # cell 8
        # last four
        add_node(9, 2, 'left')  # cell 9
        add_node(10, 7, 'right')  # cell 10
        add_node(11, 3, 'right')  # cell 11

    # step 2: remove cells which don't exist yet based on shape of A
    num_cells = A.shape[0]
    for i in range(num_cells, n_lacewing):
        pos_nodes.pop(i)

    return pos_nodes, xlims, ylims


def preset_layout_drosophila_M20(A):
    """
    Given an array A, return pos -- dict mapping node_idx -> position (x, y)
    """
    ncells_full = 20
    # step 1: build the full layout 20 cells
    # - method: pick positions on a 5x4 grid
    dx, dy = 1.0, 1.0
    xv = [x * dx for x in range(5)]
    yv = [y * dy for y in range(4)]

    # specify x,y limits to regularize plots of growing network
    dx_buffer, dy_buffer = 0.5 * dx, 0.5 * dy
    xlims = (xv[0] - dx_buffer, xv[-1] + dx_buffer)
    ylims = (yv[0] - dy_buffer, yv[-1] + dy_buffer)

    # initialize node position dictionary and place cell 0 somewhere in xy-plane
    pos_nodes = dict()
    pos_nodes[0] = (xv[3], yv[2])

    def add_node(duaghter_idx, mother_idx, direction):
        x0, y0 = pos_nodes[mother_idx]
        if direction == 'up':
            pos_nodes[duaghter_idx] = (x0, y0 + dy)
        if direction == 'down':
            pos_nodes[duaghter_idx] = (x0, y0 - dy)
        if direction == 'left':
            pos_nodes[duaghter_idx] = (x0 - dx, y0)
        if direction == 'right':
            pos_nodes[duaghter_idx] = (x0 + dx, y0)

    # first three divisions
    add_node(1, 0, 'left')  # cell 1 is "left" of cell 0
    add_node(2, 1, 'left')  # cell 2
    add_node(3, 0, 'down')  # cell 3
    # next four
    add_node(4, 2, 'down')  # cell 5
    add_node(5, 1, 'down')  # cell 4
    add_node(6, 0, 'up')  # cell 6
    add_node(7, 3, 'down')  # cell 7
    # last four
    add_node(8, 4, 'down')  # cell 8
    add_node(9, 2, 'up')  # cell 9
    add_node(10, 1, 'up')  # cell 10
    add_node(11, 5, 'down')  # cell 11
    add_node(12, 0, 'right')  # cell 12
    add_node(13, 3, 'right')  # cell 13
    add_node(14, 6, 'right')  # cell 14
    add_node(15, 7, 'right')  # cell 15
    add_node(16, 8, 'left')  # cell 16
    add_node(17, 4, 'left')  # cell 17
    add_node(18, 2, 'left')  # cell 18
    add_node(19, 9, 'left')  # cell 19

    # step 2: remove cells which don't exist yet based on shape of A
    num_cells = A.shape[0]
    for i in range(num_cells, ncells_full):
        pos_nodes.pop(i)

    return pos_nodes, xlims, ylims


def preset_layout_drosophila_M15(A):
    """
    Given an array A, return pos -- dict mapping node_idx -> position (x, y)
    """
    ncells_full = 15
    # step 1: build the full layout 15 cells
    # - method: pick positions on a 4x4 grid
    dx, dy = 1.0, 1.0
    xv = [x * dx for x in range(4)]
    yv = [y * dy for y in range(4)]

    # specify x,y limits to regularize plots of growing network
    dx_buffer, dy_buffer = 0.5 * dx, 0.5 * dy
    xlims = (xv[0] - dx_buffer, xv[-1] + dx_buffer)
    ylims = (yv[0] - dy_buffer, yv[-1] + dy_buffer)

    # initialize node position dictionary and place cell 0 somewhere in xy-plane
    pos_nodes = dict()
    pos_nodes[0] = (xv[2], yv[2])

    def add_node(duaghter_idx, mother_idx, direction):
        x0, y0 = pos_nodes[mother_idx]
        if direction == 'up':
            pos_nodes[duaghter_idx] = (x0, y0 + dy)
        if direction == 'down':
            pos_nodes[duaghter_idx] = (x0, y0 - dy)
        if direction == 'left':
            pos_nodes[duaghter_idx] = (x0 - dx, y0)
        if direction == 'right':
            pos_nodes[duaghter_idx] = (x0 + dx, y0)

    # first three divisions
    add_node(1, 0, 'left')  # cell 1 is "left" of cell 0
    add_node(2, 0, 'down')  # cell 2
    add_node(3, 1, 'down')  # cell 3
    # next four
    add_node(4, 0, 'up')  # cell 5
    add_node(5, 2, 'down')  # cell 4
    add_node(6, 1, 'up')  # cell 6
    add_node(7, 3, 'down')  # cell 7
    # last four
    add_node(8, 4, 'right')  # cell 8
    add_node(9, 0, 'right')  # cell 9
    add_node(10, 2, 'right')  # cell 10
    add_node(11, 1, 'left')  # cell 11
    add_node(12, 5, 'right')  # cell 12
    add_node(13, 6, 'left')  # cell 13
    add_node(14, 3, 'left')  # cell 14

    # step 2: remove cells which don't exist yet based on shape of A
    num_cells = A.shape[0]
    for i in range(num_cells, ncells_full):
        pos_nodes.pop(i)

    return pos_nodes, xlims, ylims


def preset_layout_drosophila_M12(A):
    """
    Given an array A, return pos -- dict mapping node_idx -> position (x, y)
    """
    ncells_full = 12
    # step 1: build the full layout 12 cells
    # - method: pick positions on a 4x4 grid
    dx, dy = 1.0, 1.0
    xv = [x * dx for x in range(4)]
    yv = [y * dy for y in range(4)]

    # specify x,y limits to regularize plots of growing network
    dx_buffer, dy_buffer = 0.5 * dx, 0.5 * dy
    xlims = (xv[0] - dx_buffer, xv[-1] + dx_buffer)
    ylims = (yv[0] - dy_buffer, yv[-1] + dy_buffer)

    # initialize node position dictionary and place cell 0 somewhere in xy-plane
    pos_nodes = dict()
    pos_nodes[0] = (xv[3], yv[2])

    def add_node(duaghter_idx, mother_idx, direction):
        x0, y0 = pos_nodes[mother_idx]
        if direction == 'up':
            pos_nodes[duaghter_idx] = (x0, y0 + dy)
        if direction == 'down':
            pos_nodes[duaghter_idx] = (x0, y0 - dy)
        if direction == 'left':
            pos_nodes[duaghter_idx] = (x0 - dx, y0)
        if direction == 'right':
            pos_nodes[duaghter_idx] = (x0 + dx, y0)

    # first three divisions
    add_node(1, 0, 'left')  # cell 1 is "left" of cell 0
    add_node(2, 1, 'down')  # cell 2
    add_node(3, 0, 'down')  # cell 3
    # next four
    add_node(4, 2, 'down')   # cell 5
    add_node(5, 1, 'up')     # cell 4
    add_node(6, 0, 'up')  # cell 6
    add_node(7, 3, 'down')   # cell 7
    # last four
    add_node(8, 4, 'left')  # cell 8
    add_node(9, 2, 'left')   # cell 9
    add_node(10, 1, 'left')  # cell 10
    add_node(11, 5, 'left')  # cell 11

    # step 2: remove cells which don't exist yet based on shape of A
    num_cells = A.shape[0]
    for i in range(num_cells, ncells_full):
        pos_nodes.pop(i)

    return pos_nodes, xlims, ylims
