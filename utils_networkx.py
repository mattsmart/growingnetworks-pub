import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from networkx.algorithms.isomorphism.tree_isomorphism import rooted_tree_isomorphism, tree_isomorphism
from networkx.algorithms import isomorphism

from preset_bionetworks import PRESET_BIONETWORKS, VALID_BIONAMES
from settings import DIR_OUTPUT


def draw_from_adjacency(A, node_color=None, labels=None, draw_edge_labels=False, draw_division=None,
                        cmap='Pastel1', title='Cell graph', spring_seed=None, fpath=None, fmod='', ftype='.pdf',
                        gviz_prog='twopi', explicit_layout=None,
                        xlims=None, ylims=None, clims=None,
                        ax=None, figsize=(4, 4)):
    """
    create_using=nx.DiGraph -- store as directed graph with possible self-loops
    create_using=nx.Graph -- store as undirected graph with possible self-loops

    draw_division: None, or cellgraph.division_events
        if the cellgraph.division_events attribute is passed, draw as Directed graph where the arrows point to daughters

    explicit_layout: specify node positions for A explicitly (the edges are inferred from A)

    cmap options: 'Blues', 'Pastel1', 'Spectral_r'
    """
    # TODO alternative visualization wth legend for discrete data (nDiv) and colorbar for continuous data (birth times)
    M = A.shape[0]

    def pick_seed_using_num_cells():
        seed_predefined = {
            1: 0,
            2: 0,
            4: 0,
            8: 0,
            16: 0,
            32: 0,
        }
        seed = seed_predefined.get(M, 0)  # M = A.shape[0] and seed_default = 0
        return seed

    # plot settings
    ns = 200  # 800
    alpha = 1.0
    fs = 10  # 6
    if labels is not None:
        fs = 6
    font_color = 'k'  # options: 'whitesmoke', 'k'
    if spring_seed is not None:
        print('Note: forcing spring seed using pick_seed_using_num_cells()')
        spring_seed = pick_seed_using_num_cells()
        fmod += '_Vspring%d' % spring_seed
    else:
        fmod += '_V%s' % gviz_prog

    # initialize the figure
    fresh_figure = False
    if ax is None:
        fresh_figure = True
        plt.figure(figsize=figsize)  # default 8,8; try 4,4 for quarter slide, or 6,6 for half a slide
        ax = plt.gca()
    ax.set_title(title)

    # initialize the graph
    if draw_division is None:
        G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.Graph)
    else:
        # created a directed graph based on the array draw_division (cellgraph.division_events)
        # each row in draw_division is a tuple of the form (mother_int, daughter_int, time_int)
        A_directed = np.zeros_like(A)
        for i in range(draw_division.shape[0]):
            a, b, _ = draw_division[i, :]
            A_directed[a, b] = 1
        G = nx.from_numpy_matrix(np.matrix(A_directed), create_using=nx.DiGraph)
    # determine node positions
    if explicit_layout is not None:
        layout = explicit_layout
    else:
        if spring_seed is not None:
            layout = nx.spring_layout(G, seed=spring_seed)
        else:
            # prog options: twopi, circo, dot
            layout = nx.nx_agraph.graphviz_layout(G, prog=gviz_prog, args="")

    # draw the nodes
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if clims is not None:
        vmin, vmax = clims
    else:
        vmin, vmax = None, None
    nx.draw_networkx(G, layout, with_labels=False,
                     node_color=node_color, cmap=cmap, node_size=ns, alpha=alpha,
                     width=1.0, linewidths=2.0, edgecolors='black', vmin=vmin,vmax=vmax,
                     ax=ax)

    # write node labels
    if labels is not None:
        nx.draw_networkx_labels(G, layout, labels, font_size=fs, font_color=font_color, verticalalignment='bottom',
                                ax=ax)
        cell_labels = {idx: r'Cell $%d$' % (idx) for idx in range(M)}
        nx.draw_networkx_labels(G, layout, cell_labels, font_size=fs, font_color=font_color, verticalalignment='top',
                                ax=ax)
    else:
        cell_labels = {idx: r'$%d$' % (idx) for idx in range(M)}
        nx.draw_networkx_labels(G, layout, cell_labels, font_size=fs, font_color=font_color, ax=ax)
    # write edge labels
    if draw_edge_labels:
        nx.draw_networkx_edge_labels(G, pos=layout, ax=ax)

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.axis("off")
    if fresh_figure:
        if fpath is None:
            plt.show()
        else:
            plt.savefig(fpath + fmod + ftype)
        plt.close()
    return ax


def check_tree_isomorphism(A1, A2, root1=None, root2=None, rooted=False):
    """
    See documentation here: (fast methods for checking tree type graph ismorphism
    https://networkx.org/documentation/stable/reference/algorithms/isomorphism.html?highlight=isomorphism#module-networkx.algorithms.isomorphism.tree_isomorphism
    """
    G1 = nx.from_numpy_matrix(np.matrix(A1), create_using=nx.Graph)
    G2 = nx.from_numpy_matrix(np.matrix(A2), create_using=nx.Graph)
    if rooted:
        print("WARNING - Use tree_isomorphism, not the rooted variant, as it seems to give incorrect results")
        iso_list = rooted_tree_isomorphism(G1, root1, G2, root2)
    else:
        iso_list = tree_isomorphism(G1, G2)

    if not iso_list:
        is_isomorphic = False  # i.e. it is an empty list, so no isomorphism found
    else:
        is_isomorphic = True
    return is_isomorphic, iso_list


def check_tree_isomorphism_with_insect(A1, insectname, test_subgraph_iso = False):
    """
    Suppose I have a graph specified by adjacency matrix A1.
    I would like to compare the A1 graph with a known insect cyst graph.

    There are two kinds of comparisons:
    (1) check whether A1 graph is isomorphic to cyst graph of specified insect
    (2) check whether A1 graph is isomorphic to a subgraph of cyst of specified insect

    (1) is primarily used for checking in a parameter sweep whether I have produced any naturally observed graphs
    (2) is primarily used for checking whether continued growth of A1 graph could ever plausibly lead to naturally observed graph

    A1 is a square matrix.

    insectname is a string, which must be from the list VALID_BIONAMES defined in presets_bionetworks.py

    test_subgraph_iso says whether I should check for whole graph isomorphism (False, default) or subgraph isomorphism (True)
    """
    assert insectname in VALID_BIONAMES
    A2 = PRESET_BIONETWORKS[insectname]

    if test_subgraph_iso is False:
        is_isomorphic, iso_list = check_tree_isomorphism(A1, A2)
    else:
        G1 = nx.from_numpy_matrix(np.matrix(A1), create_using=nx.Graph)
        G2 = nx.from_numpy_matrix(np.matrix(A2), create_using=nx.Graph)
        GM = isomorphism.GraphMatcher(G2, G1)
        is_isomorphic = GM.subgraph_is_isomorphic()

    return is_isomorphic


if __name__ == '__main__':

    flag_insect = False
    flag_draw = False
    flag_isomorphism_check = False
    flag_plot_specific_insect = True

    if flag_insect:
        A2 = np.zeros((24, 24))

        A2[0, 1] = 1
        A2[1, 0] = 1

        A2[2, 1] = 1
        A2[1, 2] = 1

        A2[2, 3] = 1
        A2[3, 2] = 1

        A2[4, 3] = 1
        A2[3, 4] = 1

        A2[4, 5] = 1
        A2[5, 4] = 1

        A2[6, 5] = 1
        A2[5, 6] = 1

        A2[6, 7] = 1
        A2[7, 6] = 1

        A2[8, 7] = 1
        A2[7, 8] = 1

        A2[8, 9] = 1
        A2[9, 8] = 1

        A2[10, 9] = 1
        A2[9, 10] = 1

        A2[10, 11] = 1
        A2[11, 10] = 1

        A2[12, 11] = 1
        A2[11, 12] = 1

        A2[12, 13] = 1
        A2[13, 12] = 1

        A2[14, 13] = 1
        A2[13, 14] = 1

        A2[15, 2] = 1
        A2[2, 15] = 1

        A2[15, 16] = 1
        A2[16, 15] = 1

        A2[17, 3] = 1
        A2[3, 17] = 1

        A2[18, 7] = 1
        A2[7, 18] = 1

        A2[18, 19] = 1
        A2[19, 18] = 1

        A2[20, 19] = 1
        A2[19, 20] = 1

        A2[19, 21] = 1
        A2[21, 19] = 1

        A2[22, 11] = 1
        A2[11, 22] = 1

        A2[23, 13] = 1
        A2[13, 23] = 1

        is_isomorphic = check_tree_isomorphism_with_insect(A2, 'parthenogeneticus', 'sub')
        print(is_isomorphic)
        draw_from_adjacency(A2, fpath='foo')

    if flag_draw:
        A1 = np.array([
            [0, 1, 0, .8, 0],
            [0, 0, .4, 0, .3],
            [0, 0, 0, 0, 0],
            [0, 0, .6, 0, .7],
            [0, 0, 0, .2, 0]
        ])

        A2 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        draw_from_adjacency(A2, fpath='foo')

    if flag_isomorphism_check:
        A1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])

        A2_iso_to_A1 = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        A3_distinct = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])

        print("Empty list means its found to be NOT isomorphic (no mapping)")
        print('Check A1, A1 - trivially isomorphic')
        print(check_tree_isomorphism(A1, A1, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A1))
        print('Check A1, A2 - expect isomorphism by swap node 0 and node 1')
        print(check_tree_isomorphism(A1, A2_iso_to_A1, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A2_iso_to_A1))
        print('Repeat for A1, A3 - expect distinct')
        print(check_tree_isomorphism(A1, A3_distinct, root1=0, root2=0, rooted=True))
        print(check_tree_isomorphism(A1, A3_distinct))

    if flag_plot_specific_insect:
        bioname = 'melanogaster'
        adj = PRESET_BIONETWORKS[bioname]

        figsize = (4, 4)
        degree = np.diag(np.sum(adj, axis=1))
        degree_vec = np.diag(degree)
        title = '%s (%d cells) - graph degree' % (bioname, adj.shape[0])
        fpath = DIR_OUTPUT + os.sep + '%s_Degree' % bioname

        for gviz_prog in ['dot', 'circo', 'twopi']:
            draw_from_adjacency(
                adj, title=title, node_color=degree_vec,
                labels=None, cmap='Pastel1', fpath=None,
                figsize=figsize, gviz_prog=gviz_prog, spring_seed=None)
        draw_from_adjacency(
            adj, title=title, node_color=degree_vec,
            labels=None, cmap='Pastel1', fpath=None,
            figsize=figsize, gviz_prog=gviz_prog, spring_seed=0)
