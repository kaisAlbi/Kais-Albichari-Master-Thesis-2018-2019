import SignalsModel
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import math

def getCoopDefectStrats(strategies):
    coop_strats = []
    defect_strats = []
    for i in range(len(strategies)):
        signal = strategies[i][0]
        if strategies[i][signal + 1] == "C":
            coop_strats.append("".join(map(str, strategies[i])))
        else:
            defect_strats.append("".join(map(str, strategies[i])))
    return [coop_strats, defect_strats]


def getColorsForStrats(strats):
    colors = []
    for i in range(len(strats)):
        signal = strats[i][0]
        if strats[i][signal + 1] == "C":
            colors.append("blue")
        else:
            colors.append("red")
    return colors


def showTransForStrats1Sig(egt_model):
    strats_mat = egt_model.getStrategies()
    strats = []
    n = len(strats_mat)
    for i in range(len(strats_mat)):
        strats.append(strats_mat[i][1])
    ncolors = ["blue", "red"]
    fix_probs = egt_model.getFixProbs()
    transition = egt_model.getTransitionMatrix()
    stationary = egt_model.getStationaryDistrib()
    drift = 1 / egt_model.getPopSize()
    G = nx.DiGraph()
    G.add_nodes_from(strats)
    for j in range(n):
        for i in range(n):
            if (fix_probs[i, j] > drift):
                G.add_edge(strats[i], strats[j], weight=transition[i, j])


    eselect = [(u, v) for (u, v, d) in G.edges(data=True)]
    eselect_labels = dict(((u, v), '$'+str(round((d['weight'] / drift),1))+str("\\")+'rho_N$')
                          for (u, v, d) in G.edges(data=True))
    fig = plt.figure(figsize=(10, 10), dpi=150)

    nodes_labels = {}
    for i in range(len(strats)):
        nodes_labels["".join(map(str, strats[i]))] = "".join(
            map(str, strats[i] + "\n" + str(math.trunc(round((stationary[i]*100),0))) + "%"))
    pos = {strats[0]: np.array([0, 1]), strats[1]: np.array([0, 0])}
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=ncolors, with_labels=True)

    # edges
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect,
                                            width=3, style="dashdot", arrowsize=50)

    # node labels

    nx.draw_networkx_labels(G, pos, nodes_labels,
                            font_size=8, font_weight='bold', font_color='white')

    # edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=14)

    plt.axis('off')
    plt.show()  # display


def showTransitionForStrategies(egt_model):
    strats_mat = egt_model.getStrategies()
    n = len(strats_mat)
    strats = ["".join(map(str, strats_mat[i])) for i in range(len(strats_mat))]
    ncolors = getColorsForStrats(strats_mat)
    fix_probs = egt_model.getFixProbs()
    stationary = egt_model.getStationaryDistrib()
    SignalsModel.displayTransitionAndStationaryDistrib(egt_model)
    drift = 1 / egt_model.getPopSize()
    G = nx.DiGraph(directed=True)
    G.add_nodes_from(strats)
    for j in range(n):
        for i in range(n):
            if fix_probs[i, j] > drift:
                G.add_edge(strats[i], strats[j], weight=fix_probs[i, j]) #transition[i, j]*10)

    eselect_8= [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 8 * drift and (u,v))]
    eselect_4 = [(u, v) for (u, v, d) in G.edges(data=True) if
                          (d['weight'] > 4 * drift and (u, v) not in eselect_8)]
    eselect_1_same_sig = []
    for (u, v, d) in G.edges(data=True):
        if (d['weight'] >  drift and (u,v) not in eselect_8 and (u,v) not in eselect_4):
            if (u[0] == v[0]):
                eselect_1_same_sig.append((u, v))
    eselect_1_diff_sig = [(u, v) for (u, v, d) in G.edges(data=True) if
                          (d['weight'] > drift and (u, v) not in eselect_8 and (u,v) not in eselect_4 and (u,v) not in eselect_1_same_sig)]


    fig = plt.figure(figsize=(10, 10), dpi=150)

    nodes_labels = {}
    for i in range(len(strats)):
        nodes_labels["".join(map(str, strats[i]))] = "".join(
            map(str, strats[i] + "\n" + str(math.trunc(round(stationary[i] * 100, 0))) + "%"))
    #strats = [0CC, 0CD, 0DC, 0DD, 1CC, 1CD, 1DC, 1DD]
    pos = {strats[0]: np.array([10, 10]), strats[1]: np.array([0, 10]), strats[2]: np.array([5, 5]), strats[3]: np.array([5, 15]),
           strats[4]: np.array([0, 5]), strats[5]: np.array([5, 10]), strats[6]: np.array([10, 5]) ,strats[7]: np.array([5, 0])}
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=ncolors, with_labels=True)

    # edges
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_8,
                                            width=4, arrowsize=50, edge_color="black")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_4,
                                            width=2, arrowsize=40, edge_color="blue")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_1_same_sig,
                                            width=0.5, arrowsize=30, edge_color="grey")
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edgelist=eselect_1_diff_sig,
                                            width=0.5, arrowsize=30, edge_color="orange")



    # node labels

    nx.draw_networkx_labels(G, pos, nodes_labels,
                            font_size=8, font_weight='bold', font_color='white')

    # edge labels

    legend_elements = [Line2D([0], [0], marker='$\leftarrow$', color='orange', lw=1, label='$>'+str("\\")+'rho_N$'+" diff sig",
                              markerfacecolor='orange', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='grey', lw=1, label='$>'+str("\\")+'rho_N$'+" same sig",
                              markerfacecolor='grey', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='blue', lw=1,
                              label='$>4' + str("\\") + 'rho_N$',
                              markerfacecolor='blue', markersize=25),
                       Line2D([0], [0], marker='$\leftarrow$', color='black',lw=1, label='$>8'+str("\\")+'rho_N$',
                              markerfacecolor='black', markersize=25)
                       ] #([0], [0], marker='$\leftarrow$',color='blue', lw=1, label='>ρ'),
    plt.legend(handles=legend_elements, loc='upper right')

    plt.axis('off')
    plt.show()  # display


R, S, T, P = 1., -0.5, 1.5, 0.
Z = 150 #Population size
beta = 0.05

egt_1pd= SignalsModel.EGTModel(R, S, T, P, Z, beta, 1)
#egt_2pd= SignalsModel.EGTModel(R, S, T, P, Z, beta, 2)
showTransForStrats1Sig(egt_1pd)
#showTransitionForStrategies(egt_2pd)