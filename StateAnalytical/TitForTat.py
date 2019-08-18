import numpy as np
import time
import matplotlib.pyplot as plt


strategies = ["AllC", "AllD", "TFT"]#, "NotTFT"]
nb_strats = len(strategies)
game = "PD"
R, P = 1, 0
T = 1.5
S = -0.5

beta = 0.05
rounds = 10
Z = 150


"""
payoff=np.array([lambda k: R, # C C
                 lambda k: S, # C D
                 lambda k: R, # C TFT
                 lambda k: S,   #C NotTFT
                 lambda k: T, # D C
                 lambda k: P, # D D
                 lambda k: ((T + ((rounds-1)*P))/rounds), # D TFT
                 lambda k: ((P + ((rounds-1)*T))/rounds),   #D NotTFT
                 lambda k: R, # TFT C
                 lambda k: ((S + ((rounds-1)*P))/rounds), # TFT D
                 lambda k: R, # TFT TFT
                 lambda k: (S + P + T + R) /4,        #TFT NotTFT
                 lambda k: T,       #NotTFT C
                 lambda k: ((P + ((rounds-1)*S))/rounds),         #NotTFT D
                 lambda k: (S + P + T + R) /4,          #NotTFT TFT
                 lambda k: (R + P) / 2        #NotTFT NotTFT

                ]).reshape(nb_strats,nb_strats)
"""
payoff=np.array([lambda k: R, # C C
                 lambda k: S, # C D
                 lambda k: R, # C TFT
                 lambda k: T, # D C
                 lambda k: P, # D D
                 lambda k: ((T + ((rounds-1)*P))/rounds), # D TFT
                 lambda k: R, # TFT C
                 lambda k: ((S + ((rounds-1)*P))/rounds), # TFT D
                 lambda k: R, # TFT TFT


                ]).reshape(nb_strats,nb_strats)


def fermiDistrib(first_fitness, second_fitness, positive):
    """
    :param first_payoff: payoff obtained by the first agent after the interaction
    :param second_payoff: payoff obtained by the second agent after the interaction
    :param positive: boolean value used to calculate probability increase and decrease (T +-)
    :return: probability that the first agent imitates the second ond
    """
    if positive:
        return 1. / (1. + np.exp(-beta * (first_fitness - second_fitness)))
    else:
        return 1. / (1. + np.exp(beta * (first_fitness - second_fitness)))


def probIncDec(n_A, inv_strat, res_strat):
    """
    :param n_A: number of invaders
    :param inv_strat: invaders' strategy
    :param res_strat: residents' strategy
    :return: probability to change the number of k invaders (by +- one at each time step)
    """
    result_inv = (((n_A) * payoff[inv_strat][inv_strat](n_A)) + ((Z - n_A) * payoff[inv_strat, res_strat](Z - n_A))) / float(Z)
    result_res = ((n_A * payoff[res_strat][inv_strat](n_A)) + ((Z - n_A) * payoff[res_strat, res_strat](Z - n_A))) / float(Z)


    inc = np.clip(((Z - n_A) * (n_A / Z) * fermiDistrib(result_res, result_inv, False)) / Z, 0., 1.)
    dec = np.clip(((Z - n_A) * (n_A / Z) * fermiDistrib(result_res, result_inv, True)) / Z, 0., 1.)
    return [inc, dec]


def fixationProba(res_strat, inv_strat):
    """
    :param res_index: resident index
    :param inv_index: invader index
    :return: fixation probability of the invader in a population of residents
    """

    result = 0.
    for i in range(0, Z):
        mul = 1.
        for j in range(1, i + 1):
            inc, dec = probIncDec(j, inv_strat, res_strat)
            lambda_j = np.float(dec / float(inc))
            mul *= lambda_j
        result += mul

    return np.clip(1. / result, 0., 1.)


def transitionMatrix(strats):
    """
    Compute the fixation probability for each pair invader-resident of strategies and build the fixation probabilities
    matrix and the transition matrix
    :return: transition matrix and fixation probabilities matrix
    """
    n = len(strats)
    norm_fact = 1 / float((n - 1))
    fix_probs = np.zeros((n, n))
    transitions = np.zeros((n, n))
    for i in range(n):
        start_time = time.time()
        transitions[i, i] = 1
        for j in range(n):
            if i != j:
                f_proba = fixationProba(i, j)
                fix_probs[i, j] = f_proba
                trans_value = f_proba * norm_fact
                transitions[i, j] = trans_value
                transitions[i, i] -= trans_value
        print("transitions values calculations for resident strat ", strats[i],
              " took --- %s seconds---" % (time.time() - start_time))
    return [transitions, fix_probs]


def stationaryDistrib(strats):
    """
    Calculate the transition matrix, and based on that matrix, the stationary distribution of each strategy
    :return: transition matrix, fixation probabilities matrix, stationary distribution
    """
    # start_time = time.time()
    t, f = transitionMatrix(strats)
    # print("The transition and fixation proba matrices calculation took --- %s seconds ---" % (time.time() - start_time))
    val, vect = np.linalg.eig(t.transpose())
    j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
    p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
    p_stationary /= p_stationary.sum()  # normalize
    # print("The stationary distribution calculation took --- %s seconds ---" % (
    #            time.time() - start_time))
    return t, f, p_stationary


def showStationaryDistrib(game, strats, stationary):
    """
    Plot a bar graph showing the stationary distribution
    :param game: evolutionary game being played
    :param strats: array of binary strategies
    :param stationary: stationary distribution array
    """
    n = len(strats)
    x = [i for i in range(n)]

    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("stationary distribution")

    plt.xticks(x, strats)
    for i in range (n):
        plt.bar(x[i], stationary[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


t, f, stationary = stationaryDistrib(strategies)

showStationaryDistrib(game, strategies, stationary)