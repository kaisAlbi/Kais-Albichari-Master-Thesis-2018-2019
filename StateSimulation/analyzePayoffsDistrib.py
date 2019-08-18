import numpy as np
import json
from itertools import product
from scipy.stats import t


def probaCouplesToPayoffs(filename, R, S, T, P, nb_strats):
    """
    Read probabilities of cooperation for pairs of strategies and computes the payoffs
    :param filename: name of the file in which lie the matrix of probabilities
    :param R: Reward for mutual cooperation
    :param S: Sucker payoff
    :param T: Temptation to defect payoff
    :param P: Punishment for mutual defection
    :param nb_strats: number of strategies
    :return: payoff matrix
    """
    payoffs_analytical = np.zeros((nb_strats, nb_strats))
    with open(filename, "r") as f:
        proba_c_pairs = np.asarray(json.load(f))
        for i in range (nb_strats):
            for j in range (i, nb_strats):
                if  i == j:
                    proba_c = proba_c_pairs[i, j]
                    payoffs_analytical[i, j] = getPayoff(R, S, T, P, proba_c, proba_c)
                else:
                    proba_a = proba_c_pairs[i, j]
                    proba_b = proba_c_pairs[j, i]
                    payoffs_analytical[i, j] = getPayoff(R, S, T, P, proba_a, proba_b)
                    payoffs_analytical[j, i] = getPayoff(R, S, T, P, proba_b, proba_a)
    return payoffs_analytical

def AnalyticalPayoffsFromFile(filename):
    """
    Load a payoff matrix from a file
    :param filename: name of the file in which lie the payoff matrix
    :return: payoff matrix
    """
    with open(filename, "r") as f:
        payoffs_analytical = json.load(f)
        return np.asarray(payoffs_analytical)

def getPayoff(R, S, T, P, proba_c1, proba_c2):
    """
    :param proba_c1: probability of first strategy cooperating
    :param proba_c2: probability of second strategy cooperating
    :return: payoff value an agent obtains after an interaction with another agent
    """
    payoff = 0
    payoff += np.float64(R * proba_c1 * proba_c2)
    payoff += np.float64(S * proba_c1 * (1 - proba_c2))
    payoff += np.float64(T * (1 - proba_c1) * proba_c2)
    payoff += np.float64(P * (1 - proba_c1) * (1 - proba_c2))
    return payoff

def createStrategies(nb_states):
    """
    Generate all possible combination of strategies, depending on the number of states
    :return: list of all strategies in the form [T_c,T_d, action state_0, ..., action state_maxState]
            transition = 1 = Left ; action = 1 = C
    """
    action_choice = list(list(item) for item in product("CD", repeat=nb_states))
    state_change = list(list(item) for item in product("LR", repeat=2))
    strats = []
    for action in action_choice:
        action_c_tr = []
        for i in range(len(action)):

            if action[i] == "C":
                action_c_tr.append(1)
            else:
                action_c_tr.append(0)

        for state_c in state_change:
            state_c_tr = []
            if state_c[0] == "L":
                state_c_tr.append(1)
            else:
                state_c_tr.append(0)
            if state_c[1] == "L":
                state_c_tr.append(1)
            else:
                state_c_tr.append(0)
            list_c_tr = [state_c_tr, action_c_tr]
            strat = [item for sublist in list_c_tr for item in sublist]
            strats.append(strat)
    return strats

def loadPayoffsArrayToNpArrays(filename, strats):
    n = len(strats)
    with open(filename, "r") as f:
        p = [[None for j in range(n)] for i in range(n)]
        for line in f.readlines():
            all = json.loads(line)
            strat_A = all[0]
            strat_B = all[1]
            index_A = strats.index(strat_A)
            if strat_A != strat_B:
                index_B = strats.index(strat_B)
                payoffs_A = all[2]
                payoffs_B = all[3]
                p[index_A][index_B] = payoffs_A
                p[index_B][index_A] = payoffs_B
            else:
                payoffs_A = all[2]
                p[index_A][index_A] = payoffs_A
        return np.asarray(p)

def describeNpArray(payoffs_mat):
    """
    Give the standard metrics of a payoff matrix
    :param payoffs_mat: payoff matrix
    :return: mean, median, std, variance and std mean
    """
    nb_sim = len(payoffs_mat[0][0])
    payoffs_mean = np.mean(payoffs_mat[:,:], axis=2)
    payoffs_median = np.median(payoffs_mat[:,:], axis=2)
    payoffs_std = np.std(payoffs_mat[:,:], axis=2)
    payoffs_var = np.var(payoffs_mat[:, :], axis=2)
    payoffs_std_mean = np.divide(payoffs_std, np.sqrt(nb_sim))
    return payoffs_mean, payoffs_median, payoffs_std, payoffs_var, payoffs_std_mean



def confidenceIntervalForAllPairs(payoffs_mean, payoffs_std, nb_sim, confidence):
    """
    Compute a confidence interval for all pairs of strategies
    :param payoffs_mean: matrix of mean payoffs
    :param payoffs_std: matrix of std payoffs
    :param nb_sim: number of simulation
    :param confidence: confidence level
    :return: confidence intervals
    """
    n = len(payoffs_mean)
    confidence_intervals = np.zeros((n,n,2))
    for i in range (n):
        for j in range (n):
            h = payoffs_std[i, j] * t.ppf((1 + confidence) / 2, nb_sim - 1)
            confidence_intervals[i, j] = [payoffs_mean[i, j] - h, payoffs_mean[i, j] + h]
    return confidence_intervals



def hypothesisTest(strats, confidence_intervals, payoffs_analytical):
    """
    Displays which payoffs for any pair of strategies is in it confidence interval or not
    :param strats: strategies
    :param confidence_intervals: confidence intervals
    :param payoffs_analytical: matrix of payoffs for each pair of strategy
    """
    n = (len(strats))
    for i in range (n):
        for j in range (n):
            if not(payoffs_analytical[i, j] >= confidence_intervals[i, j, 0] and payoffs_analytical[i, j] <= confidence_intervals[i, j, 1]):
                print("payoffs analytical is not is the interval of confidence for strat ",displayableStrat(strats[i])," against strat", displayableStrat(strats[j]))
                print("payoffs analytical is : ", payoffs_analytical[i, j]," --- interval is ", confidence_intervals[i, j])


def displayableStrat(strat):
    displayable_strat = []
    for i in range (len(strat)):
        if strat[i] == 1:
            if i < 2:
                displayable_strat.append("L")
            else:
                displayable_strat.append("C")
        else:
            if i < 2:
                displayable_strat.append("R")
            else:
                displayable_strat.append("D")
    return displayable_strat

if __name__ == '__main__':

    game = "SH"
    R, P = 1, 0
    T = 0.5
    S = -T
    nb_states = 4
    strats = createStrategies(nb_states)
    print(strats)
    proba_filename = "ProbaCouples/" + game + "/proba_couples_matrix_" + str(nb_states) + "st.txt"
    #payoffs_analytical = probaCouplesToPayoffs(proba_filename, R, S, T, P, len(strats))
    payoffs_analytical_filename = "PayoffPairs/" + game + "/10rounds/payoffs_pairs_" + str(nb_states) + "st.txt"
    payoffs_analytical = AnalyticalPayoffsFromFile(payoffs_analytical_filename)

    #payoffs_simulation_2.txt with 100 rounds : better precision overall, but less precise with certain strategies
    #payoffs_simulation.txt with 10 rounds : less good precision overall, but individual errors may be lower than with 100 rounds
    #payoffs_simulation_50.txt with 50 rounds :
    payoffs_filename = "payoffs_simulation_3st.txt"
    #list_df = loadPayoffsArrayToPandasDF(payoffs_filename, strats)
    #df_info_mat_mean, df_info_mat_median, df_info_mat_std = describeDF(list_df)

    payoffs_mat = loadPayoffsArrayToNpArrays(payoffs_filename, strats)
    nb_sim = len(payoffs_mat[0,0])
    payoffs_mean, payoffs_median, payoffs_std, payoffs_var, payoffs_std_mean = describeNpArray(payoffs_mat)
    confidence_intervals_95 = confidenceIntervalForAllPairs(payoffs_mean, payoffs_std, nb_sim, 0.95)
    print("Test with confidence interval 95%")
    hypothesisTest(strats, confidence_intervals_95, payoffs_analytical)
    confidence_intervals_99 = confidenceIntervalForAllPairs(payoffs_mean, payoffs_std, nb_sim, 0.99)
    print("Test with confidence interval 99%")
    hypothesisTest(strats, confidence_intervals_99, payoffs_analytical)


