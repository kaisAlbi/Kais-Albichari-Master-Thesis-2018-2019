import numpy as np
from itertools import product
from scipy.stats import hypergeom
import time
import pickle
import json
from sympy import Symbol

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
        for i in range (len(action)):

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

def reducedStrategies(strategies):

    all_c = strategies[0]
    all_d = all_c[:2] + strategies[-1][2:]
    reduced_strats = [all_c]
    for strat in strategies:
        if not (hasOnlyOneAction(strat) or hasOnlyOneDirection(strat)):
            reversed_strat = []
            for i in range (2):
                reversed_strat.append((strat[i]+1)%2)   #Bit flip for transitions : LR -> RL
            for i in range (len(strat)-1, 1, -1):       #Reverse actions : CDD -> DDC
                reversed_strat.append(strat[i])
            if reversed_strat not in reduced_strats:
                reduced_strats.append(strat)
    reduced_strats.append(all_d)
    return reduced_strats


def changeState(cur_state, cur_strat, opp_action, alpha):
    """
    Change the current state, depending on this given states, the strategy of the player, the opponent action
    and the probability to change the state
    :param cur_state: current state
    :param cur_strat: strategy of the considered agent
    :param opp_action: opponent action
    :param alpha: probability to change a state
    :return: new state (modified or not)
    """
    max_state = len(cur_strat[2:]) - 1
    dir_index = 0 if opp_action == 1 else 1
    direction = cur_strat[dir_index]
    if direction == 1 and cur_state > 0:       #possible to go left
        if np.random.random() < alpha:
            cur_state -= 1
    elif direction == 0 and cur_state < max_state:
        if np.random.random() < alpha:
            cur_state += 1
    return cur_state

def hasOnlyOneDirection(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible transition direction, False otherwise
    """
    directions = strat[:2]
    if 1 not in directions or 0 not in directions:
        return True
    return False

def hasOnlyOneAction(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible action choice, False otherwise
    """
    actions = strat[2:]
    if 1 not in actions or 0 not in actions:
        return True
    return False

def getPayoff(first, second, R, S, T, P):
    """
    :param first: first agent
    :param second: second agent
    :return: payoff value an agent obtains after an interaction with another agent
    :return: payoff value an agent obtains after an interaction with another agent
    """
    if first == 1:
        if second == 1:
            return R
        else:
            return S
    else:
        if second == 1:
            return T
        else:
            return P

def loadAllExpr(filename):
    """
    Load the matrix of expressions for each pair of strategy and each group composition
    :param filename: file in which the expressions are stored
    :return: matrix of expressions
    """
    with open(filename, "rb") as f:
        all_exprs = pickle.load(f)
        return all_exprs

def computeFitDiffMatrix(strategies, Z, N, expr_filename, alpha, R, S, T, P):
    """
    Computes the fitness difference, for each pair invader-resident of strategy, and for each number of possible
    invaders in the population
    :param strategies: all strategies
    :param Z: size of the population
    :param expr_filename: file in which the expressions are stored
    :param alpha: probability to change a state after an interaction
    :param R: reward value
    :param S: sucker value
    :param T: temptation to defect value
    :param P: punishment value
    :return: array of all fitness differences
    """
    all_exprs = loadAllExpr(expr_filename)
    n = len(strategies)
    fit_diff_matrix = np.zeros((n, n, Z))

    for i in range(n):
        strat_B = strategies[i]
        print("Strat ", strat_B, " resident")
        start_time_B = time.time()
        for j in range(n):
            if i != j:
                expr_a_a = all_exprs[j][j]
                expr_a_b = all_exprs[j][i]
                expr_b_a = all_exprs[i][j]
                expr_b_b = all_exprs[i][i]

                fit_diff = computeFitDiff(Z, N, [expr_a_a, expr_a_b, expr_b_a, expr_b_b], alpha, R, S, T, P)
                fit_diff_matrix[i, j] = fit_diff
            else:
                fit_diff_matrix[i, i] = np.nan
        print("fit diff calculations for resident strat ", strat_B,
              " took --- %s seconds---" % (time.time() - start_time_B))
    return fit_diff_matrix


def computeFitDiff(Z, N, exprs, alpha, R, S, T, P):
    """
    Compute the fitness difference for a given pair of strategies. The corresponding expressions are given for this pair
    :param Z: size of the population
    :param N: size of the group
    :param exprs: array of expressions containing the expressions of A against A, A against B, B against A and B against B
    :param alpha: probability to change the state after an interaction
    :param R: reward value
    :param S: sucker value
    :param T: temptation to defect value
    :param P: punishment value
    :return: fitness differences for the pair strategies A-B, where A is the invader
    """
    expr_a_a, expr_a_b, expr_b_a, expr_b_b = exprs[0], exprs[1], exprs[2], exprs[3]
    fitness_pairs = np.zeros(Z)

    for i in range (1, Z):                #i = nb invaders

        distrib_group = hypergeom(Z, i, N)
        nb_invaders_arr = np.arange(0, N+1, dtype=np.int32)
        #print("nb invaders arr : ", nb_invaders_arr)
        # There can be 0 up to N invaders (generally speaking)
        group_pmf = distrib_group.pmf(nb_invaders_arr)
        inv_result = 0.
        res_result = 0.


        for group_id in range (N+1):
            if group_pmf[group_id] != 0:            #No need to compute the avg payoffs of such group if its proba(appearance)=0
                if group_id == 0:   #No invaders
                    inv_result += computePayoffFromExpr(expr_a_b[group_id], N, alpha, R, S, T, P) * group_pmf[group_id]
                    res_result += computePayoffFromExpr(expr_b_b[0], N, alpha, R, S, T, P) * group_pmf[group_id]

                elif group_id == N:     #Only invaders
                    inv_result += computePayoffFromExpr(expr_a_a[0], N, alpha, R, S, T, P) * group_pmf[group_id]
                    res_result += computePayoffFromExpr(expr_b_a[-group_id], N, alpha, R, S, T, P) * group_pmf[group_id]
                else:  # Mixed group
                    inv_result += computePayoffFromExpr(expr_a_b[group_id], N, alpha, R, S, T, P) * group_pmf[group_id]
                    res_result += computePayoffFromExpr(expr_b_a[-group_id], N, alpha, R, S, T, P) * group_pmf[group_id]

        fit_diff = res_result - inv_result
        fitness_pairs[i-1] = fit_diff

    return fitness_pairs


def computePayoffFromExpr(expr, N, alpha, R, S, T, P):
    """
    Replace the variable in the expressions by the corresponding values to compute the payoff value
    :param expr: expressions to be evaluated
    :param N: size of the group
    :param alpha: probability to change the state after an interaction
    :param R: reward value
    :param S: sucker value
    :param T: temptation to defect value
    :param P: punishment value
    :return: evaluation of the expression
    """
    alpha_var = Symbol('a')
    R_var = Symbol('R')
    S_var = Symbol('S')
    T_var = Symbol('T')
    P_var = Symbol('P')
    #print("1",expr)
    #print("2", expr.subs({R_var: R})/(N))
    return expr.subs({R_var: R, S_var: S, T_var: T, P_var: P, alpha_var: alpha})/N

def storePickle(filename, fit_diff_matrix):
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(fit_diff_matrix, f)

def storeJson(filename, fit_diff_matrix):
    store_filename = filename + ".txt"
    with open(store_filename, "w") as f:
        json.dump(fit_diff_matrix.tolist(), f)


if __name__ == '__main__':

    game = "HD"
    #R, S, T, P = 1, -0.5,1.5, 0
    #R, P = 1, 0
    alpha = 0.8
    N = 150
    Z = 150
    nb_states = 3
    strats = createStrategies(nb_states)
    r_strats = reducedStrategies(strats)
    T_list = [0,0.25,0.5,0.75,1]
    T_subf = ["T0/", "T025/", "T05/", "T075/", "T1/"]
    V = 2.
    for i in range (len(T_list)):
        T = T_list[i]
        C = V + 0.5
        R = float(V - C) / 2
        S = V
        P = V / 2
        expr_filename = "ExpressionsAnalytical/Reduced/expressions_" + str(nb_states) + "st_groupsize_" + str(
                    N) + "_alpha08.pickle"
        start_time = time.time()

        fit_diff_matrix = computeFitDiffMatrix(r_strats, Z, N, expr_filename, alpha, R, S, T, P)
        store_filename = "FitDiff/" + game + "/" + T_subf[i] + "fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
        storePickle(store_filename, fit_diff_matrix)
    #nb_states = 1
    #for nb_states in [1,2]:
    #    strats = createStrategies(nb_states)
    #    r_strats = reducedStrategies(strats)
    #    for N in [1,2,6,10,20,75,150]:

    #        expr_filename = "ExpressionsAnalytical/Reduced/expressions_" + str(nb_states) + "st_groupsize_" + str(
    #            N) + "_alpha08.pickle"
    #        start_time = time.time()

    #        fit_diff_matrix = computeFitDiffMatrix(r_strats, Z, N, expr_filename, alpha, R, S, T, P)
    #        store_filename = "FitDiff/" + game + "/Reduced/fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
    #        storePickle(store_filename, fit_diff_matrix)


    """
    #expr_filename = "ExpressionsAnalytical/expressions_" + str(nb_states) + "st_groupsize_" + str(N) + "_alpha08.pickle"
    expr_filename = "ExpressionsAnalytical/Reduced/expressions_" + str(nb_states) + "st_groupsize_" + str(N) + "_alpha08.pickle"
    #expr_filename = "ExpressionsAnalytical/LRCD_RLCD_groupsize_"+str(N)+"_alpha08.pickle"

    strategies = createStrategies(nb_states)
    r_strats = reducedStrategies(strategies)
    #r_strats = [[1,0,1,0],[0,1,1,0]]
    t_folders = ["T0/", "T025/", "T05/", "T075/", "T1/"]
    t_values = [0, 0.25, 0.5, 0.75, 1]
    for i in range (len(t_values)):
        folder = t_folders[i]
        t = t_values[i]
        s = -t
        fit_diff_matrix = computeFitDiffMatrix(r_strats, Z, N, expr_filename, alpha, R, s, t, P)
        store_filename = "FitDiff/" + game + "/"+folder+"fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
        storePickle(store_filename, fit_diff_matrix)
    print("SH done")
    game = "PD"
    t_folders = ["T1/", "T125/", "T15/", "T175/", "T2/"]
    t_values = [1, 1.25, 1.5, 1.75, 2]
    for i in range (len(t_values)):
        folder = t_folders[i]
        t = t_values[i]
        s = 1-t
        fit_diff_matrix = computeFitDiffMatrix(r_strats, Z, N, expr_filename, alpha, R, s, t, P)
        store_filename = "FitDiff/" + game + "/"+folder+"fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
        storePickle(store_filename, fit_diff_matrix)
    print("PD done")
    #fit_diff_matrix = computeFitDiffMatrix(r_strats, Z, N, expr_filename, alpha, R, S, T, P)


    #store_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
    #store_filename = "FitDiff/" + game + "/Reduced/fit_diff_" + str(nb_states) + "st_groupsize" + str(N)
    #store_filename = "FitDiff/" + game + "/analytical_groups/fit_diff_LRCD_RLCD_groupsize" + str(N)

    #storePickle(store_filename, fit_diff_matrix)
    """