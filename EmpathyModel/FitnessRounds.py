import numpy as np
import time
import pickle
import json
from sympy import Symbol
from StrategyBuilder import loadStrategies



def loadAllExpr(filename):
    """
    Load the matrix of expressions for each pair of strategy and each group composition
    :param filename: file in which the expressions are stored
    :return: matrix of expressions
    """
    with open(filename+".pickle", "rb") as f:
        all_exprs = pickle.load(f)
        return all_exprs

def computeFitDiffMatrix(strategies, Z, N, expr_filename, R, S, T, P):
    """
    Computes the fitness difference, for each pair invader-resident of strategy, and for each number of possible
    invaders in the population
    :param strategies: all strategies
    :param Z: size of the population
    :param expr_filename: file in which the expressions are stored
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
        start_time_B = time.time()
        for j in range(n):
            if i != j:
                expr_a_a = all_exprs[j][j]
                expr_a_b = all_exprs[j][i]
                expr_b_a = all_exprs[i][j]
                expr_b_b = all_exprs[i][i]

                fit_diff = computeFitDiff(Z, N, [expr_a_a, expr_a_b, expr_b_a, expr_b_b], R, S, T, P)
                fit_diff_matrix[i, j] = fit_diff
            else:
                fit_diff_matrix[i, i] = np.nan
        print("fit diff calculations for resident strat ", strat_B,
              " (strat ", i+1,"/", len(strategies), ") took --- %s seconds---" % (time.time() - start_time_B))

    return fit_diff_matrix


def computeFitDiff(Z, N, exprs, R, S, T, P):
    """
    Compute the fitness difference for a given pair of strategies. The corresponding expressions are given for this pair
    :param Z: size of the population
    :param N: size of the group
    :param exprs: array of expressions containing the expressions of A against A, A against B, B against A and B against B
    :param R: reward value
    :param S: sucker value
    :param T: temptation to defect value
    :param P: punishment value
    :return: fitness differences for the pair strategies A-B, where A is the invader
    """
    expr_a_a, expr_a_b, expr_b_a, expr_b_b = exprs[0], exprs[1], exprs[2], exprs[3]
    fitness_pairs = np.zeros(Z)
    p_inv_inv = computePayoffFromExpr(expr_a_a, N, R, S, T, P)
    p_inv_res = computePayoffFromExpr(expr_a_b, N, R, S, T, P)
    p_res_res = computePayoffFromExpr(expr_b_b, N, R, S, T, P)
    p_res_inv = computePayoffFromExpr(expr_b_a, N, R, S, T, P)

    for i in range (1, Z):                #i = nb invaders
        result_inv = (i * p_inv_inv + (Z - i) * p_inv_res) / float(Z)
        result_res = (i * p_res_inv + (Z - i) * p_res_res) / float(Z)
        fit_diff = result_res - result_inv
        fitness_pairs[i-1] = fit_diff
    return fitness_pairs


def computePayoffFromExpr(expr, N, R, S, T, P):
    """
    Replace the variable in the expressions by the corresponding values to compute the payoff value
    :param expr: expressions to be evaluated
    :param N: number of rounds
    :param R: reward value
    :param S: sucker value
    :param T: temptation to defect value
    :param P: punishment value
    :return: evaluation of the expression
    """
    R_var = Symbol('R')
    S_var = Symbol('S')
    T_var = Symbol('T')
    P_var = Symbol('P')
    return expr.subs({R_var: R, S_var: S, T_var: T, P_var: P})/N

def storePickle(filename, fit_diff_matrix):
    """
    Store the strategies in a .pickle file
    :param strats: strategies
    :param filename: name of the created file
    """
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(fit_diff_matrix, f)

def storeJson(filename, fit_diff_matrix):
    store_filename = filename + ".txt"
    with open(store_filename, "w") as f:
        json.dump(fit_diff_matrix.tolist(), f)


if __name__ == '__main__':

    game = "PD"
    R, S, P = 1, -0.5, 0
    T = 0.5
    if game == "PD":
        T = 1.5

    alpha = 0.8
    N = 1
    Z = 150

    nb_states = 2
    nb_signals = nb_states
    strats_filename = "Strategies/" + str(nb_states) + "_st_strats"
    strats = loadStrategies(strats_filename)
    expr_filename = "ExprAnalytical/" + str(nb_states) + "states/expressions_" + str(N) + "_rounds_alpha08"

    all_exprs = loadAllExpr(expr_filename)

    start_time = time.time()
    fit_diff_matrix = computeFitDiffMatrix(strats, Z, N, expr_filename, alpha, R, S, T, P)
    print("all fit diff for ", N,
          " rounds computed in --- %s seconds --- " % (time.time() - start_time))
    store_filename = "FitDiff/" + game + "/" + str(nb_states) + "states/fit_diff_" + str(N) + "_rounds_alpha08"
    storePickle(store_filename, fit_diff_matrix)
