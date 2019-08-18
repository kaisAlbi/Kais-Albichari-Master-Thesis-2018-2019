import matplotlib.pyplot as plt
from sympy import Symbol, add, mul
import pickle

def loadExprsMonomorphic(filename):
    """
    Load the matrix of expressions for each pair of strategy and each group composition
    :param filename: file in which the expressions are stored
    :return: list of expressions
    """
    with open(filename + ".pickle", "rb") as f:
        exprs_monomorphic = pickle.load(f)
        return exprs_monomorphic

def getStationaryFromFile(filename):
    """
    Load the stationary distributions value from a file
    :param filename: name of the file in which the stationary distributions values are stored
    :return: stationary distributions values
    """
    stationary = []
    with open(filename + ".txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            index = line.index(": ")
            stationary_i = line[index + 1:]
            stationary.append(float(stationary_i))
    return stationary


def findSymbolAndEval(args, symbol):
    """
    :param args: list of argument constituting an expression
    :param symbol: researched symbol
    :return: coefficient associated to the researched symbol and remaining list of argument without it
    """
    value = 0
    found = False
    i = 0
    while not found and i < len(args):
        if symbol in args[i].free_symbols:
            found = True
            value = args[i].subs({symbol: 1})
            args.pop(i)
        i+=1
    return value, args

def getCoopFromExpr(expr, N):
    """
    Extract the cooperation level from an expression
    :param expr: expression
    :param N: number of rounds
    :return: cooperation level
    """
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    P = Symbol('P')
    sum_args = list(expr.args)
    if len(sum_args) == 0: #It's only a symbol
        if expr == R or expr == S:
            coop = 1
        else:
            coop = 0
    else:
        if isinstance(expr, add.Add):
            R_val, sum_args = findSymbolAndEval(sum_args, R)
            S_val, sum_args = findSymbolAndEval(sum_args, S)
            coop = (R_val + S_val) / N
        else:   #Mul
            value = expr.subs({R: 1, S: 1})
            if T in value.free_symbols or P in value.free_symbols:
                coop = 0
            else:
                coop = value/N

    return coop

def coopRatioDifferentN(game, nb_states, N):
    """
    Compute the cooperation ratio for a given group size
    :param game: considered dilemma
    :param nb_states: number of states
    :param N: group size
    :return: cooperation ratio
    """
    expr_filename = "ExprAnalyticalMonomorphic/" + str(nb_states) + "st/expressions_" + str(
        nb_states) + "st_groupsize_" + str(N)
    stationary_filename = "stationaryDistrib/" + game + "/Groups/Different_N/N" + str(N) + "/stationary_" + str(nb_states)+"st"
    exprs_monomorphic = loadExprsMonomorphic(expr_filename)
    stationary = getStationaryFromFile(stationary_filename)
    coop_ratio = 0.
    for i in range(len(exprs_monomorphic)):
        expr = exprs_monomorphic[i]
        coop_monomorphic = getCoopFromExpr(expr, N)
        coop_ratio += coop_monomorphic*stationary[i]
    return round(coop_ratio, 5)

def coopRatioDifferentT(game, nb_states, N, T_folder):
    """
    Compute the cooperation ratio for different temptation to defect T values
    :param game: considered dilemma
    :param nb_states: number of states
    :param N: Group size
    :param T_folder: subfolder corresponding to a T value
    :return: cooperation ratio
    """
    expr_filename = "ExprAnalyticalMonomorphic/" + str(nb_states) + "st/expressions_" + str(
        nb_states) + "st_groupsize_" + str(N)
    stationary_filename = "stationaryDistrib/" + game + "/Groups/Different_T/" + T_folder + "stationary_" + str(nb_states)+"st"
    exprs_monomorphic = loadExprsMonomorphic(expr_filename)
    stationary = getStationaryFromFile(stationary_filename)
    coop_ratio = 0.
    for i in range(len(exprs_monomorphic)):
        expr = exprs_monomorphic[i]
        coop_monomorphic = getCoopFromExpr(expr, N)
        coop_ratio += coop_monomorphic*stationary[i]
    return round(coop_ratio, 5)

def computeCoopDifferentN():
    """
    Compute the cooperation ratio for each one of the two considered game (SH and PD), and for the different group sizes
    And write these into different files
    """
    for game in ["SH", "PD"]:
        for N in [1,2,6,10,20]:
            coop_filename = "EvalCoop/" + game + "/Different_N/N" + str(N) + "/coop_evolution.csv"
            coop_ratios = []
            for nb_states in [1,2,3,4,5]:
                coop_ratios.append(coopRatioDifferentN(game, nb_states, N))
            with open(coop_filename, "w") as f:
                fields = "NbStates,Cooperation\n"
                f.write(fields)
                for i in range (len(coop_ratios)):
                    nb_states = i+1
                    coop_ratio = coop_ratios[i]
                    res = "{},{}".format(nb_states, coop_ratio)
                    res += "\n"
                    f.write(res)

def computeCoopDifferentT():
    """
    Compute the cooperation ratio for each one of the two considered game (SH and PD), and for the different T values
    And write these into different files
    """
    N = 150
    for game in ["SH", "PD"]:
        if game == "SH":
            T_folders = ["T0/", "T025/", "T05/", "T075/", "T1/"]
        else:
            T_folders = ["T1/", "T125/", "T15/", "T175/", "T2/"]
        for T_folder in T_folders:
            coop_filename = "EvalCoop/" + game + "/Different_T/" + T_folder + "coop_evolution.csv"
            coop_ratios = []
            for nb_states in [1,2,3,4,5]:
                coop_ratios.append(coopRatioDifferentT(game, nb_states, N, T_folder))
            with open(coop_filename, "w") as f:
                fields = "NbStates,Cooperation\n"
                f.write(fields)
                for i in range (len(coop_ratios)):
                    nb_states = i+1
                    coop_ratio = coop_ratios[i]
                    res = "{},{}".format(nb_states, coop_ratio)
                    res += "\n"
                    f.write(res)

computeCoopDifferentT()