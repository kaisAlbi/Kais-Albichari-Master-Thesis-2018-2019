from StrategyBuilder import loadStrategies
from FitnessRounds import loadAllExpr
import matplotlib.pyplot as plt
from sympy import Symbol

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


def showInteractionResults(expr, N):
    """
    Plot a bar graph showing the proportion of each possible variable from the payoff matrix for a given strategy
    and a given number of rounds
    :param expr: expression being considered
    :param N: number of rounds
    """
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    P = Symbol('P')
    sum_args = list(expr.args)
    R_val, sum_args = findSymbolAndEval(sum_args, R)
    S_val, sum_args = findSymbolAndEval(sum_args, S)
    T_val, sum_args = findSymbolAndEval(sum_args, T)
    P_val, sum_args = findSymbolAndEval(sum_args, P)
    x = [i for i in range(4)]
    values = [R_val/N, S_val/N, T_val/N, P_val/N]
    fig = plt.figure()
    coop = (R_val + S_val)/N
    defect = (T_val + P_val)/N
    print("coop proportion : ", coop)
    print("defect proportion : ", defect)
    ax = fig.add_subplot(111)
    ax.set_ylabel("payoff distribution")
    strats_ticks = ["R", "S", "T", "P"]
    plt.xticks(x, strats_ticks)
    for i in range (4):
        plt.bar(x[i], values[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


if __name__ == '__main__':

    N = 15
    nb_states= 2
    strats_filename = "Strategies/" + str(nb_states) + "_st_strats"
    strats = loadStrategies(strats_filename)

    expr_filename = "ExprAnalytical/" + str(nb_states) + "states/expressions_" + str(N) + "_rounds_alpha08"

    all_exprs = loadAllExpr(expr_filename)
    selected_expr = all_exprs[24][24] # expression of strategy RL-CD-DD against itself
    showInteractionResults(selected_expr, N)
