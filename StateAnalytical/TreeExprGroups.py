from TreeStrategyGroups import Tree
from itertools import product
from sympy import Symbol
from sympy.utilities.iterables import multiset_permutations
import time
import pickle

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

def createGroupsConfig(N):
    """
    Create all possible combinations of invaders-residents in a group of size N-1
    :param N: Size of group (containing the strategy playing against the others)
    :return: matrix of groups ordered by the number of invaders in the groups (value 0 is the invader)
    """
    all_groups = list(list(item) for item in product(range(2), repeat=N-1))

    ordered_groups = [[] for i in range (N)]
    for group in all_groups:
        nb_0 = group.count(0)
        ordered_groups[nb_0].append(group)
    return ordered_groups

def getPayoff(first, second, R, S, T, P):
    """
    :param first: first agent
    :param second: second agent
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

def buildExpr(my_action, opp_action):
    """
    Choose the variable associated to the result of the interaction
    :param my_action: action of current strategy
    :param opp_action: opponent action
    :return: Symbol object corresponding to the variable
    """
    if my_action == 1:
        if opp_action == 1:
            return Symbol('R')
        else:
            return Symbol('S')
    else:
        if opp_action == 1:
            return Symbol('T')
        else:
            return Symbol('P')

def computeExprGroups(strategies, N, nb_states):
    """
    Compute the expressions for each group, for each pair of strategies
    :param strategies: all the strategies
    :param N: size of the groups
    :param nb_states: number of states
    :return: array of each expression
    """
    all_expr_groups = []
    for strat in strategies:
        start_time = time.time()
        expr_strat = []
        for opp_strat in strategies:
            print(strat, " against ",opp_strat)
            if strat == opp_strat:

                if hasOnlyOneAction(strat) or hasOnlyOneDirection(strat):
                    print("1")
                    direction = strat[0]
                    action = strat[2] if direction == 1 else strat[-1]
                    expr = buildExpr(action, action) * (N-1)
                else:
                    print("2")
                    expr = computeExprAgainstItself(strat, N)
                expr_strat.append([expr])
            else:

                expr_diff_compos = []
                if hasOnlyOneAction(strat) or hasOnlyOneDirection(strat):
                    direction = strat[0]
                    action = strat[2] if direction == 1 else strat[-1]
                    if hasOnlyOneAction(opp_strat) or hasOnlyOneDirection(opp_strat):       #Action is fixed
                        print("3")
                        opp_direction = opp_strat[0]
                        opp_action = opp_strat[2] if opp_direction == 1 else opp_strat[-1]
                        for i in range (N-1):     #At least one same strat, at most N-2 included
                            expr = buildExpr(action, action) * i + buildExpr(action, opp_action) * (N-1-i)
                            expr_diff_compos.append(expr)

                    else:           #Action can change over time
                        print("4")
                        for i in range (N-1):
                            #print("loop ",i, " of ", N-1)


                            expr = buildExpr(action, action)*i
                            opp_actions = opp_strat[:2]
                            var_states_expr = 0
                            for opp_action in opp_actions:
                                var_states_expr += (N-1-i)*buildExpr(action, opp_action)
                            var_states_expr /= nb_states
                            expr += var_states_expr
                            expr_diff_compos.append(expr)
                            #states_config = list(list(item) for item in product(range(nb_states), repeat=N-1-i))
                            #print(len(states_config))
                            #var_states_expr = 0
                            #for states in states_config:
                            #    #print("loop")
                            #    for state in states:
                            #        opp_action = opp_strat[2+state]
                            #        var_states_expr += buildExpr(action, opp_action)
                            #print(var_states_expr)
                            #expr += (var_states_expr/len(states_config))
                            expr_diff_compos.append(expr)

                else:
                    print("5")
                    #print("computeExprTreeActionFixed")
                    #ordered_groups = createGroupsConfig(N)
                    ordered_groups = []
                    #print(len(ordered_groups))
                    if hasOnlyOneAction(opp_strat) or hasOnlyOneDirection(opp_strat):
                        #print("8")
                        opp_direction = opp_strat[0]
                        opp_action = opp_strat[2] if opp_direction == 1 else opp_strat[-1]
                        expr_diff_compos = computeExprTreeActionFixed(strat, opp_action, ordered_groups, N)

                    else:
                        #print("9")
                        expr_diff_compos = computeExprTreeComplete(strat, opp_strat, ordered_groups, N)

                expr_strat.append(expr_diff_compos)
        all_expr_groups.append(expr_strat)
        print("Expression for strat ", strat, " for group size ", N, " computed in --- %s seconds --- " % (time.time() - start_time))
    return all_expr_groups

def computeExprAgainstItself(strategy, N):
    """
    Compute the expression of the payoff a strategy will obtain when facing a group of only the same strategy
    :param strategy: strategy being analyzed
    :param N: size of group
    :return: expression of strategy against itself
    """
    nb_states = len(strategy[2:])
    all_states_config = list(list(item) for item in product(range(nb_states), repeat=N))    #First element of each array will be the agent playing against everyone else
    expr_all_states = 0.
    for states_config in all_states_config:
        root = Tree(states_config[0], strategy, 1)

        opp_action = strategy[2+states_config[1]]
        root.buildExpr(opp_action)
        root.addChildren(opp_action)
        current_level = root.getChildren()

        expr_state_config = root.getExpr()
        for j in range (2, N):
            opp_action = strategy[2+states_config[j]]
            next_level = []
            for child in current_level:
                child.buildExpr(opp_action)
                child.addChildren(opp_action)
                expr_state_config += child.getExpr()
                for new_child in child.getChildren():
                    next_level.append(new_child)
            current_level = list(next_level)
        expr_all_states += expr_state_config
    expr_all_states /= len(all_states_config)
    return expr_all_states



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



def computeExprTreeComplete(cur_strat, other_strat, ordered_groups, N):
    """
    Compute the expression of a strategy against another strategy, knowing that their action vary over time
    :param cur_strat: current strategy
    :param other_strat: opponent strategy
    :param ordered_groups: combination of all possible groups of size (N-1) of cur_strat and other_strat
    :param N: size of the group
    :return: array of expressions for each combination of the groups
    """
    nb_states = len(cur_strat[2:])
    expr_all_opponent_nb = []
    for i in range (N-1):            #Only against one same strat up to N-1 in group of N containing opponent strat
        #print("loop ", i)
        strat_loop = time.time()
        opponents = i * [0] + (N - 1 - i) * [1]
        groups_comb = list(multiset_permutations(opponents))
        #print(len(groups_comb))
        #groups = ordered_groups[i]
        expr_groups = 0.
        for group in groups_comb:
            expr_all_states = 0.
            for state in range(nb_states):
                root = Tree(state, cur_strat, 1)
                if group[0] == 1:       #If first element has the opponent strat
                    root.buildExprVariableActionsStrat(other_strat)
                    root.addChildrenVariableAction(other_strat)
                else:
                    root.buildExprVariableActionsStrat(cur_strat)
                    root.addChildrenVariableAction(cur_strat)
                current_level = root.getChildren()

                expr_state_config = root.getExpr()
                for j in range(1, N-1):
                    if group[j] == 1:
                        next_level = []
                        for child in current_level:
                            child.buildExprVariableActionsStrat(other_strat)
                            child.addChildrenVariableAction(other_strat)
                            expr_state_config += child.getExpr()
                            for new_child in child.getChildren():
                                next_level.append(new_child)
                        current_level = list(next_level)
                    else:
                        next_level = []
                        for child in current_level:
                            child.buildExprVariableActionsStrat(cur_strat)
                            child.addChildrenVariableAction(cur_strat)
                            expr_state_config += child.getExpr()
                            for new_child in child.getChildren():
                                next_level.append(new_child)
                        current_level = list(next_level)

                # expr_state_config /= N-1       #in comment because I want the total payoff expression after N rounds
                expr_all_states += expr_state_config
            expr_all_states /= nb_states
            expr_groups += expr_all_states

        expr_groups /= len(groups_comb)      #Avg expression of strategy A against group of A/B's of size N-1 with i B's in it
        expr_all_opponent_nb.append(expr_groups)
        print("Loop computeExprTreeComplete ", i, " of " , N-1,
              " computed in --- %s seconds --- " % (time.time() - strat_loop))
    return expr_all_opponent_nb

def computeExprTreeActionFixed(cur_strat, opp_action, ordered_groups, N):
    """
    Compute the expression of a strategy against another strategy, knowing that the opponent's action is fixed
    :param cur_strat: current strategy
    :param opp_action: opponent action
    :param ordered_groups: combination of all possible groups of size (N-1) of cur_strat and other_strat
    :param N: size of the group
    :return: array of expressions for each combination of the groups
    """
    nb_states = len(cur_strat[2:])

    expr_all_opponent_nb = []
    print(len(ordered_groups))
    for i in range (N-1):            #Only against 0same strat up to N-1 same strat (the rest in group is opponent strat)
        print("loop ",i)
        opponents = i*[0] + (N-1-i)*[1]
        groups_comb = list(multiset_permutations(opponents))
        print(len(groups_comb))
        strat_loop = time.time()
        expr_groups = 0.
        for group in groups_comb:

            expr_all_states = 0.
            for state in range (nb_states):
                root = Tree(state, cur_strat, 1)
                if group[0] == 1:       #If first element has the opponent strat
                    root.buildExpr(opp_action)
                    root.addChildren(opp_action)
                else:
                    root.buildExprVariableActionsStrat(cur_strat)
                    root.addChildrenVariableAction(cur_strat)
                current_level = root.getChildren()

                expr_state_config = root.getExpr()
                for j in range (1, N-1):
                    if group[j] == 1:
                        next_level = []
                        for child in current_level:
                            child.buildExpr(opp_action)
                            child.addChildren(opp_action)
                            expr_state_config += child.getExpr()
                            for new_child in child.getChildren():
                                next_level.append(new_child)
                        current_level = list(next_level)
                    else:
                        next_level = []
                        for child in current_level:
                            child.buildExprVariableActionsStrat(cur_strat)
                            child.addChildrenVariableAction(cur_strat)
                            expr_state_config += child.getExpr()
                            for new_child in child.getChildren():
                                next_level.append(new_child)
                        current_level = list(next_level)
                expr_all_states += expr_state_config
            expr_all_states /= nb_states
            expr_groups += expr_all_states
        expr_groups /= len(groups_comb)      #Avg expression of strategy A against group of A/B's of size N-1 with i B's in it
        print(expr_groups)
        expr_all_opponent_nb.append(expr_groups)
        print("Loop computeExprTreeActionFixed ", i, " of ", N - 1,
              " computed in --- %s seconds --- " % (time.time() - strat_loop))
    print(len(expr_all_opponent_nb))
    return expr_all_opponent_nb


def storePickle(filename, all_expr_groups):
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(all_expr_groups, f)

#def storeJson(filename, all_expr_groups):
#    store_filename = filename + ".txt"
#    modified_expr_groups = [[str(expr) for expr in expr_groups] for expr_groups in all_expr_groups]
#    with open(store_filename, "w") as f:
#        json.dump(modified_expr_groups, f)

if __name__ == '__main__':
    N = 5
    nb_states = 2
    #strats = createStrategies(nb_states)
    strats = [[1,0,1,0],[1,1,0,1]]
    all_expr_groups = computeExprGroups(strats, N, nb_states)

    #filename = "expressions_" + str(nb_states) + "st_groupsize_" + str(N)
    filename = "ExpressionsAnalytical/test_LRCD_AllD"

    storePickle(filename, all_expr_groups)