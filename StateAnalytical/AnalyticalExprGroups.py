from itertools import product
from sympy import Symbol
import time
import pickle

alpha = 0.8

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
    """
    Remove the symmetric strategies from the set of strategies
    :param strategies: list of all strategies
    :return: reduced list of strategies
    """
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
            if strat == opp_strat:
                if hasOnlyOneAction(strat) or hasOnlyOneDirection(strat):
                    direction = strat[0]
                    action = strat[2] if direction == 1 else strat[-1]
                    expr = buildExpr(action, action) * N
                else:
                    expr = computeExprSolo(strat, N)
                expr_strat.append([expr])
            else:
                expr_diff_compos = []
                if hasOnlyOneAction(strat) or hasOnlyOneDirection(strat):
                    direction = strat[0]
                    action = strat[2] if direction == 1 else strat[-1]
                    if hasOnlyOneAction(opp_strat) or hasOnlyOneDirection(opp_strat):       #Action is fixed
                        opp_direction = opp_strat[0]
                        opp_action = opp_strat[2] if opp_direction == 1 else opp_strat[-1]
                        for i in range (N):     #At least one same strat, at most N-2 included
                            expr = buildExpr(action, action) * i + buildExpr(action, opp_action) * (N-i)
                            expr_diff_compos.append(expr)

                    else:           #opponent action can change over time
                        for i in range (N):
                            opp_actions = opp_strat[:2]
                            count_C_opp = opp_actions.count(1)
                            p_c_opp = count_C_opp / nb_states
                            expr = buildExpr(action, action)*i + (N-i) * buildExprProba(action, p_c_opp)
                            #Action is fixed here, so if action == 1(C) p(C) = 1, if action == 0(D), p(C) = 0

                            expr_diff_compos.append(expr)

                else:
                    if hasOnlyOneAction(opp_strat) or hasOnlyOneDirection(opp_strat):  # Action is fixed
                        opp_direction = opp_strat[0]
                        opp_action = opp_strat[2] if opp_direction == 1 else opp_strat[-1]
                        for i in range (N):
                            expr_diff_compos.append(computeExprComplexOppfixed(strat, opp_action, i, N))
                    else:
                        for i in range (N):
                            expr_diff_compos.append(computeExprComplexBoth(strat, opp_strat, i, N))
                expr_strat.append(expr_diff_compos)
        all_expr_groups.append(expr_strat)
        print("Expression for strat ", strat, " for group size ", N, " computed in --- %s seconds --- " % (time.time() - start_time))
    return all_expr_groups

def computeNewState(cur_state, max_state, transition):
    """
    Compute the potential new state, given a current_state, the upper bound for a state value, and the transition direction
    :param cur_state: current state
    :param max_state: upper bound for a state value
    :param transition: transition direction
    :return: new state value
    """
    new_state = cur_state
    if cur_state > 0 and transition == 1:      #1 if Left
        new_state -= 1
    elif cur_state < max_state and transition == 0:   #0 if Right
        new_state += 1
    return new_state

def updateProbaStates(cur_strat, p_c_opp, p_states):
    """
    Update the probabilities of a strategy being in each state
    :param strat: considered strategy
    :param p_c_opp: probability of cooperation of the opponent
    :param p_states: probability of being in each state before interaction
    :return: updated probabilities of being in each state
    """
    res_p_states = [0 for i in range (len(p_states))]#np.zeros(len(p_states))
    max_state = len(p_states) - 1
    transition_c = cur_strat[0]  # 1 if Left, 0 if Right
    transition_d = cur_strat[1]  # 1 if Left, 0 if Right
    a = alpha
    for i in range (len(p_states)):

        new_state_c = computeNewState(i, max_state, transition_c)      #new state after the opponent has cooperated (could be the same if it is a outer state (0 or n-1)
        new_state_d = computeNewState(i, max_state, transition_d)      #new state after the opponent has defected (could be the same if it is a outer state (0 or n-1)

        proba_new_state_c = a * p_c_opp
        proba_new_state_d = a * (1 - p_c_opp)
        proba_no_change = round(1-a,2)

        res_p_states[new_state_c] += proba_new_state_c * p_states[i]
        res_p_states[new_state_d] += proba_new_state_d * p_states[i]
        res_p_states[i] += proba_no_change * p_states[i]

    return res_p_states

def computeExprSolo(strat, N):
    """
    Compute expression of a strategy playing in a monomorphic group
    :param strat: strategy being played
    :param N: group size
    :return: expression
    """
    nb_states = len(strat[2:])
    states_distrib = 1 / nb_states
    count_C = strat[2:].count(1)
    p_c = count_C / nb_states  #proba(Cooperation)
    p_states = [states_distrib for i in range (nb_states)]  #Proba (being in each state)

    expr = buildExprForGroup(strat, p_c, p_c, p_states, N)
    return expr



def computeExprComplexOppfixed(strat, opp_action, i, N):
    """
    Compute the expression for a group in which the other strategy has a fixed action
    :param strat: strategy being analysed
    :param opp_action: opponent action
    :param i: number of strat in the group
    :param N: group size
    :return: expression
    """
    nb_states = len(strat[2:])
    states_distrib = 1 / nb_states
    count_C = strat[2:].count(1)
    p_c = count_C / nb_states  # proba(Cooperation)

    p_c_opp_group = ((i/N) * p_c)  + (((N-i)/N) * opp_action) #Average probability of cooperation within a group of i strat and N-i-1 opp_strat
    p_states = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)

    expr = buildExprForGroup(strat, p_c, p_c_opp_group, p_states, N)
    return expr

def computeExprComplexBoth(strat, opp_strat, i, N):
    """
    Compute the expression for a group with two strategies in it
    :param strat: strategy being analysed
    :param opp_strat: opponent strategy
    :param i: number of strat in the group
    :param N: group size
    :return: expression
    """
    nb_states = len(strat[2:])
    states_distrib = 1 / nb_states
    count_C = strat[2:].count(1)
    count_C_opp = opp_strat[2:].count(1)
    p_c = count_C / nb_states  # proba(Cooperation)
    p_c_opp = count_C_opp / nb_states
    p_c_opp_group = ((i/N) * p_c)  + (((N-i)/N) * p_c_opp) #Average probability of cooperation within a group of i strat and N-i-1 opp_strat

    p_states = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)

    expr = buildExprForGroup(strat, p_c, p_c_opp_group, p_states, N)
    return expr

def buildExprProba(proba_c1, proba_c2):
    """
    :param proba_c1: probability of first agent cooperating
    :param proba_c2: probability of second agent cooperating
    :return: expression
    """
    R = Symbol('R')
    S = Symbol('S')
    T = Symbol('T')
    P = Symbol('P')
    expr = 0
    expr += R * proba_c1 * proba_c2
    expr += S * proba_c1 * (1 - proba_c2)
    expr += T * (1 - proba_c1) * proba_c2
    expr += P * (1 - proba_c1) * (1 - proba_c2)
    return expr



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

def buildExprForGroup(cur_strat, p_c, p_c_opp, p_states, N):
    """
    Build an expr for a polymorphic group
    :param cur_strat: strategy being analysed
    :param p_c: probability of cooperation of cur_strat
    :param p_c_opp: probability of cooperation of the opponent strategy
    :param p_states: list containing the probabilities of being in each internal state
    :param N: group size
    :return: expression
    """
    expr = 0
    actions = cur_strat[2:]
    for i in range (N):
        expr += buildExprProba(p_c, p_c_opp)
        p_states = updateProbaStates(cur_strat, p_c_opp, p_states)

        #Update probability of cooperation for current agent
        p_c = 0
        for j in range (len(actions)):
            if actions[j] == 1:
                p_c += p_states[j]
    return expr


def computeExprMonomorphic(strategies, N):
    """
    Compute expressions for monomorphic scenarios
    :param strategies: strategies
    :param N: group size
    :return: expressions
    """
    all_expr_monomorphic = []
    for strat in strategies:
        start_time = time.time()
        if hasOnlyOneAction(strat) or hasOnlyOneDirection(strat):
            direction = strat[0]
            action = strat[2] if direction == 1 else strat[-1]
            expr = buildExpr(action, action) * N
        else:
            expr = computeExprSolo(strat, N)
        all_expr_monomorphic.append(expr)

        print("Expression for strat ", strat, " for group size ", N,
              " computed in --- %s seconds --- " % (time.time() - start_time))
    return all_expr_monomorphic

def storePickle(filename, all_expr_groups):
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(all_expr_groups, f)


if __name__ == '__main__':
    #N = 150
    #nb_states = 2

    #strats = createStrategies(nb_states)
    #r_strats = reducedStrategies(strats)
    #print("nb strats : ", len(strats))
    #print("nb reduced strats : ", len(r_strats))
    #r_strats = [[1,0,1,0],[0,1,1,0]]

    N = 150
    for nb_states in [1,2,3,4,5]:

    #for N in [1,2,6,10,20]:
        strats = createStrategies(nb_states)
        r_strats = reducedStrategies(strats)
        start_time = time.time()
        exprs_monomorphic = computeExprMonomorphic(r_strats, N)

        filename = "ExprAnalyticalMonomorphic/" +  str(nb_states) + "st/expressions_" + str(nb_states) + "st_groupsize_" + str(N)
        storePickle(filename, exprs_monomorphic)
        #all_expr_groups = computeExprGroups(r_strats, N, nb_states)
        #print("all expressions for all groups of size ", N,
        #      " computed in --- %s seconds --- " % (time.time() - start_time))



        #filename = "ExpressionsAnalytical/Reduced/expressions_" + str(nb_states) + "st_groupsize_" + str(N) + "_alpha08"
        #filename = "ExpressionsAnalytical/LRCD_RLCD_groupsize_"+str(N)+"_alpha08"

        #print("store pickle")
        #storePickle(filename, all_expr_groups)
        #print("store json")
        #storeJson(filename, all_expr_groups)
        #print("finish")