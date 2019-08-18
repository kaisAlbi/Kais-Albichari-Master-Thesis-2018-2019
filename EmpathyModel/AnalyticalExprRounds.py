from sympy import Symbol
from StrategyBuilder import loadStrategies, hasOnlyOneDirection, hasOnlyOneActionInState, hasOnlyOneAction
import time
import pickle
import numpy as np
alpha = 0.8

def computeExprRounds(strategies, N, nb_states):
    """
    Compute the expressions for each group, for each pair of strategies
    :param strategies: all the strategies
    :param N: number of rounds
    :param nb_states: number of states
    :return: array of each expression
    """
    all_expr_groups = []
    for i in range (len(strategies)):# in strategies:
        strat = strategies[i]
        start_time = time.time()
        expr_strat = []
        for opp_strat in strategies:
            #strat playing against itself
            if np.array_equal(strat, opp_strat):
                #having one possible action
                if hasOnlyOneAction(strat):
                    action = strat[1][0][0]
                    expr = buildExpr(action, action) * N
                #having one possible direction
                elif hasOnlyOneDirection(strat):
                    direction = strat[0][0]
                    converging_state = 0 if direction == 1 else nb_states-1
                    opp_signal = converging_state #Since opp_strat == strat
                    action = strat[1][converging_state][opp_signal]
                    expr = buildExpr(action, action) * N
                else:
                    expr = buildExprForRounds(strat, opp_strat, N)
                expr_strat.append(expr)
            #strat playing against another strategy
            else:
                if hasOnlyOneAction(strat):
                    action = strat[1][0][0]
                    if hasOnlyOneAction(opp_strat):
                        opp_action = opp_strat[1][0][0]
                        expr = buildExpr(action, opp_action) * N
                    else:
                        current_strat_action_fixed = True
                        expr = buildExprForRoundsOneFixedAction(strat, opp_strat, N, current_strat_action_fixed)
                else:
                    if hasOnlyOneAction(opp_strat):
                        current_strat_action_fixed = False
                        expr = buildExprForRoundsOneFixedAction(opp_strat, strat, N, current_strat_action_fixed)
                    else:
                        if hasOnlyOneDirection(strat) and hasOnlyOneDirection(opp_strat):
                            direction_cur = strat[0][0]
                            direction_opp = opp_strat[0][0]
                            converging_state_cur = 0 if direction_cur == 1 else nb_states - 1
                            converging_state_opp = 0 if direction_opp == 1 else nb_states - 1
                            action_cur = strat[1][converging_state_cur][converging_state_opp]
                            action_opp = opp_strat[1][converging_state_opp][converging_state_cur]
                            expr = buildExpr(action_cur, action_opp) * N
                        else:
                            if hasOnlyOneDirection(strat):
                                direction = strat[0][0]
                                converging_state = 0 if direction == 1 else nb_states - 1
                                if hasOnlyOneActionInState(strat, converging_state):
                                    current_strat_action_fixed = True
                                    expr = buildExprForRoundsOneFixedActionInState(strat, converging_state, opp_strat, N, current_strat_action_fixed)
                                else:
                                    current_strat_direction_fixed = True
                                    strat_actions = strat[1][converging_state]
                                    expr = buildExprForRoundOneFixedDirection(strat_actions, converging_state, opp_strat, N, current_strat_direction_fixed)

                            elif hasOnlyOneDirection(opp_strat):
                                direction = opp_strat[0][0]
                                converging_state = 0 if direction == 1 else nb_states - 1
                                if hasOnlyOneActionInState(opp_strat, converging_state):
                                    current_strat_action_fixed = False
                                    expr = buildExprForRoundsOneFixedActionInState(opp_strat, converging_state, strat, N, current_strat_action_fixed)
                                else:
                                    current_strat_direction_fixed = False
                                    opp_actions = opp_strat[1][converging_state]
                                    expr = buildExprForRoundOneFixedDirection(opp_actions, converging_state,strat, N, current_strat_direction_fixed)
                            else:
                                expr = buildExprForRounds(strat, opp_strat, N)
                expr_strat.append(expr)
        all_expr_groups.append(expr_strat)
        print("Expression for strat ", strat, " (strat ", i+1, "/", len(strategies), ") for ", N, " rounds computed in --- %s seconds --- " % (time.time() - start_time))
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

def buildExprForRounds(cur_strat, opp_strat, N):
    """
    Build an expression, considering both strategies have multiple possible actions and transition directions
    :param cur_strat: strategy being analysed
    :param opp_strat: opponent strategy
    :param N: number of rounds
    :return: expression
    """
    nb_states = len(cur_strat[1])
    states_distrib = 1. / nb_states
    p_states = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)
    p_states_opp = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)
    expr = 0
    for i in range (N):
        p_c = 0.
        p_c_opp = 0.
        for j in range (len(p_states)):
            signal_curr = j
            p_st_curr = p_states[j]
            for k in range (len(p_states_opp)):
                signal_opp = k
                p_st_opp = p_states_opp[k]
                if cur_strat[1][j][signal_opp] == 1:
                    p_c += (p_st_curr * p_st_opp)
                if opp_strat[1][k][signal_curr] == 1:
                    p_c_opp += (p_st_curr * p_st_opp)

        expr += buildExprProba(p_c, p_c_opp)
        p_states = updateProbaStates(cur_strat, p_c_opp, p_states)
        p_states_opp = updateProbaStates(opp_strat, p_c, p_states_opp)
    return expr

def buildExprForRoundsOneFixedActionInState(cur_strat, converging_state, opp_strat, N, current_action_fixed):
    """
    Build an expression, considering that one of the two strategies has only one possible action in a given state
    :param cur_strat: strategy being analysed
    :param converging_state: state in which one of the two given strategies will converge
    :param opp_strat: opponent strategy
    :param N: number of rounds
    :param current_action_fixed: boolean value to True if the strategy being analysed is the one with only one action in the given state
    :return: expression
    """
    action = cur_strat[1][converging_state][0]
    nb_states = len(cur_strat[1])
    opp_direction = opp_strat[0][0] if action == 1 else opp_strat[0][1]
    converging_state_opp = 0 if opp_direction == 1 else nb_states - 1
    opp_actions = opp_strat[1][converging_state_opp]

    if hasOnlyOneDirection(cur_strat):
        cur_direction = cur_strat[0][0]
        converging_state_cur = 0 if cur_direction == 1 else nb_states - 1
        action_opp = opp_actions[converging_state_cur]
        if current_action_fixed:
            expr = buildExpr(action, action_opp) * N
            # print("expr : ", expr)
        else:
            expr = buildExpr(action_opp, action) * N
    else:
        states_distrib = 1. / nb_states
        p_states = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)
        expr = 0
        for i in range (N):
            p_c_opp = 0.
            for j in range (len(p_states)):
                signal_curr = j
                p_st_curr = p_states[j]
                if opp_actions[signal_curr] == 1:
                    p_c_opp += p_st_curr
            if current_action_fixed:
                expr += buildExprProba(action, p_c_opp)     #action == 1 (proba = 1 cooperating) or action == 0 (proba = 0 cooperating)
            else:
                expr += buildExprProba(p_c_opp, action)
            p_states = updateProbaStates(cur_strat, p_c_opp, p_states)

    return expr


def buildExprForRoundsOneFixedAction(cur_strat, opp_strat, N, current_action_fixed):
    """
    Build an expression, considering that one of the two strategies has only one possible action
    :param cur_strat: strategy being analysed
    :param opp_strat: opponent strategy
    :param N: number of rounds
    :param current_direction_fixed: boolean value to True if the strategy being analysed is the one with only one action
    :return: expression
    """
    action = cur_strat[1][0][0]
    nb_states = len(cur_strat[1])
    opp_direction = opp_strat[0][0] if action == 1 else opp_strat[0][1]
    converging_state_opp = 0 if opp_direction == 1 else nb_states - 1
    opp_actions = opp_strat[1][converging_state_opp]

    if hasOnlyOneDirection(cur_strat):
        cur_direction = cur_strat[0][0]
        converging_state_cur = 0 if cur_direction == 1 else nb_states - 1
        action_opp = opp_actions[converging_state_cur]
        if current_action_fixed:
            expr = buildExpr(action, action_opp) * N
        else:
            expr = buildExpr(action_opp, action) * N
    else:
        states_distrib = 1. / nb_states
        p_states = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)
        expr = 0
        for i in range (N):
            p_c_opp = 0.
            for j in range (len(p_states)):
                signal_curr = j
                p_st_curr = p_states[j]
                if opp_actions[signal_curr] == 1:
                    p_c_opp += p_st_curr
            if current_action_fixed:
                expr += buildExprProba(action, p_c_opp)     #action == 1 (proba = 1 cooperating) or action == 0 (proba = 0 cooperating)
            else:
                expr += buildExprProba(p_c_opp, action)
            p_states = updateProbaStates(cur_strat, p_c_opp, p_states)

    return expr


def buildExprForRoundOneFixedDirection(strat_actions, converging_state, variable_dir_strat, N, current_direction_fixed = True):
    """
    Build an expression, considering that one of the two strategies has only one possible transition direction
    :param strat_actions: possible actions for the state in which the strategy with one possible transition direction will converge
    :param converging_state: state in which the strategy with one possible transition direction will converge to
    :param variable_dir_strat: strategy that has multiple possible transition directions
    :param N: number of rounds
    :param current_direction_fixed: boolean value to True if the strategy being analysed is the one with only one transition direction
    :return: expression
    """
    nb_states = len(strat_actions)
    states_distrib = 1. / nb_states
    p_states_variable_dir = [states_distrib for i in range(nb_states)]  # initial Proba (being in each state)
    expr = 0
    for i in range (N):
        p_c_opp = 0.
        p_c = 0.
        for j in range (len(p_states_variable_dir)):
            signal_opp = j
            p_st_opp = p_states_variable_dir[j]
            if strat_actions[signal_opp] == 1:
                p_c += p_st_opp
            if variable_dir_strat[1][j][converging_state] == 1:
                p_c_opp += p_st_opp
        if current_direction_fixed:
            expr += buildExprProba(p_c, p_c_opp)
        else:
            expr += buildExprProba(p_c_opp, p_c)
            p_states_variable_dir = updateProbaStates(variable_dir_strat, p_c, p_states_variable_dir)

    return expr



def updateProbaStates(strat, p_c_opp, p_states):
    """
    Update the probabilities of a strategy being in each state
    :param strat: considered strategy
    :param p_c_opp: probability of cooperation of the opponent
    :param p_states: probability of being in each state before interaction
    :return: updated probabilities of being in each state
    """
    res_p_states = [0. for i in range (len(p_states))]
    max_state = len(p_states) - 1
    transition_c = strat[0][0]  # 1 if Left, 0 if Right
    transition_d = strat[0][1]  # 1 if Left, 0 if Right
    a = alpha       #Transition probability

    for i in range (len(p_states)):
        # new state after the opponent has cooperated (could be the same if it is a outer state (0 or n-1)
        new_state_c = computeNewState(i, max_state, transition_c)
        # new state after the opponent has defected (could be the same if it is a outer state (0 or n-1)
        new_state_d = computeNewState(i, max_state, transition_d)

        proba_new_state_c = a * p_c_opp
        proba_new_state_d = a * (1 - p_c_opp)
        proba_no_change = round(1-a,2)
        res_p_states[new_state_c] += proba_new_state_c * p_states[i]
        res_p_states[new_state_d] += proba_new_state_d * p_states[i]
        res_p_states[i] += proba_no_change * p_states[i]

    return res_p_states



def computeExprMonomorphic(strategies, N):
    """
    Compute expression for pair of strategies (X,X) for N rounds
    :param strategies: strategies
    :param N: number of rounds
    :return: list of expressions
    """
    all_expr_monomorphic = []
    for strat in strategies:
        start_time = time.time()
        # having one possible action
        if hasOnlyOneAction(strat):
            action = strat[1][0][0]
            expr = buildExpr(action, action) * N
        # having one possible direction
        elif hasOnlyOneDirection(strat):
            direction = strat[0][0]
            converging_state = 0 if direction == 1 else nb_states - 1
            opp_signal = converging_state  # Since opp_strat == strat
            action = strat[1][converging_state][opp_signal]
            expr = buildExpr(action, action) * N
        else:
            expr = buildExprForRounds(strat, strat, N)
        all_expr_monomorphic.append(expr)
        print("Expression for strat ", strat, " for group size ", N,
              " computed in --- %s seconds --- " % (time.time() - start_time))
    return all_expr_monomorphic


def storePickle(filename, all_expr_groups):
    """
    Store element in .pickle file
    :param filename: file in which we store an object
    :param all_expr_groups: matrix containing all the expressions for every pair of strategies
    """
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(all_expr_groups, f)


if __name__ == '__main__':
    N = 10
    for nb_states in [1,2,3]:
        nb_signals = nb_states
        strats_filename = "Strategies/" + str(nb_states) + "_st_strats"
        strats = loadStrategies(strats_filename)
        start_time = time.time()

        exprs_monomorphic = computeExprMonomorphic(strats, N)

        filename = "ExprAnalyticalMonomorphic/" + str(nb_states) + "st/expressions_" + str(
            nb_states) + "st_rounds_" + str(N)
        storePickle(filename, exprs_monomorphic)

    #nb_states = 2
    #nb_signals = nb_states
    #strats_filename = "Strategies/" + str(nb_states) + "_st_strats"
    #strats = loadStrategies(strats_filename)
    #start_time = time.time()
    #all_exprs = computeExprRounds(strats, N, nb_states)

    #print("all expressions for ", N,
    #      " rounds computed in --- %s seconds --- " % (time.time() - start_time))
    #filename = "ExprAnalytical/" + str(nb_states) + "states/expressions_"  + str(N) +  "_rounds_alpha08"
    #storePickle(filename, all_exprs)
