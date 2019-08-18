
from itertools import product

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
    :param strategies: set of all possibles strategies
    :return: reduced set of strategies without the symmetry
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