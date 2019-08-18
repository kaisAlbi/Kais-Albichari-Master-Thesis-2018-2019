
from itertools import product
import time
import pickle
import numpy as np


def createStrategies(nb_states, nb_sig):
    action_sig = [item for item in product(range(2), repeat=nb_sig)]
    actions = [item for item in product(action_sig, repeat=nb_states)]
    state_change = [item for item in product(range(2), repeat=2)]

    strats = [item for item in product(state_change, actions)]
    return strats


def displayableStrat(strat):
    """
    Transforms a binary strategy array into a human readable strategy array
    :param strat: strategy array
    :return: human readable strategy string
    """
    displayable_strat = ""
    trans = strat[0]
    actions = strat[1]
    for choice in trans:
        if choice == 1:
            displayable_strat += "L"
        else:
            displayable_strat += "R"
    for choices in actions:
        displayable_strat += "-"
        for choice in choices:
            if choice == 1:
                displayable_strat += "C"
            else:
                displayable_strat += "D"

    return displayable_strat

def hasOnlyOneDirection(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible transition direction, False otherwise
    """
    directions = strat[0]
    if 1 not in directions or 0 not in directions:
        return True
    return False

def hasOnlyOneActionInState(strat, state):
    """
    :param strat: strategy
    :param state: internal state value
    :return: True if the strategy given as argument has only one possible action choice when being in the state given
    as argument, False otherwise
    """
    actions_in_state = strat[1][state]
    if 1 not in actions_in_state or 0 not in actions_in_state:
        return True
    return False

def hasOnlyOneAction(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible action choice, False otherwise
    """
    actions = strat[1]
    flat_actions = [item for sublist in actions for item in sublist]
    if 1 not in flat_actions or 0 not in flat_actions:
        return True
    return False


def storeStrategies(strats, filename):
    """
    Store the strategies in a .pickle file
    :param strats: strategies
    :param filename: name of the created file
    """
    store_filename = filename + ".pickle"
    with open(store_filename, "wb") as f:
        pickle.dump(strats, f)

def loadStrategies(filename):
    """
    load the strategies from a .pickle file
    :param filename: file in which the strategies are stored
    :return: strategies
    """
    f = open(filename + ".pickle", "rb")
    strats = np.asarray(pickle.load(f, encoding='latin1'))
    f.close()
    return strats

if __name__ == "__main__":
    nb_states = 3
    nb_sig = nb_states

    strats_filename = "Strategies/" + str(nb_states) + "_st_strats"

    start_time = time.time()
    strats = createStrategies(nb_states, nb_sig)

    print("time needed to compute strategies  --- %s seconds --- " % (time.time() - start_time))

    storeStrategies(strats, strats_filename)

