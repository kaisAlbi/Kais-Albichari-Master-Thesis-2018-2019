from TreeStrategy import Tree
import numpy as np
from itertools import product
import time
import json



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


def generateTrees(strategies, nb_states, rounds, trans_states_proba):
    """
    Generates all decision trees to compute the probability of a strategy A cooperating against another strategy B (or itself)
    :param strategies: array of binary strategies
    :param nb_states: number of available internal states values
    :param rounds: number of rounds, i.e. depth of decision trees
    :param trans_states_proba: array of probabilities of transitions between states
    :return: matrix where cell [i, j] is the probability of strategy i cooperating with strategy j
    """
    n = len(strategies)
    proba_c_couples = np.zeros((n, n))

    #start_whole_time = time.time()
    for i in range(n):
        #start_time = time.time()
        strat_1 = strategies[i]
        if hasOnlyOneAction(strat_1):
            proba_c_couples[i,i] = strat_1[2]       #1 if cooperate, 0 if defect
        elif hasOnlyOneDirection(strat_1):
            if strat_1[0] == 1:     #Only direction is left
                proba_c_couples[i, i] = strat_1[2]
            else:                   #Only direction is right
                proba_c_couples[i, i] = strat_1[nb_states+1]
        else:
            proba_c_couples[i, i] = computeSingleProba(strat_1, nb_states, rounds, trans_states_proba)
        for j in range(i + 1, n):
            strat_2 = strategies[j]
            proba_c1, proba_c2 = computeProbaCCouples(strat_1, strat_2, nb_states, rounds, trans_states_proba)

            proba_c_couples[i, j] = proba_c1
            proba_c_couples[j, i] = proba_c2
        #print("Generate every Tree for strat ", displayStrat(strat_1), " took --- %s seconds---" % (time.time() - start_time))
    #print("Generate every Tree took --- %s seconds---" % (time.time() - start_whole_time))
    return proba_c_couples


def computeSingleProba(strat, nb_states, rounds, trans_states_proba):
    """
    Build a decision tree for a strategy with itself
    :param strat: current strategy
    :param nb_states: number of available internal states values
    :param rounds: depth of trees
    :param trans_states_proba: array of probabilities of transitions between states
    :return: probability of a strategy cooperating with itself
    """
    init_weight = 1
    proba_c = 0
    for state in range(nb_states):
        # Computes the probability of a strategy cooperating with itself, for the same initial state value
        root = Tree(state, strat, init_weight, nb_states - 1, trans_states_proba)
        proba_c_current = root.getProbaC()
        root.addChildren(proba_c_current)
        current_level = root.getChildren()
        for i in range(rounds - 1):
            proba_c_current = sum([node.getProbaC() for node in current_level])

            next_level = []
            for child in current_level:
                child.addChildren(proba_c_current)
                for new_child in child.getChildren():
                    next_level.append(new_child)
            current_level = list(next_level)
        proba_c += np.clip(sum([node.getProbaC() for node in next_level]), 0., 1.)
        del root

        for state_2 in range (state+1, nb_states):
            # Computes the probability of a strategy cooperating with itself, for different initial state value
            root_1 = Tree(state, strat, init_weight, nb_states - 1, trans_states_proba)
            root_2 = Tree(state_2, strat, init_weight, nb_states - 1, trans_states_proba)
            proba_c_current = root_1.getProbaC()
            proba_c2 = root_2.getProbaC()
            root_1.addChildren(proba_c2)
            root_2.addChildren(proba_c_current)
            current_level_1 = root_1.getChildren()
            current_level_2 = root_2.getChildren()
            del root_2
            del root_1
            for i in range(rounds - 1):
                proba_c1 = sum([node.getProbaC() for node in current_level_1])
                proba_c2 = sum([node.getProbaC() for node in current_level_2])

                next_level_1 = []
                for child in current_level_1:
                    child.addChildren(proba_c2)
                    for new_child in child.getChildren():
                        next_level_1.append(new_child)

                next_level_2 = []
                for child in current_level_2:
                    child.addChildren(proba_c1)
                    for new_child in child.getChildren():
                        next_level_2.append(new_child)

                for elem in current_level_1:
                    del elem
                for elem in current_level_2:
                    del elem
                current_level_1 = list(next_level_1)
                current_level_2 = list(next_level_2)
            proba_c += sum([node.getProbaC() for node in next_level_1])

            proba_c += sum([node.getProbaC() for node in next_level_2])

    return proba_c / np.power(nb_states, 2)


def computeProbaCCouples(strat_1, strat_2, nb_states, rounds, trans_states_proba):
    """
    Computes the probabilities of cooperation of two strategies with each other
    :param strat_1: strategy for the first decision tree
    :param strat_2: strategy for the second decision tree
    :param nb_states: number of available internal states values
    :param rounds: depth of trees
    :param trans_states_proba: array of probabilities of transitions between states
    :return: probability of strat_1 cooperating with strat_2 and probability of strat_2 cooperating with strat_1
    """
    init_weight = 1
    proba_c1 = 0
    proba_c2 = 0

    if hasOnlyOneAction(strat_1) and hasOnlyOneAction(strat_2):
        proba_c1 = strat_1[2]
        proba_c2 = strat_2[2]

    elif hasOnlyOneDirection(strat_1) and hasOnlyOneAction(strat_2):
        if strat_1[0] == 1:  # Only direction is left
            proba_c1 = strat_1[2]
        else:  # Only direction is right
            proba_c1 = strat_1[nb_states + 1]
        proba_c2 = strat_2[2]

    elif hasOnlyOneAction(strat_1) and hasOnlyOneDirection(strat_2):
        proba_c1 = strat_1[2]
        if strat_2[0] == 1:  # Only direction is left
            proba_c2 = strat_2[2]
        else:  # Only direction is right
            proba_c2 = strat_2[nb_states + 1]

    elif hasOnlyOneDirection(strat_1) and hasOnlyOneDirection(strat_2):
        if strat_1[0] == 1:  # Only direction is left
            proba_c1 = strat_1[2]
        else:  # Only direction is right
            proba_c1 = strat_1[nb_states + 1]

        if strat_2[0] == 1:  # Only direction is left
            proba_c2 = strat_2[2]
        else:  # Only direction is right
            proba_c2 = strat_2[nb_states + 1]

    else:       # Have to build two decision tree without having any fixed probability of cooperation
        for state_1 in range(nb_states):
            for state_2 in range(nb_states):
                root_1 = Tree(state_1, strat_1, init_weight, nb_states - 1, trans_states_proba)
                root_2 = Tree(state_2, strat_2, init_weight, nb_states - 1, trans_states_proba)
                cur_prob_c1, cur_prob_c2 = buildTrees(root_1, root_2, rounds)
                proba_c1 += cur_prob_c1
                proba_c2 += cur_prob_c2
        proba_c1 /= np.power(nb_states, 2)
        proba_c2 /= np.power(nb_states, 2)




    return proba_c1, proba_c2


def buildTrees(root_1, root_2, rounds):
    """
    Build two decision trees in parallel for two different strategies playing against each other
    :param root_1: root of the decision tree representing the first strategy
    :param root_2: root of the decision tree representing the second strategy
    :param rounds: depth of the trees
    :return: probability of strat 1 cooperating with strat 2 and probability of strat 2 cooperating with strat 1
    """
    proba_c1 = root_1.getProbaC()
    proba_c2 = root_2.getProbaC()

    root_1.addChildren(proba_c2)
    root_2.addChildren(proba_c1)
    current_level_1 = root_1.getChildren()
    current_level_2 = root_2.getChildren()

    del root_1
    del root_2

    for i in range(rounds - 1):

        proba_c1 = sum([node.getProbaC() for node in current_level_1])
        proba_c2 = sum([node.getProbaC() for node in current_level_2])

        next_level_1 = []
        for child in current_level_1:
            child.addChildren(proba_c2)
            for new_child in child.getChildren():
                next_level_1.append(new_child)

        next_level_2 = []
        for child in current_level_2:
            child.addChildren(proba_c1)
            for new_child in child.getChildren():
                next_level_2.append(new_child)

        for elem in current_level_1:
            del elem
        for elem in current_level_2:
            del elem
        current_level_1 = list(next_level_1)
        current_level_2 = list(next_level_2)
    proba_c1 = sum([node.getProbaC() for node in next_level_1])
    proba_c2 = sum([node.getProbaC() for node in next_level_2])
    for elem in next_level_1:
        del elem
    for elem in next_level_2:
        del elem

    return np.clip(proba_c1, 0., 1.), np.clip(proba_c2, 0., 1.)

def getPayoff(R, S, T, P, proba_c1, proba_c2):
    """
    :param proba_c1: probability of first agent cooperating
    :param proba_c2: probability of second agent cooperating
    :return: payoff value an agent obtains after an interaction with another agent
    """
    payoff = 0
    payoff += np.float64(R * proba_c1 * proba_c2)
    payoff += np.float64(S * proba_c1 * (1 - proba_c2))
    payoff += np.float64(T * (1 - proba_c1) * proba_c2)
    payoff += np.float64(P * (1 - proba_c1) * (1 - proba_c2))
    return payoff


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



def displayStrat(strat):
    """
    Transforms a binary strategy array into a human readable strategy array
    :param strat: strategy array
    :return: human readable strategy array
    """
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


def computeAvgPayoffPairs(R, S, T, P, n, proba_couples):
    """
    Computes a matrix of analytical payoffs for all pairs of strategies
    :param R: mutual cooperation payoff
    :param S:  payoff when the considered strategy cooperated while the other defected
    :param T: temptation to defect - payoff when the considered strategy defected while the other cooperated
    :param P: mutual defection payoff
    :param n: number of strategies
    :param proba_couples: matrix of probability of cooperation of strategies against other strategies and themselves
    :return: matrix of analytical payoffs
    """
    payoff_pairs = np.zeros((n,n))
    for i in range (n):
        proba_i_i = proba_couples[i,i]
        payoff_pairs[i,i] = getPayoff(R, S, T, P, proba_i_i, proba_i_i)

        for j in range (i+1, n):
            proba_i_j = proba_couples[i, j]
            proba_j_i = proba_couples[j, i]
            payoff_pairs[i, j] = getPayoff(R, S, T, P, proba_i_j, proba_j_i)
            payoff_pairs[j, i] = getPayoff(R, S, T, P, proba_j_i, proba_i_j)
    return payoff_pairs


def storePayoffPairs(filename, payoff_pairs):
    f = open(filename, "w")
    json.dump(payoff_pairs.tolist(), f)
    f.close()


def storeProbaCouple(filename, proba_couples):
    f = open(filename, "w")
    json.dump(proba_couples.tolist(), f)
    f.close()


def computeProbaAndPayoffsCouples(strats, rounds, rho_list, R, S, T, P):
    for i in range (len(rho_list)):

        for j in range (len(rho_list)):
            index = i*len(rho_list) + j
            trans_proba = [rho_list[i], rho_list[j]]
            proba_couples = generateTrees(strats, nb_states, rounds, trans_proba)
            proba_filename = "ProbaCouples/Simplified/proba_couples_matrix_" + str(nb_states) + "st.txt"
            storeProbaCouple(proba_filename, proba_couples)
            payoff_pairs = computeAvgPayoffPairs(R, S, T, P, len(strategies), proba_couples)
            payoffs_pairs_filename = "PayoffPairs/Simplified/" + game + "/payoffs_pairs_" + str(nb_states) + "st.txt"
            storePayoffPairs(payoffs_pairs_filename, payoff_pairs)
            #proba_filename = "ProbaCouples/" + game + "/DiffTrans/proba_couples_matrix_" + str(nb_states) + "st_"+str(index)+".txt"
            #storeProbaCouple(proba_filename, proba_couples)
            #payoff_pairs = computeAvgPayoffPairs(R, S, T, P, len(strategies), proba_couples)
            #payoffs_pairs_filename = "PayoffPairs/" + game + "/DiffTrans/payoffs_pairs_" + str(nb_states) + "st_"+str(index)+".txt"
            #storePayoffPairs(payoffs_pairs_filename, payoff_pairs)
            #displayProbaCouples(strats, proba_couples)
            #displayAvgPayoffsCouples(strats, payoff_pairs)

def displayProbaCouples(strats, proba_couples):
    for i in range (len(proba_couples)):
        strat_A = strats[i]
        for j in range (len(proba_couples)):
            strat_B = strats[j]
            print("strategy " + "".join(displayStrat(strat_A)) + " has a probability of cooperating with strategy " + "".join(displayStrat(strat_B)) + " of %s" %(proba_couples[i, j]))

def displayAvgPayoffsCouples(strats, payoff_pairs):
    for i in range (len(payoff_pairs)):
        strat_A = strats[i]
        for j in range (len(payoff_pairs)):
            strat_B = strats[j]
            print("strategy " + "".join(displayStrat(strat_A)) + " has an average payoff against strategy "+ "".join(displayStrat(strat_B)) + " of %s" %(payoff_pairs[i, j]))

if __name__ == '__main__':


    #strategies = [[1,0,1,0],[1,0,0,1]]#,[1,0,0,1],[1,0,0,0]]
    start_time = time.time()
    game = "SH"
    R, P = 1, 0
    T = 0.5
    S = -0.5
    Z = 150  # Population size
    nb_states = 2
    rounds = 10
    strategies = createStrategies(nb_states)
    #rho = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    rho = [0.8]
    #print("For " + str(rounds) + " rounds:")
    computeProbaAndPayoffsCouples(strategies, rounds, rho, R, S, T, P)




    print("the whole program executed in --- %s seconds --- "%(time.time() - start_time))


