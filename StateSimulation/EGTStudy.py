import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import networkx as nx
from itertools import product
import json
import time
from Player import Player

class EGTModel:
    def __init__(self, R, S, T, P, pop_size, beta, nb_states, state_change_thresh, rounds):
        self.Z = pop_size
        self.beta = beta
        self.nb_states = nb_states
        self.alpha = state_change_thresh
        self.R = R
        self.S = S
        self.T = T
        self.P = P
        self.rounds = rounds

        self.strategies = self.createStrategies()
        self.transition_proba, self.fix_probs, self.stationary = self.stationaryDistrib()

    def getNbStates(self):
        return self.nb_states

    def getStrategies(self):
        return self.strategies

    def getBeta(self):
        return self.beta

    def getPopSize(self):
        return self.Z

    def getFixProbs(self):
        return self.fix_probs

    def getTransitionMatrix(self):
        return self.transition_proba

    def getStationaryDistrib(self):
        return self.stationary

    def createStrategies(self):
        """
        Generate all possible combination of strategies, depending on the number of states
        :return: list of all strategies in the form [[T_c,T_d],[action state_0, ..., action state_maxState]]
        """
        action_choice = list(list(item) for item in product("CD", repeat=self.getNbStates()))
        state_change = list(list(item) for item in product("LR", repeat=2))
        strats = []
        for action in action_choice:
            for state_c in state_change:
                strats.append([state_c, action])
        return strats

    def player_factory(self, k, strat_A, strat_B):
        """
        Generates a numpy array of players (with equi-distributed states values) of size Z
        that define the distribution of the population.
        :param k: number of invading players
        :param strat_A: invader strategy
        :param strat_B: resident strategy
        :return: list population of agents
        """
        freq = [1. / self.nb_states for i in range(self.nb_states)]
        population = []
        for i in range(self.getPopSize()):
            population.append(
                Player(i, np.random.choice(self.nb_states, 1, p=freq)[0], strat_B,
                                     self.nb_states - 1))
        for i in range (k):
            invader_added = False
            while not invader_added:
                index = np.random.randint(self.Z)
                if population[index].getStrategy() == strat_B:
                    population[np.random.randint(self.Z)].imitateStrat(strat_A)
                    invader_added = True

        return population


    def getPayoff(self, first, second):
        """
        :param first: first agent
        :param second: second agent
        :return: payoff value an agent obtains after an interaction with another agent
        """
        if first == "C":
            if second == "C":
                return self.R
            else:
                return self.S
        else:
            if second == "C":
                return self.T
            else:
                return self.P


    def fermiDistrib(self, first_fitness, second_fitness, positive):
        """
        :param first_payoff: payoff obtained by the first agent after the interaction
        :param second_payoff: payoff obtained by the second agent after the interaction
        :param positive: boolean value used to calculate probability increase and decrease (T +-)
        :return: probability that the first agent imitates the second ond
        """
        if positive:
            return 1. / (1. + np.exp(-self.getBeta() * (first_fitness - second_fitness)))
            #return 1./(1. + np.exp(self.getBeta()*(second_fitness-first_fitness)))     #Francisco's fermi distribution
        else:
            return 1. / (1. + np.exp(self.getBeta() * (first_fitness - second_fitness)))
            #return 1./(1. + np.exp(-self.getBeta()*(second_fitness-first_fitness)))

    def getBothTypesPlayers(self, population, inv_strat):
        """
        :return: randomly chosen invader and resident (A and B) indexes
        """
        agents_strats_index = {}

        for index, agent in enumerate(population):
            if agent.getStrategy() == inv_strat:
                agents_strats_index.setdefault("A", []).append(index)
            else:
                agents_strats_index.setdefault("B", []).append(index)

        invader_index = np.random.choice(agents_strats_index["A"])
        resident_index = np.random.choice(agents_strats_index["B"])

        return invader_index, resident_index

    def playAgainstEveryone(self, first_player, second_player, first_opponents, second_opponents):
        """
        Simulate one round consisting of two player interacting with everyone else.
        :return: fitness of both players
        """
        fitness = np.array([0., 0.], dtype=np.float64)
        for i in range(len(first_opponents)):
            fitness_first = 0
            fitness_second = 0
            for j in range(self.rounds):

                first_action = first_player.play()
                other_f_action = first_opponents[i].play()
                if np.random.rand() < self.alpha:
                    first_player.changeState(other_f_action)
                if np.random.rand() < self.alpha:
                    first_opponents[i].changeState(first_action)
                fitness_first += self.getPayoff(first_action, other_f_action)

                second_action = second_player.play()
                other_s_action = second_opponents[i].play()
                if np.random.rand() < self.alpha:
                    second_player.changeState(other_s_action)
                if np.random.rand() < self.alpha:
                    second_opponents[i].changeState(first_action)

                fitness_second += self.getPayoff(second_action, other_s_action)

            f_state = np.random.choice(self.nb_states)
            s_state = np.random.choice(self.nb_states)

            first_player.setState(f_state)
            second_player.setState(s_state)

            fitness[0] += fitness_first / self.rounds
            fitness[1] += fitness_second / self.rounds

        return np.divide(fitness, len(first_opponents))

    def game(self, gen_players, population):
        """
        :param gen_players: list of the indexes of the two players that will play the game
        :param population: list of all agents
        :return: fitness values for the two players after having played the game
        """
        first_player = population[gen_players[0]]  # invader
        second_player = population[gen_players[1]]  # resident
        first_opponents = list(population)
        first_opponents.pop(gen_players[0])
        second_opponents = list(population)
        second_opponents.pop(gen_players[1])

        fitness = self.playAgainstEveryone(first_player, second_player, first_opponents, second_opponents)
        return fitness

    def changingStateGame(self, gen_players, population):
        """
        Plays the game to simulate the states changes, without the fitness.
        """
        first_player = population[gen_players[0]]  # invader
        second_player = population[gen_players[1]]  # resident
        first_opponents = list(population)
        first_opponents.pop(gen_players[0])
        second_opponents = list(population)
        second_opponents.pop(gen_players[1])

        self.playAgainstEveryoneWithoutFitness(first_player, second_player, first_opponents, second_opponents)


    def playAgainstEveryoneWithoutFitness(self, first_player, second_player, first_opponents, second_opponents):
        """
        Simulate one round consisting of two player interacting with everyone else without computing their fitness values.
        Useful to keep the population's internal states consistent.
        """
        for i in range(len(first_opponents)):
            other_f_action = first_opponents[i].play()
            if np.random.rand() < self.alpha:
                first_player.changeState(other_f_action)
            other_s_action = second_opponents[i].play()
            if np.random.rand() < self.alpha:
                second_player.changeState(other_s_action)


    def getFitnessSimpleCases(self, n_A, action_inv, action_res):
        """
        Compute analytically the average payoff of an invader strategy and a resident strategy when the simulation is not needed
        :param n_A: number of invaders
        :param action_inv: action chosen by the invaders
        :param action_res: action chosen by the residents
        :return: the average payoff of both the invaders and the residents
        """
        p_inv_inv = self.getPayoff(action_inv, action_inv)
        p_inv_res = self.getPayoff(action_inv, action_res)
        p_res_res = self.getPayoff(action_res, action_res)
        p_res_inv = self.getPayoff(action_res, action_inv)
        result_inv = ((n_A) * p_inv_inv + (self.Z - n_A) * p_inv_res) / float(self.Z - 1)
        result_res = (n_A * p_res_inv + (self.Z - n_A) * p_res_res) / float(self.Z - 1)
        return result_inv, result_res

    def getFitness(self, n_A, gen_players, population):
        """
        :param n_A: number of invaders
        :param gen_players: list of indexes of the two players that will play the game
        :param population:  list of all the agents
        :return: fitness values for the two players after having played the game
        """
        inv_strat = population[gen_players[0]].getStrategy()
        res_strat = population[gen_players[1]].getStrategy()

        if self.hasOnlyOneAction(inv_strat):
            action_inv = inv_strat[1][0]
            if self.hasOnlyOneAction(res_strat):
                action_res = res_strat[1][0]
                return self.getFitnessSimpleCases(n_A, action_inv, action_res)
            elif self.hasOnlyOneDirection(res_strat):
                action_res = res_strat[1][0]
                if res_strat[0][0] == "R":
                    action_res = res_strat[1][-1]
                return self.getFitnessSimpleCases(n_A, action_inv, action_res)
            else:
                return self.game(gen_players, population)
        elif self.hasOnlyOneDirection(inv_strat):
            action_inv = inv_strat[1][0]
            if inv_strat[0][0] == "R":
                action_inv = inv_strat[1][-1]

            if self.hasOnlyOneAction(res_strat):
                action_res = res_strat[1][0]
                return self.getFitnessSimpleCases(n_A, action_inv, action_res)

            elif self.hasOnlyOneDirection(res_strat):
                action_res = res_strat[1][0]
                if res_strat[0][0] == "R":
                    action_res = res_strat[1][-1]
                return self.getFitnessSimpleCases(n_A, action_inv, action_res)
            else:
                return self.game(gen_players, population)
        else:
            return self.game(gen_players, population)


    def probIncDec(self, n_A, gen_players, population):
        """
        :param n_A: number of invaders
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: probability to change the number of k invaders (by +- one at each time step)
        """
        inv_strat = population[gen_players[0]].getStrategy()
        res_strat = population[gen_players[1]].getStrategy()
        fitness_val = self.getFitness(n_A, gen_players, population)
        f_A = np.exp(self.beta * fitness_val[0])
        f_B = np.exp(self.beta * fitness_val[1])

        a_birth = n_A * f_A / ((n_A * f_A) + (self.Z - n_A) * f_B)
        a_death = n_A / self.Z
        p_no_change = a_birth * a_death + (1 - a_birth) * (1 - a_death)
        nb_step_required = np.random.geometric(1 - p_no_change)

        # Check whether the simulation of the steps (without the computation of fitness) are needed or not
        if not (self.hasOnlyOneAction(inv_strat) and self.hasOnlyOneAction(res_strat)) \
            and not (self.hasOnlyOneAction(inv_strat) and self.hasOnlyOneDirection(res_strat)) \
            and not (self.hasOnlyOneDirection(inv_strat) and self.hasOnlyOneAction(res_strat)) \
            and not (self.hasOnlyOneDirection(inv_strat) and self.hasOnlyOneDirection(res_strat)):

            population_indexes = np.arange(0, self.getPopSize(), dtype=np.int64)
            for i in range (nb_step_required):
                changing_states_players = np.random.choice(population_indexes, replace=True, size=2)
                self.changingStateGame(changing_states_players, population)

        inc = f_A / (f_A + f_B)
        dec = f_B / (f_A + f_B)

        return [inc, dec]



    def hasOnlyOneDirection(self, strat):
        """
        :param strat: strategy
        :return: True if the strategy given as argument has only one possible transition direction, False otherwise
        """
        directions = strat[0]
        if "L" not in directions or "R" not in directions:
            return True
        return False

    def hasOnlyOneAction(self, strat):
        """
        :param strat: strategy
        :return: True if the strategy given as argument has only one possible action choice, False otherwise
        """
        actions = strat[1]
        if "C" not in actions or "D" not in actions:
            return True
        return False

    def fixationProba(self, inv_strat, res_strat):
        """
        :param inv_strat: invaders' strategy
        :param res_strat: residents' strategy
        :return: fixation probability of the invader in a population of residents
        """

        result = 0.
        for i in range(0, self.Z):
            mul = 1.
            population = self.player_factory(1, inv_strat, res_strat)
            for j in range(1, i + 1):

                gen_players = self.getBothTypesPlayers(population, inv_strat)
                inc, dec = self.probIncDec(j, gen_players, population)
                lambda_j = dec / float(inc)
                mul *= lambda_j
                population[gen_players[1]].imitateStrat(inv_strat)

            result += mul
            if i == self.Z - 1:
                print("MUL : ",mul, " ----- Strategy ", res_strat, " invaded by ",inv_strat )
        return np.clip(1. / result, 0., 1.)

    def transitionMatrix(self):
        """
        Compute the fixation probability for each pair invader-resident of strategies and build the fixation probabilities
        matrix and the transition matrix
        :return: transition matrix and fixation probabilities matrix
        """
        strats = self.getStrategies()
        n = len(strats)
        norm_fact = 1 / float((n - 1))
        fix_probs = np.zeros((n, n))
        transitions = np.zeros((n, n))
        for i in range(n):
            start_time = time.time()
            res_strat = strats[i]
            transitions[i, i] = 1
            for j in range(n):
                inv_strat = strats[j]
                if i != j:
                    f_proba = self.fixationProba(inv_strat, res_strat)
                    fix_probs[i, j] = f_proba
                    trans_value = f_proba * norm_fact
                    transitions[i, j] = trans_value
                    transitions[i, i] -= trans_value
            print("transitions values calculations for resident strat ", strats[i], " took --- %s seconds---" % (time.time() - start_time))
        return [transitions, fix_probs]

    def stationaryDistrib(self):
        """
        Calculate the transition matrix, and based on that matrix, the stationary distribution of each strategy
        :return: transition matrix, fixation probabilities matrix, stationary distribution
        """
        start_time = time.time()
        t, f = self.transitionMatrix()
        print("The transition and fixation proba matrices calculation took --- %s seconds ---" % (time.time() - start_time))
        n = len(self.getStrategies())
        val, vect = np.linalg.eig(t.transpose())
        j_stationary = np.argmin(abs(val - 1.0))  # look for the element closest to 1 in the list of eigenvalues
        p_stationary = abs(vect[:, j_stationary].real)  # the, is essential to access the matrix by column
        p_stationary /= p_stationary.sum()  # normalize
        print("The stationary distribution calculation took --- %s seconds ---" % (
                    time.time() - start_time))
        return t, f, p_stationary


def displayStrats(strats):
    """
    Displays the  strategies
    """
    for i in range(len(strats)):
        if i == 0:
            print("1st ", end=" ")
        elif i == 1:
            print("2nd ", end=" ")
        elif i == 2:
            print("3rd ", end=" ")
        else:
            print(i + 1, "th ", end=" ")
        print(strats[i], end=" ")
        if i != len(strats) - 1:
            print(", ", end=" ")
        else:
            print()


def displayTransMat(transition_matrix):
    for i in range(len(transition_matrix)):
        print("[", end="")
        for j in range(len(transition_matrix[i])):
            if j != len(transition_matrix[i]) - 1:
                print(round(transition_matrix[i][j], 7), end=", ")
            else:
                print(round(transition_matrix[i][j], 7), end="")
        print("]")


def save(filename, egt_model):
    """
    Saves the transition matrix, the fixation probabilities matrix, the strategies and the stationary distribution of a given model in a given file
    :param filename: file in which we store the information
    :param egt_model: model whose information is saved
    """
    f = open(filename, "w")
    all = [egt_model.getTransitionMatrix().tolist(), egt_model.getFixProbs().tolist(), egt_model.getStrategies(), egt_model.getStationaryDistrib().tolist()]
    json.dump(all, f)
    f.close()

def load(filename):
    """
    Loads the transition matrix, the fixation probabilities matrix, the strategies and the stationary distribution of a model from a given file
    :param filename: file in which the information are stored
    """
    f = open(filename, "r")
    transition_matrix, fix_probs, strats, stationary_dist = json.load(f)
    return transition_matrix, fix_probs, strats, stationary_dist


if __name__ == '__main__':
    R, P = 1, 0
    T = 0.5
    S = -T
    Z = 150 #Population size
    beta = 0.05
    nb_states = 2
    state_change_thresh = 0.8
    rounds = 10
    egt_2states = EGTModel(R, S, T, P, Z, beta, nb_states, state_change_thresh, rounds)
    stationary = egt_2states.getStationaryDistrib()
    strats = egt_2states.getStrategies()
    for i in range(len(strats)):
        print(strats[i], " : ", round(stationary[i], 8))
    #save("egt_2states.txt", egt_2states)
