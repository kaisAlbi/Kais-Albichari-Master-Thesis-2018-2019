import numpy as np
from Player import Player
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RunStationary:
    def __init__(self, pop_size, nb_states, strategies, stationary, nb_runs, nb_sim):

        self.state_chg_thresh = 0.8
        self.Z = pop_size
        self.nb_states = nb_states
        self.strategies = strategies
        self.stationary = stationary
        self.nb_runs = nb_runs
        self.nb_sim = nb_sim
        self.states_array = [i for i in range(nb_states)]
        self.states_per_run = np.zeros((len(self.strategies),self.nb_states))
        #self.states_per_run = np.zeros((self.nb_states,self.nb_runs))

    def getStatesPerRun(self):
        return self.states_per_run

    def player_factory(self):
        freq_state = [1. / len(self.states_array) for i in range(len(self.states_array))]

        population = []

        pop_strats = np.random.choice(len(self.strategies), self.Z, p=self.stationary)
        for i in range(self.Z):
            cur_strat = self.strategies[pop_strats[i]]
            if hasOnlyOneDirection(cur_strat):
                if cur_strat[0] == 1:    #Left
                    population.append(Player(i,0, cur_strat, self.nb_states - 1))
                else:
                    population.append(Player(i, self.nb_states - 1, cur_strat, self.nb_states - 1))
            else:

                population.append(
                    Player(i, np.random.choice(self.states_array, 1, p=freq_state)[0], cur_strat,
                                        self.nb_states - 1))
        return population, pop_strats


    def addStateDistribution(self, population, run):
        for agent in population:
            current_state = agent.getState()
            self.states_per_run[current_state, run] += 1

    def getStatesDistribForRun(self, population):
        current_states_distrib = np.zeros((len(self.strategies),self.nb_states))
        for agent in population:
            current_strat = agent.getStrategy()
            current_state = agent.getState()
            strat_index = self.strategies.index(current_strat)
            current_states_distrib[strat_index, current_state] += 1
        return current_states_distrib


    def runStationary(self):
        total_states_distrib = np.zeros((self.nb_runs, len(self.strategies),self.nb_states))
        population, pop_strats = self.player_factory()
        for i in range (self.nb_runs):
            population_indexes = np.arange(0, self.Z, dtype=np.int64)
            gen_players = np.random.choice(population_indexes, replace=False, size=2)
            first_player = population[gen_players[0]]
            second_player = population[gen_players[1]]
            self.play(first_player, second_player)

            total_states_distrib[i] = self.getStatesDistribForRun(population)
        return np.mean(total_states_distrib, axis=0)


    def runSimulation(self):
        total_states_distrib_sim = np.zeros((self.nb_sim, len(self.strategies),self.nb_states))
        for i in range (self.nb_sim):
            total_states_distrib_sim[i] = self.runStationary()
        return np.mean(total_states_distrib_sim, axis=0)

    def play(self, first_player, second_player):
        first_action = first_player.play()
        second_action = second_player.play()
        if np.random.rand() < self.state_chg_thresh:        #Change first player's internal state
            first_player.changeState(second_action)
        if np.random.rand() < self.state_chg_thresh:        #Change first player's internal state
            first_player.changeState(first_action)

    def getStatesForStrats(self, population, pop_strats):
        states_for_strats = np.zeros((len(self.strategies),self.nb_states))
        for i in range (len(population)):
            strat_index = pop_strats[i]
            state = population[i].getState()
            states_for_strats[strat_index, state] += 1
        return states_for_strats

def hasOnlyOneDirection(strat):
    """
    :param strat: strategy
    :return: True if the strategy given as argument has only one possible transition direction, False otherwise
    """
    directions = strat[:2]
    if 1 not in directions or 0 not in directions:
        return True
    return False

def createStrategies(nb_states):
    """
    Generate all possible combination of strategies, depending on the number of signals
    :return: list of all strategies
    """
    action_choice = list(list(item) for item in product("CD", repeat=nb_states))
    state_change = list(list(item) for item in product("LR", repeat=2))
    strats = []
    for action in action_choice:
        action_c_tr = []
        for i in range(len(action)):

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


def getStationaryFromFile(filename):
    f = open(filename, "r")
    stationary = []
    for line in f.readlines():
        index = line.index(": ")
        stationary_i = line[index + 1:]
        stationary.append(float(stationary_i))

    r = 1 - sum(stationary)  # Handle loss of precision
    print(r)
    print(sum(stationary))
    stationary[0] += r
    print(sum(stationary))
    f.close()
    return stationary

def makeXTicks(strats):
    """
    Transforms the binary strategies into 'x_ticks' to plot them on a graph
    :param strats: array of binary strategies
    :return: array of transformed strategies
    """
    x_ticks = []
    for strat in strats:
        x_tick = []
        for i in range (len(strat)):
            if i == 0 or i == 1:
                if strat[i] == 1:
                    x_tick.append("L")
                else:
                    x_tick.append("R")
            else:
                if strat[i] == 1:
                    x_tick.append("C")
                else:
                    x_tick.append("D")
        x_ticks.append(x_tick)
    return ["".join(map(str, x_ticks[i])) for i in range(len(strats))]

def plotStatesForStrats(game, pop_size, strats, stationary, states_distrib):


    weighted_states_distrib = np.divide(states_distrib, pop_size)
    nb_bars = len(strats[0][2:]) + 1        #Nb internal states + stationary distribution
    n = len(strats)

    ind = np.arange(n)
    width = 0.25

    #x = np.arange

    fig = plt.figure()
    plt.title(game)
    ax = fig.add_subplot(111)
    ax.set_ylabel("fraction of the population")
    color = iter(cm.rainbow(np.linspace(0, 1, nb_bars)))
    c = next(color)
    state_0 = []
    state_1 = []
    for i in range (len(weighted_states_distrib)):
        if hasOnlyOneDirection(strategies[i]):
            if strategies[i][0] == 1:  # Left
                state_0.append(stationary[i])
                state_1.append(0)
            else:
                state_0.append(0)
                state_1.append(stationary[i])
        else:
            state_0.append(weighted_states_distrib[i][0])
            state_1.append(weighted_states_distrib[i][1])

    bar1 = ax.bar(ind, stationary, width, color="r")
    bar2 = ax.bar(ind + width, state_0, width, color="g")
    bar3 = ax.bar(ind + width*2, state_1, width, color="y")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(makeXTicks(strats))

    ax.legend((bar1, bar2, bar3), ("stationary distribution", "state 0", "state 1"))
    #plt.xticks(x, makeXTicks(strats), rotation='vertical')
    #for i in range (n):
    #    plt.bar(x[i], stationary[i], width, color="g", label="stationary distribution")
    #    plt.bar(x[i]+ width, weighted_states_distrib[i][0], width, color="r", label="state "+str(0))
    #    plt.bar(x[i]+ width*2, weighted_states_distrib[i][1], width, color="b", label="state "+str(1))
    #plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()



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


def displayStatesForStrats(strats, states_distrib):
    for i in range (len(strats)):
        print("".join(displayStrat(strats[i])), ": ", end="")
        for j in range (len(states_distrib[i])):
            print(states_distrib[i, j], " agents in state", j, end="")
            if j != (len(states_distrib[i]) -1):
                print(", ", end="")
        print()

game = "SH"
nb_states = 2
Z = 150
nb_runs = 350
nb_sim = 500
stationary_filename = "stationaryDistrib/" + game + "/Rounds/stationary_" + str(nb_states) + "st.txt"

stationary = getStationaryFromFile(stationary_filename)
strategies = createStrategies(nb_states)

test_distrib = RunStationary(Z, nb_states, strategies, stationary, nb_runs, nb_sim)
states_distrib = test_distrib.runSimulation()

displayStatesForStrats(strategies, states_distrib)
plotStatesForStrats(game, Z, strategies, stationary, states_distrib)

