import numpy as np
from analyzePayoffsDistrib import createStrategies, displayableStrat
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm

class StateDistrib:

    def __init__(self, strategy, nb_states):
        self.strategy = strategy
        self.nb_states = nb_states


    def changeState(self, current_state, other_action):
        """
        Switch the state of a strategy according to the transition directions, the other action
        and some probability of changing the state
        :param current_state: current state
        :param other_action: opponent action
        :return: new state
        """
        transition = self.strategy[0]
        if other_action == 0:  # Opponent defected
            transition = self.strategy[1]
        if current_state > 0 and transition == 1:  # Possible to get left
            if np.random.rand() < 0.8:
                return current_state - 1
            else:
                return current_state
        elif current_state < (self.nb_states - 1) and transition == 0:  # Possible to get right
            if np.random.rand() < 0.8:
                return current_state + 1
            else:
                return current_state
        else:
            return current_state


    def runStatesDistrib(self, nb_sim, nb_rounds):
        """
        Simulate encounters to see the evolution of the states distributions
        :param nb_sim: number of simulations
        :param nb_rounds: number of rounds
        """
        total_states = np.zeros((nb_sim, nb_rounds+1))
        for i in range(nb_sim):

            state_A = np.random.choice(self.nb_states)
            state_B = np.random.choice(self.nb_states)
            state_distrib_A = np.zeros(nb_rounds+1)
            state_distrib_A[0] = state_A
            for j in range(nb_rounds):
                action_A = self.strategy[2 + state_A]
                action_B = self.strategy[2 + state_B]

                state_A = self.changeState(state_A,  action_B)
                state_B = self.changeState(state_B, action_A)
                state_distrib_A[j + 1] = state_A
            total_states[i] = state_distrib_A

        return total_states


def plotStateDistrib(setOfStrats, setOfStatesDistrib):

    n= len(setOfStrats)
    x = [i for i in range(len(setOfStatesDistrib[0]))]

    fig = plt.figure()
    for index in range (n):
        ax = plt.subplot(n/2, n/2, index+1)
        strat = setOfStrats[index]
        plt.ylabel("state 1 distribution")
        plt.xlabel("".join(map(str, displayableStrat(strat))))
        plt.ylim(-0.1, 1.1)

        #for i in range(n):
        plt.plot(x, setOfStatesDistrib[index])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.show()



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

def plotVariableStateStrats(var_states_strats, setOfStatesDistrib_0, setOfStatesDistrib_1):
    nb_bars = 2  # Nb internal states
    n = len(var_states_strats)

    ind = np.arange(n)
    width = 0.25


    fig = plt.figure()
    plt.suptitle("Distribution of states")
    plt.title("in monomorphic population")

    ax = fig.add_subplot(111)
    ax.set_ylabel("proportion")
    color = iter(cm.rainbow(np.linspace(0, 1, nb_bars)))
    c = next(color)

    bar1 = ax.bar(ind, setOfStatesDistrib_0, width, color="r")
    bar2 = ax.bar(ind + width, setOfStatesDistrib_1, width, color="g")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(makeXTicks(var_states_strats))

    ax.legend((bar1, bar2), ("state 0", "state 1"), loc='upper right')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


if __name__ == '__main__':
    nb_states = 2
    strategies = createStrategies(nb_states)
    #var_states_strats = [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]
    nb_sim = 500000
    nb_rounds = 10
    #for strat in var_states_strats:
    #    states_distrib = StateDistrib(strat, nb_states)
    #    total_states = states_distrib.runStatesDistrib(nb_sim, nb_rounds)
    #    avg_states_over_rounds = np.mean(total_states, axis=0)
    #    strat_name = "".join(map(str, displayableStrat(strat)))
    #    distrib_filename = "StatesDistrib/" + strat_name + ".txt"
    #    with open(distrib_filename, "a+") as f:
    #        json.dump(avg_states_over_rounds.tolist(), f)
     #   print(strat, " done")

    #for i in range (0,len(strategies), 4):
    #    setOfStrats = [strategies[i], strategies[i + 1], strategies[i + 2], strategies[i + 3]]
    #    setOfStatesDistrib = []
    #    for strat in setOfStrats:
    #        strat_name = "".join(map(str, displayableStrat(strat)))
    #        distrib_filename = "StatesDistrib/" + strat_name + ".txt"
    #        with open(distrib_filename, "r") as f:
    #            states_distrib = json.load(f)
    #            setOfStatesDistrib.append(states_distrib)
    #    plotStateDistrib(setOfStrats, setOfStatesDistrib)

    setOfStatesDistrib_0 = []
    setOfStatesDistrib_1 = []
    for strat in strategies:
        strat_name = "".join(map(str, displayableStrat(strat)))
        distrib_filename = "StatesDistrib/" + strat_name + ".txt"
        with open(distrib_filename, "r") as f:
            states_distrib = json.load(f)
            setOfStatesDistrib_0.append(1 - states_distrib[-1])
            setOfStatesDistrib_1.append(states_distrib[-1])
    plotVariableStateStrats(strategies, setOfStatesDistrib_0, setOfStatesDistrib_1)
