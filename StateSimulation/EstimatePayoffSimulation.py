import numpy as np
import json
from itertools import product


class EstimatePayoffSimulation:
    def __init__(self, R, S, T, P, nb_states, nb_rounds, nb_sim):
        self.alpha = 0.8
        self.R, self.S, self.T, self.P = R, S, T, P
        self.nb_states = nb_states
        self.strategies = self.createStrategies()
        #self.strat_A = strat_A
        #self.strat_B = strat_B

        self.nb_rounds = nb_rounds
        self.nb_sim = nb_sim



    def getPayoff(self, first_action, second_action):
        if first_action == 1:
            if second_action == 1:
                return self.R, self.R
            else:
                return self.S, self.T
        else:
            if second_action == 1:
                return self.T, self.R
            else:
                return self.P, self.P

    def createStrategies(self):
        """
        Generate all possible combination of strategies, depending on the number of states
        :return: list of all strategies in the form [T_c,T_d, action state_0, ..., action state_maxState]
                transition = 1 = Left ; action = 1 = C
        """
        action_choice = list(list(item) for item in product("CD", repeat=self.nb_states))
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
                #strats.append([state_c, action])
        return strats

    def changeState(self, current_state, strat, observation):
        transition = strat[0]
        if observation == 0:    #Opponent defected
            transition = strat[1]
        if current_state > 0 and transition == 1:   #Possible to get left
            if np.random.rand() < self.alpha:
                return current_state - 1
            else:
                return current_state
        elif current_state < (self.nb_states - 1) and transition == 0:   #Possible to get right
            if np.random.rand() < self.alpha:
                return current_state + 1
            else:
                return current_state
        else:
            return current_state



    def run(self, filename):
        n = len(self.strategies)
        for i in range (n):
            strat_A = self.strategies[i]
            for j in range (i, n):
                strat_B = self.strategies[j]
                if strat_A == [0,0,0,1] and strat_B == [1,0,0,1]:
                    print("new pair of strat : ", strat_A, " against ", strat_B)
                payoffs_A, payoffs_B = self.runForTwoStrats(strat_A, strat_B)

                self.storePayoffs(filename, strat_A, payoffs_A, strat_B, payoffs_B)

    def runForTwoStrats(self, strat_A, strat_B):
        payoffs_A = np.zeros(self.nb_sim)
        payoffs_B = np.zeros(self.nb_sim)
        for i in range (self.nb_sim):
            #if i % 250 == 0:
            #    print("sim ", i)
            state_A = np.random.choice(self.nb_states)
            state_B = np.random.choice(self.nb_states)
            payoff_A = 0
            payoff_B = 0
            for j in range (self.nb_rounds):
                action_A = strat_A[2 + state_A]
                action_B = strat_B[2 + state_B]

                cur_payoff_A, cur_payoff_B = self.getPayoff(action_A, action_B)

                state_A = self.changeState(state_A, strat_A, action_B)
                state_B = self.changeState(state_B, strat_B, action_A)
                payoff_A += cur_payoff_A
                payoff_B += cur_payoff_B

            #print(type(payoffs_A))
            payoff_A /= self.nb_rounds
            if strat_A == [0, 0, 0, 1] and strat_B == [1, 0, 0, 1]:
                print(payoff_A)
            payoff_B /= self.nb_rounds
            if strat_B == [0, 0, 0, 1] and strat_A == [1, 0, 0, 1]:
                print(payoff_B)
            payoffs_A[i] = payoff_A
            payoffs_B[i] = payoff_B
        return payoffs_A, payoffs_B

    def storePayoffs(self, filename, strat_A, payoffs_A, strat_B, payoffs_B):
        with open(filename, "a+") as f:
            all = [strat_A, strat_B, payoffs_A.tolist()]
            if strat_A != strat_B:
                all.append(payoffs_B.tolist())
            json.dump(all, f)
            f.write("\n")

def getPayoff(first_action, second_action, R, S, T, P):
    if first_action == 1:
        if second_action == 1:
            return R, R
        else:
            return S, T
    else:
        if second_action == 1:
            return T, S
        else:
            return P, P

def changeState(current_state, strat, observation):
    transition = strat[0]
    if observation == 0:    #Opponent defected
        transition = strat[1]
    if current_state > 0 and transition == 1:   #Possible to get left
        if np.random.rand() < 0.8:
            return current_state - 1
        else:
            return current_state
    elif current_state < (nb_states - 1) and transition == 0:   #Possible to get right
        if np.random.rand() < 0.8:
            return current_state + 1
        else:
            return current_state
    else:
        return current_state

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



def run(filename, strats, nb_sim, nb_rounds, R, S, T, P):
    n = len(strats)
    for i in range (n):
        print("starting strat nb ", i)
        strat_A = strats[i]
        for j in range (i, n):
            strat_B = strats[j]
            payoffs_A, payoffs_B = runForTwoStrats(strat_A, strat_B, nb_sim, nb_rounds, R, S, T, P)

            storePayoffs(filename, strat_A, payoffs_A, strat_B, payoffs_B)


def runForTwoStrats(strat_A, strat_B, nb_sim, nb_rounds, R, S, T, P):
    payoffs_A = np.zeros(nb_sim)
    payoffs_B = np.zeros(nb_sim)
    for i in range(nb_sim):
        # if i % 250 == 0:
        #    print("sim ", i)
        state_A = np.random.choice(nb_states)
        state_B = np.random.choice(nb_states)
        payoff_A = 0
        payoff_B = 0
        for j in range(nb_rounds):
            action_A = strat_A[2 + state_A]
            action_B = strat_B[2 + state_B]

            cur_payoff_A, cur_payoff_B = getPayoff(action_A, action_B, R, S, T, P)

            state_A = changeState(state_A, strat_A, action_B)
            state_B = changeState(state_B, strat_B, action_A)
            payoff_A += cur_payoff_A
            payoff_B += cur_payoff_B
        payoff_A /= nb_rounds
        payoff_B /= nb_rounds

        payoffs_A[i] = payoff_A
        payoffs_B[i] = payoff_B
        #if strat_B == [0, 0, 0, 1] and strat_A == [1, 0, 0, 1]:
        #    print(payoff_B)

    return payoffs_A, payoffs_B

def storePayoffs(filename, strat_A, payoffs_A, strat_B, payoffs_B):
    with open(filename, "a+") as f:
        all = [strat_A, strat_B, payoffs_A.tolist()]
        if strat_A != strat_B:
            all.append(payoffs_B.tolist())
        json.dump(all, f)
        f.write("\n")

def loadMean(strat_A, strat_B):
    with open("test.txt", "r") as f:
        for line in f.readlines():
            all = json.loads(line)
            cur_strat_A = all[0]
            cur_strat_B = all[1]
            if cur_strat_A == strat_A and cur_strat_B == strat_B:
                payoffs_A = all[2]
                print(len(payoffs_A))
                print(payoffs_A)
                print(np.mean(np.asarray(payoffs_A), axis = 0))

if __name__ == '__main__':


    R, P = 1, 0
    T = 1.5
    S = -0.5
    nb_states = 1
    nb_rounds = 10
    nb_sim = 10000
    filename = "payoffs_simulation_1st_pd.txt"
    estimate_payoffs = EstimatePayoffSimulation(R, S, T, P, nb_states, nb_rounds, nb_sim)
    estimate_payoffs.run(filename)

    strategies = createStrategies(nb_states)
    print(strategies)
    run(filename, strategies, nb_sim, nb_rounds, R, S, T, P)
    """
    strat_A = [1,0,1,0]
    strat_B = [1,0,1,0]
    #loadMean(strat_A, strat_B)
    
    payoffs_A = np.zeros(nb_sim)
    payoffs_B = np.zeros(nb_sim)
    for i in range(nb_sim):
        # if i % 250 == 0:
        #    print("sim ", i)
        state_A = np.random.choice(nb_states)
        state_B = np.random.choice(nb_states)
        payoff_A = 0
        payoff_B = 0
        for j in range(nb_rounds):
            action_A = strat_A[2 + state_A]
            action_B = strat_B[2 + state_B]
    
            cur_payoff_A, cur_payoff_B = getPayoff(action_A, action_B, R, S, T, P)
    
            state_A = changeState(state_A, strat_A, action_B)
            state_B = changeState(state_B, strat_B, action_A)
            payoff_A += cur_payoff_A
            payoff_B += cur_payoff_B
        payoff_A /= nb_rounds
        payoff_B /= nb_rounds
        #print(payoff_A)
        payoffs_A[i] = payoff_A
        payoffs_B[i] = payoff_B
    print(np.mean(payoffs_A))
    print(np.mean(payoffs_B))
    
    #print("mean : ", np.mean(payoffs_A))
    """