from sympy import *

class Tree:

    def __init__(self, current_state, strategy, weight):

        self.current_state = current_state
        self.max_state = len(strategy[2:])-1
        self.strategy = strategy
        self.expression = 0
        self.children = []
        self.weight = weight

    def getChildren(self):
        return self.children

    def getWeight(self):
        return self.weight

    def getAction(self):
        return self.strategy[self.current_state+2]

    def getExpr(self):
        return self.expression


    def computeNewState(self, transition):
        new_state = self.current_state
        if self.current_state > 0 and transition == 1:      #1 if Left
            new_state -= 1
        elif self.current_state < self.max_state and transition == 0:   #0 if Right
            new_state += 1
        return new_state

    def buildExpr(self, opp_action):
        """
        Choose the variable associated to the result of the interaction
        :param opp_action: opponent action
        :return: Symbol object corresponding to the variable
        """
        if self.getAction() == 1:
            if opp_action == 1:
                R = Symbol('R')
                self.expression = self.weight * R
            else:
                S = Symbol('S')
                self.expression = self.weight * S
        else:
            if opp_action == 1:
                T = Symbol('T')
                self.expression = self.weight * T
            else:
                P = Symbol('P')
                self.expression = self.weight * P

    def buildExprVariableActionsStrat(self, opp_strat):
        """
        Build the average expression of the current strategy against the other, knowing that
        the opponent' strategy has multiple possible actions
        :param opp_strat: opponent' strategy
        :return: expression
        """
        states_distrib = 1 / len(opp_strat[2:])
        count_C = opp_strat[2:].count(1)
        count_D = opp_strat[2:].count(0)
        if self.getAction() == 1:
            R = Symbol('R')
            S = Symbol('S')
            self.expression = self.weight * states_distrib * (count_C * R + count_D * S)
        else:
            T = Symbol('T')
            P = Symbol('P')
            self.expression = self.weight * states_distrib * (count_C * T + count_D * P)

    def addChildrenVariableAction(self, opp_strat):
        """
        Add children to current node, knowing that the opponent' strategy can have multiple actions
        :param opp_strat: opponent' strategy
        """
        states_distrib = 1 / len(opp_strat[2:])
        distrib_C = opp_strat[2:].count(1) * states_distrib
        distrib_D = opp_strat[2:].count(0) * states_distrib

        transition_c = self.strategy[0]     #1 if Left, 0 if Right
        transition_d = self.strategy[1]     #1 if Left, 0 if Right
        new_state_c = self.computeNewState(transition_c)
        new_state_d = self.computeNewState(transition_d)
        trans = Symbol('a')         #symbol corresponding to the probability of changing the state
        if new_state_c != self.current_state:
            if new_state_d != self.current_state:           #Both new_states are possible
                weight_c = trans * distrib_C * self.getWeight()
                weight_d = trans * distrib_D * self.getWeight()
                weight_no_change = ((1-trans) * distrib_C + (1-trans) * distrib_D) * self.getWeight()
                child_change_c = Tree(new_state_c, self.strategy, weight_c)
                child_change_d = Tree(new_state_d, self.strategy, weight_d)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change)

                self.children.append(child_no_change)
                self.children.append(child_change_c)
                self.children.append(child_change_d)


            else:                                              #Only new_state c is possible

                weight_c = trans * distrib_C * self.getWeight()
                weight_no_change = (distrib_D + ((1 - trans) * distrib_C))* self.getWeight()
                child_change_c = Tree(new_state_c, self.strategy, weight_c)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change)
                self.children.append(child_change_c)
                self.children.append(child_no_change)

        elif new_state_d != self.current_state:             #Only new state d is possible
            weight_d = trans * distrib_D * self.getWeight()
            weight_no_change = (distrib_C + (1 - trans) * distrib_D) * self.getWeight()
            child_change_d = Tree(new_state_d, self.strategy, weight_d)
            child_no_change = Tree(self.current_state, self.strategy, weight_no_change)
            self.children.append(child_change_d)
            self.children.append(child_no_change)


        else:                   #No new state at all
            weight_no_change = self.getWeight()
            only_child = Tree(self.current_state, self.strategy, weight_no_change)
            self.children.append(only_child)

    def addChildren(self, opp_action):
        """
        Add children to the current node, knowing that the opponent's action is fixed
        :param opp_action: opponent's action
        """
        if opp_action == 1:
            transition = self.strategy[0]
            new_state = self.computeNewState(transition)
            if new_state != self.current_state:
                trans_c = Symbol('a')        #symbol corresponding to the probability of changing the state
                weight_c = trans_c * self.getWeight()
                weight_no_change = (1 - trans_c) * self.getWeight()
                child_change_c = Tree(new_state, self.strategy, weight_c)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change)
                self.children.append(child_change_c)
                self.children.append(child_no_change)
            else:
                weight_no_change = self.getWeight()
                only_child = Tree(self.current_state, self.strategy, weight_no_change)
                self.children.append(only_child)
        else:
            transition = self.strategy[1]
            new_state = self.computeNewState(transition)
            if new_state != self.current_state:
                trans_d = Symbol('a')        #symbol corresponding to the probability of changing the state
                weight_d = trans_d * self.getWeight()
                weight_no_change = (1 - trans_d) * self.getWeight()
                child_change_c = Tree(new_state, self.strategy, weight_d)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change)
                self.children.append(child_change_c)
                self.children.append(child_no_change)
            else:
                weight_no_change = self.getWeight()
                only_child = Tree(self.current_state, self.strategy, weight_no_change)
                self.children.append(only_child)

