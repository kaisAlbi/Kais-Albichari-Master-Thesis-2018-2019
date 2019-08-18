class Tree:
    #static_rho = np.float(0.8)          #Probability of state transition after an interaction. Fixed for now
    def __init__(self, current_state, strategy, weight, max_state, trans_states_proba):
        self.current_state = current_state
        self.max_state = max_state
        self.strategy = strategy
        self.trans_proba = trans_states_proba
        self.children = []
        self.siblings = []
        self.weight = weight

    def getChildren(self):
        return self.children

    def getWeight(self):
        return self.weight

    def getProbaC(self):
        return self.strategy[self.current_state+2]* self.weight

    def getRhoForStates(self, state):
        return self.trans_proba[state]

    def addAllSiblings(self, siblings_list):
        for first in range (len(siblings_list)):
            for second in range (len(siblings_list)):
                if first != second:
                    siblings_list[first].addSibling(siblings_list[second])

    def addSibling(self, sibling):
        self.siblings.append(sibling)

    def getSiblings(self):
        return self.siblings

    def computeNewState(self, transition):
        new_state = self.current_state
        if self.current_state > 0 and transition == 1:      #1 if Left
            new_state -= 1
        elif self.current_state < self.max_state and transition == 0:   #0 if Right
            new_state += 1
        return new_state


    def addChildren(self, proba_c2):
        """
        Add all possible children to current node, depending on what the probability of cooperation of the other strategy is
        :param proba_c2: probability of cooperation of the other strategy
        """
        transition_c = self.strategy[0]     #1 if Left, 0 if Right
        transition_d = self.strategy[1]     #1 if Left, 0 if Right
        new_state_c = self.computeNewState(transition_c)
        new_state_d = self.computeNewState(transition_d)
        if new_state_c != self.current_state and proba_c2 > 0:
            if new_state_d != self.current_state and (1 - proba_c2) > 0:           #Both new_states are possible
                trans_c = self.getRhoForStates(new_state_c)
                trans_d = self.getRhoForStates(new_state_d)
                weight_c = round(trans_c * proba_c2 * self.getWeight(), 10)
                weight_d = round(trans_d * (1-proba_c2) * self.getWeight(), 10)
                weight_no_change = round(round(1 - trans_c, 1) * round(1 - trans_d, 1)* self.getWeight(), 10)
                child_change_c = Tree(new_state_c, self.strategy, weight_c, self.max_state, self.trans_proba)
                child_change_d = Tree(new_state_d, self.strategy, weight_d, self.max_state, self.trans_proba)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change, self.max_state, self.trans_proba)

                self.children.append(child_no_change)
                self.children.append(child_change_c)
                self.children.append(child_change_d)

                self.addAllSiblings([child_change_c, child_change_d, child_no_change])

            else:                                              #Only new_state c is possible

                trans_c = self.getRhoForStates(new_state_c)
                weight_c = round(trans_c * proba_c2 * self.getWeight(),10)
                weight_no_change = round(round((1 - proba_c2) + ((1 - trans_c) * proba_c2),10)* self.getWeight(), 10)
                child_change_c = Tree(new_state_c, self.strategy, weight_c, self.max_state, self.trans_proba)
                child_no_change = Tree(self.current_state, self.strategy, weight_no_change, self.max_state, self.trans_proba)
                self.children.append(child_change_c)
                self.children.append(child_no_change)
                self.addAllSiblings([child_change_c, child_no_change])

        elif new_state_d != self.current_state and (1 - proba_c2) > 0:             #Only new state d is possible
            trans_d = self.getRhoForStates(new_state_d)
            weight_d = round(trans_d * (1 - proba_c2) * self.getWeight(),10)
            weight_no_change = round(round(proba_c2 + (1 - trans_d) * (1 - proba_c2), 10) * self.getWeight(), 10)
            child_change_d = Tree(new_state_d, self.strategy, weight_d, self.max_state, self.trans_proba)
            child_no_change = Tree(self.current_state, self.strategy, weight_no_change ,
                                   self.max_state, self.trans_proba)
            self.children.append(child_change_d)
            self.children.append(child_no_change)

            self.addAllSiblings([child_change_d, child_no_change])

        else:                   #No new state at all
            weight_no_change = self.getWeight()
            only_child = Tree(self.current_state, self.strategy, weight_no_change, self.max_state, self.trans_proba)
            self.children.append(only_child)


