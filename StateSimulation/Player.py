

class Player:

    def __init__(self, pid, state, strategy, max_state):
        self.id = pid
        self.state = state
        self.strategy = strategy
        self.max_state = max_state


    def getID(self):
        return self.id

    def getState(self):
        return self.state


    def setState(self, new_state):
        self.state = new_state


    def getStrategy(self):
        return self.strategy

    def getMaxState(self):
        return self.max_state

    def changeState(self, observation):
        if observation == 1:      #1 -> C | 0 -> D
            #direction = self.strategy[0][self.state*(self.max_state+1)]
            direction = self.strategy[0]
            if direction == 1 and self.state > 0:         #Left:1
                self.state -= 1
            elif direction == 0 and self.state < self.max_state:      #Right:0
                self.state += 1
        else:
            #direction = self.strategy[0][self.state * (self.max_state + 1) + 1]
            direction = self.strategy[1]
            if direction == 1 and self.state > 0:     #Left:1
                self.state -= 1
            elif direction == 0 and self.state < self.max_state:      #Right:0
                self.state += 1

    def imitateStrat(self, new_strat):
        self.strategy = new_strat

    def play(self):
        action = self.strategy[2+self.state]
        return action



