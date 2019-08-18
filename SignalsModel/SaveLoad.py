import json
class SaveLoad:

    def save(self, filename, strats, stationary_dist):
        f = open(filename, "w")
        all = [strats, stationary_dist]
        json.dump(all, f)
        f.close()

    def load(self, filename):
        f = open(filename, "r")
        strats, stationary_dist = json.load(f)
        return strats, stationary_dist