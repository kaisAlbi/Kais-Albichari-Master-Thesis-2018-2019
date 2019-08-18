from FitnessGroups import *

def fitnessForDifferentT(game, nb_states):
    Z = 150
    R, P = 1,0
    t_values = [0,0.25,0.5,0.75,1]
    t_folder = ["T0", "T025", "T05", "T075", "T1"]
    s_values = [-t for t in t_values]
    strategies = reducedStrategies(createStrategies(nb_states))
    N = 150
    alpha = 0.8
    if game == "PD":
        t_values = [1, 1.25, 1.5, 1.75, 2]
        t_folder = ["T1", "T125", "T15", "T175", "T2"]

        s_values = [1-t for t in t_values]
    expr_filename = "ExpressionsAnalytical/Reduced/expressions_" + str(nb_states) + "st_groupsize_150_alpha08.pickle"

    for i in range (len(t_values)):
        fit_diff_matrix = computeFitDiffMatrix(strategies, Z, N, expr_filename, alpha, R, s_values[i], t_values[i], P)
        store_filename = "FitDiff/" + game + "/Reduced/" + t_folder[i] + "/fit_diff_" + str(nb_states) + "st_groupsize" + str(N)

        storePickle(store_filename, fit_diff_matrix)

if __name__ == '__main__':

    nb_states = 1
    print("start SH")
    fitnessForDifferentT("SH", nb_states)
    print("start PD")
    fitnessForDifferentT("PD", nb_states)