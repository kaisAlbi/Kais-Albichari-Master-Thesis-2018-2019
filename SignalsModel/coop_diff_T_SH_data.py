import SignalsModel
import sys

nb_sig = int(sys.argv[1])
T = float(sys.argv[2])
directory = str(sys.argv[3])
S = -T
R, P = 1, 0
Z = 150 #Population size
beta = 0.1
egt_model = SignalsModel.EGTModel(R, S, T, P, Z, beta, nb_sig)
filename = directory + "/SH_sig_"+str(nb_sig)+"_T_"+str(T)+".txt"

SignalsModel.save(filename, egt_model)