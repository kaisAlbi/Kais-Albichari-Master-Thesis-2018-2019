import csv

with open("params_eval_coop_pd_allT.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    T_folders = ["T1/", "T125/", "T15/", "T175/", "T2/"]
    nb_runs = 200
    writer.writerow(['RunID', 'SubFolder'])
    for run in range (nb_runs):
        for subfolder in T_folders:
            writer.writerow([str(run), subfolder])


