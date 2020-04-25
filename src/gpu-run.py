import emt6ro.simulation as sim
import numpy as np
import sys
import os
import itertools

dirPath = os.path.dirname(sys.argv[0])
EMTDataDirPath = os.path.join(dirPath, "../../EMT6-Ro/data/")
dataDirPath = os.path.join(dirPath, "../data/uniform_200k/protocols/")
params = sim.load_parameters(os.path.join(EMTDataDirPath, "default-parameters.json"))
state = sim.load_state(os.path.join(EMTDataDirPath, "tumor-lib/tumor-1.txt"), params)

protocols = []
with open(os.path.join(dataDirPath, "protocol_times_{}.csv".format(sys.argv[1])), 'r') as f:
	for doses, times in itertools.zip_longest(f, f):
		protocol = []
		doses = doses.split(' ')
		times = times.split(' ')
		doses.pop()
		times.pop()
		for i, j in enumerate (doses):
			print(i, j)
			protocol.append((int(times[i]), float(doses[i])))
		protocols.append(protocol)
		break

print(protocols)

experiment = sim.Experiment(params, [state] * len(protocols), 200, len(protocols))  

experiment.run(protocols)
res = experiment.get_results()
print("Protocol 1 result")
print(np.mean(res[0]))

