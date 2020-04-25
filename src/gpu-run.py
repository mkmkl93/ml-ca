import emt6ro.simulation as sim
import numpy as np
import sys
import os
import itertools

dirPath		= os.path.dirname(sys.argv[0])
EMTDataDirPath	= os.path.join(dirPath, "../../EMT6-Ro/data/")
dataDirPath	= os.path.join(dirPath, "../data/uniform_200k/")
results		= os.path.join(dataDirPath, "results")
protocolTimes	= os.path.join(dataDirPath, "protocols/protocol_times_{}.csv".format(sys.argv[1]))
protocolResults	= os.path.join(dataDirPath, "results/protocol_results_{}.csv".format(sys.argv[1]))
parameters	= os.path.join(EMTDataDirPath, "default-parameters.json")
tumor		= os.path.join(EMTDataDirPath, "tumor-lib/tumor-1.txt")

params = sim.load_parameters(parameters)
state = sim.load_state(tumor, params)

if not os.path.exists(results):
    os.makedirs(results)

if os.path.exists(protocolResults):
	os.remove(protocolResults)

protocols = []
with open(protocolTimes, 'r') as f:
	for doses, times in itertools.zip_longest(f, f):
		protocol = []
		doses = "0.625 0.625 0.875 0.75 2 0.75 0.875 0.875 1 0.875 0.75\n"
		times = "2300 10100 18000 22700 34300 36700 40600 45900 50100 64500 65400\n"
		doses = doses.split(' ')
		times = times.split(' ')
		doses.pop()
		times.pop()
		for i, j in enumerate (doses):
			protocol.append((int(times[i]), float(doses[i])))
		protocols.append(protocol)
	
		for j in range(1, 1000):
			experiment = sim.Experiment(params, [state] * len(protocols), 1, len(protocols))
			experiment.run(protocols)
			res = experiment.get_results()

			with open(protocolResults, 'a') as f:
				for i in res:
					f.write(str(i))
					f.write("\n")
		break
