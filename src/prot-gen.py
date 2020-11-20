import emt6ro.simulation as sim
import numpy as np
import sys
import os
import itertools
import time

dirPath			= os.path.dirname(sys.argv[0])
EMTDataDirPath	= os.path.join(dirPath, "../../EMT6-Ro/data/")
dataDirPath		= os.path.join(dirPath, "../data/2000per_simulation/")
results			= os.path.join(dataDirPath, "results")
protocolTimes	= os.path.join(dataDirPath, "protocols/protocol_times_{}.csv".format(sys.argv[1]))
protocolResults	= os.path.join(dataDirPath, "results/protocol_results_{}.csv".format(sys.argv[1]))
parameters		= os.path.join(EMTDataDirPath, "default-parameters.json")
tumor			= os.path.join(EMTDataDirPath, "tumor-lib/tumor-1.txt")

params = sim.load_parameters(parameters)
state = sim.load_state(tumor, params)

if not os.path.exists(results):
	os.makedirs(results)

if os.path.exists(protocolResults):
	os.remove(protocolResults)

with open(protocolTimes, 'r') as f:
	for doses, times in itertools.zip_longest(f, f):
		if times is None:
			break
		protocol = []
		doses = doses.split(' ')
		times = times.split(' ')
		doses.pop()
		times.pop()
		for i, j in enumerate (doses):
			protocol.append((int(times[i]), float(doses[i])))
	
		experiment = sim.Experiment(params, [state], 2000, 1)
		
		experiment.run([protocol])
		start_time = time.time()
		
		res = experiment.get_results()
		end_time = time.time()
		print(end_time - start_time)

		with open(protocolResults, 'a') as g:
			for i in res:
				for j in i:
					g.write('[' + ', '.join(str(x) for x in j) + ']')
					g.write("\n")
