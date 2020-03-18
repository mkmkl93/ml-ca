import tensorflow as tf
import ast
import json

def create_hparams(hparam_string=None, json=False):
	"""Create model hyperparameters. Parse nondefault from given string."""
	hparams = {
		"learning_rate" : 0.01,
		"cnn_filters" : [16, 32, 64],
		"batch_size" : 128,
		"num_epochs" : 500,
		"shuffle" : False,
		"num_threads" : 2,
		"hidden_size" : 256,
		"hidden_units" : [512, 1024, 512, 256],
		"num_hidden_layers" : 4,
		"initializer" : "uniform_unit_scaling",
		"initializer_gain" : 1.0,
		"weight_decay" : 0.0,
		"dropout" : 0.1,
		"activation" : 'tanh',
		"l1_regularization_strength" : 0.001
		}
	if hparam_string:
		if json:
			print(hparam_string)
			hparams = json.loads(hparam_string)
		else:
			hparams = ast.literal_eval(hparam_string)
	return hparams
