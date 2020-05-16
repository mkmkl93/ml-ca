# If the first argument is "run"...
ifeq (go,$(firstword $(MAKECMDGOALS)))
        # use the rest as arguments for "run"
        RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
	# ...and turn them into do-nothing targets
        $(eval $(RUN_ARGS):;@:)
endif

ifneq ($(RUN_ARGS),)
       RUN_ARGS := --nodelist=rysy-n${RUN_ARGS}
endif

go:
	srun -p gpu -A GR79-29 --gres=gpu:1 -c 18 --ntasks-per-node=1 ${RUN_ARGS} --pty bash -l
