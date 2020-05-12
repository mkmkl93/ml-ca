# Tumor growth cellular automaton simulation in C++ and results appoximation with ML

Authors:
Antoni Goldstein
Michał Kardaś
Piotr Kowalkowski
Magdalena Molenda
Łukasz Piekarski

Project done as a Bachelor thesis in Computer at University of Warsaw.
Continuation of the work of the previous group accessible here [tumor-ca](https://github.com/lamyiowce/tumor-ca) 
and here [EMR6-Ro](https://github.com/banasraf/EMT6-Ro).

## Thesis abstract
Existing research provides a numerical method based on cellular automata to simulate growth
of EMT6/Ro tumor spheroid under varying radiotherapy treatment protocols. As the space
of possible protocols is very vast, the model has been used to conduct a heuristic genetic
algorithm search for an optimal dosage and timing of irradiation. However, current imple-
mentation of the simulation makes the search computationally costly, which limits the extent
of explored solutions.
We have created multiple datasets with 200 000 protocols each and tested with various machine
learning algorithms. Thanks to the work of the previous group we had been able to use GPU to
speed up the process and for every simulation we have conducted 100 simulations and took the mean of
the results to predict expected value.

## Usage
Build tumor-simulation executable. In project's main folder run:
```
mkdir build
cd build
cmake ..
make
```
Usage:
Create protocols
```
./nasze-ca/build/protocol_generator
```
Run simulations using gpu
```
sbatch nasze-ca/gpu-gen.sl
```

Results will be in 
`nasze-ca/data/old_data/results/protocol_results_<number>.csv`.
