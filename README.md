
Installation and running experiments
------------
**System requirements**
All experiments were run with Python 3.7 and PyTorch 1.5 on MacOS.
They are expected to work on MacOS and Linux systems with these or newer Python and PyTorch versions

**Installation**
Install the bgtorch flow package
```
cd bgtorch
python setup.py develop
cd ..
```

Install snf_code package (specialized code for this paper)
```
cd snf_code
python setup.py develop
cd ..
```

Optional: install OpenMM for running experiment 3
```
conda install -c omnia openmm 
conda install -c omnia openmmtools 
```

**Run Experiments**
* To run experiments 1-3, open and run the respective notebooks with jupyter
* To run experiment 4, run the respective Python file directly

