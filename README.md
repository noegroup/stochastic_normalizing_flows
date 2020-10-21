# INSTRUCTIONS FOR RUNNING CODE FOR STOCHASTIC NORMALIZING FLOW PAPER
#
# 0) System requirements
# All experiments were run with Python 3.4 and PyTorch 1.2 on MacOS.
# They are expected to work on MacOS and Linux systems with these or newer Python and PyTorch versions
#
# 1) Install packages

# Install bgtorch flow package
cd bgtorch
python setup.py develop
cd ..

# Install snf_code package, specialized code for this paper
cd snf_code
python setup.py develop
cd ..

# Optional: install OpenMM for running experiment 3
conda install -c omnia openmm 
conda install -c omnia openmmtools 

# 2) Run Experiments

# To run experiments 1-3, open and run the respective notebooks with jupyter
# To run experiment 4, run the respective Python file directly

