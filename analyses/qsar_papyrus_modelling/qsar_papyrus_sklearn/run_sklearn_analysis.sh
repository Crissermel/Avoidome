#!/bin/bash
# Script to run scikit-learn model comparison in the myenv environment

echo "Starting Scikit-learn Model Comparison Analysis"
echo "==============================================="

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate myenv

# Change to the correct directory
cd /home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/qsar_papyrus_sklearn

# Run the scikit-learn model comparison
echo "Running scikit-learn model comparison..."
python sklearn_model_comparison.py

echo "Analysis completed!"
echo "Results saved in: /home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/qsar_papyrus_sklearn/sklearn_results/" 