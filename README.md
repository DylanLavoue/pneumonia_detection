#Zoidberg2.0 : Computer aided diagnosis

The aim of this project is to use machine learning to help doctors detecting pneumonia. We, thus, have to classify x-ray images into 3 classes : normal, viral pneumonia, bacterial pneumonia.
The project has 2 parts :

Analyze and model building in python. All results can be shown in notebooks
Model deployment

#Architecture

Th directory structure below is widely inspired by the cookiecutter-data-science project.

.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── interim         <- Intermediate data that has been transformed
│   ├── processed       <- Final, canonical data for modeling
│   └── raw             <- Original data dump
│
├── deployment          <- API & Front for model deployment
│
├── models              <- Trained models (history, checkpoint, etc.)
│
├── notebooks
│   ├── data_processing <- Jupyter notebooks for data analysis and
│   │                      preprocessing
│   ├── models          <- Jupyter notebooks to train models
│   └── visualization   <- Jupyter notebooks to make some useful
│                          visualizations
│
├── report              <- Generated report
│
├── requirements.yml    <- Requirements file for reproducing the analysis
│                          environment
│
├── src                 <- Source code for use in this project
    ├── data            <- Scripts to handle data (download, metrics,
    │                      tensorflow)
    ├── tests           <- Unit testing
    └── visualization   <- Scripts to help visualizations


#Visualization with Notebook

Notebooks can be run either with Google Colab or in a local environment. Steps below are provided to run them locally. If you want to use Colab, you need to load this repo on your google drive, change drive folder in notebooks and upload needed data Section 4.

#Create isolated virtual environment

conda activate zoidberg_env