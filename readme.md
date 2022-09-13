# Unsupervised Model Selection for Time-series Anomaly Detection

Hundreds of models for anomaly detection in time-series are available to practitioners, but no method exists to select the best model and its hyperparameters for a given dataset when labels are not available. We construct three classes of surrogate metrics which we show to be correlated with common supervised anomaly detection accuracy metrics such as the F1 score. 
The three classes of metrics are prediction accuracy, centrality, and performance on injected synthetic anomalies. We show that some of the surrogate metrics are useful for unsupervised model selection but not sufficient by themselves. To this end, we treat metric combinations as a rank aggregation problem and propose a robust rank aggregation approach. Large scale experiments on multiple real-world datasets demonstrate that our proposed unsupervised aggregation approach is as effective as selecting the best model based on collecting anomaly labels.

## Contents

1. [Overview of Methodology](#methodology) 
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Code Organization](#code)
5. [Citation](#citation)
----

<a id="methodology"></a>
## Overview of Methodology 

<p align="center">
<img height ="300px" src="assets/methods.png">
</p>

Figure 1: *The Model Selection Workflow.* We identify three classes of surrogate metrics of model quality, and propose a novel robust rank aggregation framework to combine multiple rankings from metrics. 

----

<a id="datasets"></a>
## Datasets

We carry out experiments on two popular and widely used real-world collections with diverse time-series and anomalies: (1) UCR Anomaly Archive (UCR) (Wu & Keogh, 2021), and (2) Server Machine Dataset (SMD) (Su et al., 2019). 

We make both the datasets available through the `datasets.load.load_data(...)` function in our support library [`PyMad`](https://github.com/cchallu/PyMAD). 

To load the UCR dataset: 

```python
    import sys
    sys.path.append('/path_to_pymad')
    
    from src.pymad.datasets.load import load_data

    # Load the data
    ENTITY = 'anomaly_archive' # 'anomaly_archive' OR 'smd' 
    
    DATASET = '028_UCR_Anomaly_DISTORTEDInternalBleeding17' # Name of timeseries in UCR or machine in SMD
    
    train_data = load_data(dataset=DATASET, 
                           group='train', 
                           entities=[ENTITY], 
                           downsampling=None, 
                           root_dir='/path_to_dataset_dir', 
                           normalize=True, 
                           verbose=True)
    
    test_data = load_data(dataset=DATASET, 
                          group='test', 
                          entities=[ENTITY], 
                          downsampling=None, 
                          root_dir='/path_to_dataset_dir', 
                          normalize=True, 
                          verbose=True)

```

To install `PyMAD`, please refer the [installation instructions](#installation). 


----

<a id="installation"></a>
## Installation

We recommend installing [Ananconda](https://conda.io/projects/conda/en/latest/index.html) to run our code. To install Anaconda, review the installation instructions [here](https://docs.anaconda.com/anaconda/install/). 

To setup the environment using [`conda`](https://conda.io/projects/conda/en/latest/index.html), run the following commands:

```console
    # To create environment from environment_explicit.yml file
    foo@bar:~$ conda env create -f environment_explicit.yml
    
    # To activate the environment
    foo@bar:~$ conda activate modelselect 
    
    # To verify if the new environment was installed correctly
    foo@bar:~$ conda env list 

```

Our support library [`PyMAD`](https://github.com/cchallu/PyMAD) is available on Github and can be cloned using the following commands: 

```console

    foo@bar:~$ git clone https://github.com/cchallu/PyMAD.git

```

**NOTE:** Currently PyMAD is under active development and has not been made public. Please contact <a href="mailto:cchallu@andrew.cmu.edu">Cristian Challu</a> or <a href="mailto:mgoswami@andrew.cmu.edu">Mononito Goswami</a> for access to the repository.

----

<a id="code"></a>
## Code Organization

```bash

    ├── assets 
    │   └── methods.png
    ├── data
    │   └── generate_synthetic_data.py # Generate synthetic time-series
    ├── dev_scripts # Development scripts 
    │   ├── results_critical_difference_diagram.ipynb # Generate critical difference diagrams. Currently not used in the paper. 
    │   └── visualization.ipynb # Generate visualizations for the paper
    ├── environment.yml # Complete Conda environment file. Likely to be incompatible across platforms. 
    ├── environment_explicit.yml # Explicit Conda environment file. Likely to be compatible across platforms. 
    ├── evaluation # Evaluate surrogate metrics, rank aggregation across datasets
    │   ├── __init__.py
    │   └── evaluation.py 
    ├── distributions # Kendall's tau distance, Plackett-Luce distribution, synthetic data experiments
    │   ├── __init__.py
    │   ├── experiments
            ├── Placekett-Luce.ipynb 
            ├── Position Weighted Kendall Tau.ipynb
            ├── Shapely.ipynb
            └── Trimmed Aggregation Experiments.ipynb
    │   ├── pl_model.py # Code for Placket
    │   └── sampling.py # Sample Mallows and PL distributions with noise
    ├── meta.py # File containing meta information 
    ├── metrics # Generate prediction error, synthetic and centrality metrics. Also includes code to evaluate ranking algorithms. 
    │   ├── metrics.py
    │   └── ranking_metrics.py
    ├── model_selection # Model selection code
    │   ├── __init__.py
    │   ├── anomaly_parameters.py # Synthetic anomaly generation hyper-parameters
    │   ├── inject_anomalies.py # Inject various kinds of synthetic anomalies
    │   ├── model_selection.py # Model selection class. Interface to model selection capablities.
    │   ├── model_selection_utils.py 
    │   ├── rank_aggregation.py # Aggregate ranks
    │   └── utils.py
    ├── model_trainer # Train models on multiple datasets
    │   ├── __init__.py
    │   ├── entities.py # List of entities in all datasets
    │   ├── hyperparameter_grids.py # Hyper-parameter grids for all models to be trained
    │   └── trainer.py # Trainer class to train multiple models from PyMAD
    ├── readme.md # Readme file
    ├── results # Results directory
    │   ├── data # Pickled results files
    │   │   ├── aggregate_stats.pkl
    │   │   ├── aggregate_stats_10Aug_top5_corrected.pkl
    │   │   ├── aggregate_stats_9Aug_top5.pkl
    │   │   ├── aggregate_stats_anomaly_archive.pkl
    │   │   └── aggregate_stats_smd.pkl
    │   └── figures # Figures for paper
    │       ├── anomalies.pdf
    │       ├── full_results_box_plot_pooled_stats_23-12-08-12-2022.pdf
    │       └── predictions.pdf
    ├── scripts # Various scripts to train models, evaluate them and perform model selection on all datasets etc. 
    │   ├── Categorize UCR Anomaly Archive.ipynb # Categorize UCR datasets
    │   ├── check_number_of_evaluated_models.py # Check the number of evaluated models
    │   ├── check_number_of_trained_models.py # Check the number of trained models
    │   ├── compute_pooled_results.py # Compute pooled model selection results of all surrogate metrics & model selection strategies
    │   ├── evalute_all_models.py # Evaluate model in terms of their prediction error, model centrality and performanc on synthetically injected anomalies
    │   ├── results.ipynb # View results (box-plots), conduct significance testing and create tables for paper
    │   └── train_all_models.py # Train all models on multiple datasets
    └── tests # Test various aspects of the project
        ├── test_anomaly_injection.ipynb # Test anomaly injection pipeline
        ├── test_pymad_models.ipynb # Test PyMAD model training pipeline
        └── test_rank_models.ipynb # Test model selection pipeline

```

----
<a id="citation"></a>
## Citation

If you use our code please cite our paper: 

```bibtex

    @article{
        goswami2022unsupervised,
        title={Unsupervised Model Selection for Time-series Anomaly Detection},
        author={Goswami, Mononito and Challu, Cristian and Callot, Laurent and Minorics, Lenon and Kan, Andrey},
        journal={Under Review.},
        year={2022},
    }

```
