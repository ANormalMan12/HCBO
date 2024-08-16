# High-Dimensional Causal Bayesian Optimization

The code and appendix for the paper "High-Dimensional Causal Bayesian Optimization" published in ECAI 2024. The appendix is in the `appendix.pdf`, which is contained in `paper` folder.

The appendix will be uploaded soon.

## Preparations

### Environment Preparation

Create the running environment with conda `23.9.0`, python `3.8.18`:

```
conda env create -f env.yaml
conda activate HCBOenv
```

### Dataset Preparation

High dimensional dataset initialization:

```
python reproduce.py
```

Real-world dataset initialization:
```
python initialize_real.py CoralGraph
python initialize_real.py HealthGraph
```

## Experiments

### CID Validation

Remember to create `./result/EffDim` folder before reproducing the result of CID validation experiments.

```
python run_eff_dim.py CoralGraph
python run_eff_dim.py HealthGraph
python run_eff_dim.py additive-50-124
python run_eff_dim.py additive-100-8
python run_eff_dim.py linear-100-124
python run_eff_dim.py non-additive-50-122
python run_eff_dim.py non-additive-100-124
```

```
python run_eff_dim_test.py linear-200-2
```

### Performance Experiments

Run baseline experiments in this form:
```
python run.py <problem_name> --run_performance
```

For example:
```
python run.py additive-100-8 --run_performance
```


### Ablation Study Experiment
```
python run_ablation.py additive-100-8
```

### Hyperparameter Experiment

```
python run_hyperparameter.py linear-100-124
```

## Visualization and statitical tests

Please refer to `result_analysis_baseline.ipynb` to visualize and conduct t-tests on baseline experiment results.

Please refer to `result_analysis_others.ipynb` to visualize the result of ablation study and hyper-parameter experiments.


