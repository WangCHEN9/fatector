# fatector (fake detector)

[![pytest](https://github.com/WangCHEN9/create_py_project/actions/workflows/pytest.yml/badge.svg)](https://github.com/WangCHEN9/create_py_project/actions/workflows/pytest.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

```
pip install --editable .
```

## Config

Config of this project are build with [hydra](https://hydra.cc/docs/intro/#:~:text=Hydra%20is%20an%20open%2Dsource,files%20and%20the%20command%20line.)

Config can be modified in ./config.yaml

## Machine learning

This project runs autoML thanks to [pycaret](https://pycaret.gitbook.io/docs/)
Runs are saved with [mlflow](https://mlflow.org/docs/latest/index.html) in ./mlruns

To check history of runs :

```
mlflow ui
```

## How to use

fatector.Fatector is the main entry point

```python
    fatector = Fatector()
    fatector.preprocess_train_test_data()  #! only need run once
    fatector.training()  #! this will run automl, and save all runs to /mlruns
    fatector.inference()  #! this runs inference
```
