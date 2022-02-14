fatector (fake detector)
========================

|pytest| |Code style: black|

Installation
------------

::

   git clone https://github.com/WangCHEN9/fatector.git

::

   pip install --editable .

Config
------

Config of this project are build with
`hydra <https://hydra.cc/docs/intro/#:~:text=Hydra%20is%20an%20open%2Dsource,files%20and%20the%20command%20line.>`__

Config can be modified in **./config.yaml**

Machine learning
----------------

This project runs autoML thanks to
`pycaret <https://pycaret.gitbook.io/docs/>`__

Runs are saved with
`mlflow <https://mlflow.org/docs/latest/index.html>`__ in **./mlruns**

To check history of runs :

::

   mlflow ui

How to use
----------

fatector.Fatector is the main entry point

.. code:: python

   fatector = Fatector()
   fatector.preprocess_train_test_data()  #! only need run once, preprocess train test data
   fatector.training()  #! this will run automl based on processed data, and save all runs to ./mlruns
   fatector.inference()  #! this will run inference on processed test data

.. |pytest| image:: https://github.com/WangCHEN9/create_py_project/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/WangCHEN9/create_py_project/actions/workflows/pytest.yml
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
