# MIRACLE (Missing data Imputation Refinement And Causal LEarning)

[![Tests](https://github.com/vanderschaarlab/MIRACLE/actions/workflows/test_miracle.yml/badge.svg)](https://github.com/vanderschaarlab/MIRACLE/actions/workflows/test_miracle.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/vanderschaarlab/MIRACLE/blob/main/LICENSE)

This is example code for running the MIRACLE algorithm. This code has been inspired by the work of [1,2,3,4].

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Tests
You can run the tests using
```bash
pip install -r requirements_dev.txt
pip install .
pytest -vsx
```

## Contents

- `miracle/MIRACLE.py` - Imputer/Refiner Class. This class takes a baseline imputation and returns a refined imputation. This code has been forked from [2].
- `miracle/third_party` - Reference imputers: Mean, Missforest, MICE, GAIN, Sinkhorn, KNN.
- `tests/run_example.py` - runs a nonlinear toy DAG example.  Uses mean imputation as a baseline and applies MIRACLE to refine.  

## Examples


Base example on toy dag.
```bash
$ python run_example.py
```

This specific instantiation returns a Baseline RMSE of approximately 0.95 with MIRACLE RMSE of approximately 0.40.

An example to run toy example with a dataset size of 2000 for 300 max_steps with a missingness of 30%
```bash
$ python3 run_example.py --dataset_sz 2000 --max_steps 300 --missingness 0.3
```

## References

[1] Jinsung Yoon, James Jordon, and Mihaela van der Schaar. Gain: Missing data imputation using generative adversarial nets. In ICML, 2018.

[2] Trent Kyono, Yao Zhang, and Mihaela van der Schaar. CASTLE: Regularization via auxiliary causal graph discovery. In NeurIPS, 2020.

[3] Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning (NeurIPS 2018).

[4] Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. P. (2020). Learning sparse nonparametric DAGs (AISTATS 2020). 
