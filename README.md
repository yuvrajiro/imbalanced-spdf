![Feature - Imbalanced Datasets](https://img.shields.io/badge/feature-imbalanced%20datasets-green?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/yuvrajiro/imbalanced-spdf?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/yuvrajiro/imbalanced-spdf?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/yuvrajiro/imbalanced-spdf?style=flat-square)
![Python Version](https://img.shields.io/badge/python-%203.9-blue?style=flat-square)
![Read the Docs](https://img.shields.io/readthedocs/imbalanced-spdf?style=flat-square)

> A  rich documentation is available at [Read the Docs](https://imbalanced-spdf.readthedocs.io/en/latest/)


# Shape Penalized Decision Forests

Shape Penalized  Decision Forests, for training ensemble classifiers tailored for imbalanced datasets. This package provides two primary implementations:

- **SPBoDF (Shape Penalized  Boosting Decision Forest)**: A boosting ensemble method that builds multiple trees sequentially, adjusting the weights of samples to focus on harder-to-classify instances.

- **SPBaDF (Shape Penalized  Bagging Decision Forest)**: A bagging ensemble method that builds multiple trees independently on bootstrap samples of the dataset, improving robustness and reducing overfitting.

Both implementations use the concept of **Surface-to-Volume Regularization (SVR)** to penalize irregular decision boundaries, thus improving generalization and addressing challenges associated with imbalanced datasets.

## Key Features

- **Boosting and Bagging**: Two ensemble approaches tailored for classification tasks.
- **Shape Penalization**: Incorporates a novel regularization technique to control decision boundary complexity.
- **Imbalanced Data Handling**: Designed with class imbalance in mind, using weighting and bootstrapping techniques.
- **Scikit-learn Compatible**: Implements `BaseEstimator` and `ClassifierMixin`, making it seamlessly integrable with the Scikit-learn ecosystem.
- **Customizability**: Hyperparameters such as the number of trees, shape penalty, and maximal leaves are configurable for fine-tuning.

---

## Installation

1. Downloading Locally and Installing

   ```bash
   git clone https://www.github.com/yuvrajiro/imbalanced-spdf.git
   cd imbalance-svr
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:

   ```bash
   python install -e .
    ```
2. Using pip install from GitHub

   ```bash
   pip install git+https://www.github.com/yuvrajiro/imbalanced-spdf.git
   ```
   
> This arrangement is temporary, and package will be available on PyPI soon.

---

## Usage

### 1. SPBoDF (Boosting)

#### Example

```python
import numpy as np
from imbalanced_spdf.ensemble import SPBoDF

# Generate synthetic data
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)

# Initialize and fit SPBoDF
boosting_model = SPBoDF(n_trees=50, weight=2, pen=1.0, random_state=42)
boosting_model.fit(X_train, y_train)

# Predict
X_test = np.random.rand(20, 5)
y_pred = boosting_model.predict(X_test)
print("Predictions:", y_pred)
```

### 2. SPBaDF (Bagging)

#### Example

```python
from imbalanced_spdf.ensemble import SPBaDF

# Generate synthetic data
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, 100)

# Initialize and fit SPBaDF
bagging_model = SPBaDF(n_trees=50, weight=2, pen=1.0, random_state=42)
bagging_model.fit(X_train, y_train)

# Predict
X_test = np.random.rand(20, 5)
y_pred = bagging_model.predict(X_test)
print("Predictions:", y_pred)
```

---

## API Reference

### `SPBoDF`

A boosting ensemble classifier that uses SVR-regularized trees to handle imbalanced datasets.

#### Parameters:

- `n_trees` (int): Number of trees in the ensemble (default: 40).
- `weight` (float): Weight for the minority class to address imbalance (default: 1).
- `pen` (float): Regularization penalty controlling decision boundary complexity (default: 0).
- `maximal_leaves` (int or float, optional): Maximum leaves per tree. Defaults to `2 * sqrt(n_samples) * 0.3333`.
- `random_state` (int): Random seed for reproducibility (default: 23).

#### Methods:

- `fit(X, y)`: Fits the ensemble on the training data.
- `predict(X)`: Predicts the labels of the test data.

### `SPBaDF`

A bagging ensemble classifier that uses SVR-regularized trees to improve robustness.

#### Parameters:

- `n_trees` (int): Number of trees in the ensemble (default: 40).
- `weight` (float): Weight for the minority class to address imbalance (default: 1).
- `pen` (float): Regularization penalty controlling decision boundary complexity (default: 0).
- `maximal_leaves` (int or float, optional): Maximum leaves per tree. Defaults to `2 * sqrt(n_samples) * 0.3333`.
- `random_state` (int): Random seed for reproducibility (default: 23).

#### Methods:

- `fit(X, y)`: Fits the ensemble on the training data.
- `predict(X)`: Predicts the labels of the test data.

---

## How It Works

1. **SPBoDF (Boosting)**:
   - Trees are built sequentially, with sample weights updated after each iteration to focus on misclassified samples.
   - Regularization (SVR) penalizes irregular decision boundaries to avoid overfitting.

2. **SPBaDF (Bagging)**:
   - Trees are built independently on bootstrap samples of the training data.
   - Each tree focuses on non-constant feature subsets, improving robustness and generalization.


## Dataset Details


| Dataset             | Available at | Comments (if any)                                                      |
|---------------------| --- |------------------------------------------------------------------------|
| Appendicitis        | https://github.com/ZixiaoShen/Datasets/blob/master/UCI/C2_F7_S106_Appendicitis/Appendicitis.csv | -----------                                                            |
| Data User Modelling | http://archive.ics.uci.edu/ml/machine-learning-databases/00257/Data_User_Modeling_Dataset_Hamdi%20Tolga%20KAHRAMAN.xls | -------------                                                          |
| Ecoli | https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv | 'pp' is considered as class 1 and cp, im, om, omL, imL, imU as class 0 |  ------------------ |
| Ecoli-0-6-7-vs-5 | https://github.com/w4k2/umce/blob/master/datasets/imb_IRhigherThan9p2/ecoli-0-6-7_vs_5/ecoli-0-6-7_vs_5.dat | ------                                                                 |
| Estate | https://github.com/MKLab-ITI/Posterior-Rebalancing/blob/1a0b561e6418e9df25a75006206598bff2babe2c/data/hddt/imbalanced/estate.data#L4 | ------                                                                 |
| Fertility Diagonosis | https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt | ------                                                                 |
| Imbalance-scale | https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data | 'B' is conisdered as 1, otherwise 0                                    |
| Oil | https://github.com/MKLab-ITI/Posterior-Rebalancing/blob/1a0b561e6418e9df25a75006206598bff2babe2c/data/hddt/imbalanced/oil.data | ------                                                                 |
| Page-blocks0 | https://github.com/w4k2/DSE/blob/ac0e824d3a7507fe9d57356150cef0def5c4a36d/streams/real/page-blocks0.arff#L4 | ------                                                                 |
| Winequality-red | https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv | -------                                                                |
| Yeast-0-3-5-9-vs-7-8 | https://github.com/w4k2/DSE/blob/ac0e824d3a7507fe9d57356150cef0def5c4a36d/streams/real/yeast-0-3-5-9_vs_7-8.arff | ------                                                                 |
| car-vgood | https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data | 0 if 'negative' otherwise 1                                            |
| cleveland_0_vs_4 | https://github.com/Jaga7/Metody-Sztucznej-Inteligencji/blob/d46ae0c897b5524d5e0c9b9b800e190b9727fd52/PROJEKT/datasets/cleveland_0_vs_4.csv#L4 | ------                                                                 |
| haberman | https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data |    |
| led7digit-0-2-4-5-6-7-8-9_vs_1 | https://github.com/ikurek/PWr-Uczenie-Maszyn/blob/f9561a959c49229f22e489b17ccb23b52e99d2a2/data/led7digit-0-2-4-5-6-7-8-9_vs_1.dat#L4 | ------                                                                 |
| new-thyroid1 | https://github.com/jamesrobertlloyd/dataset-space/blob/d195fd8748ba8def627ae2e727395aee608952ec/data/class/raw/keel/new-thyroid1.dat#L4 | ------                                                                 |
| page-blocks-1-3_vs_4 | https://github.com/ikurek/PWr-Uczenie-Maszyn/blob/f9561a959c49229f22e489b17ccb23b52e99d2a2/data/page-blocks-1-3_vs_4.dat | ------                                                                 |
| shuttle-c0-vs-c4 | https://github.com/ikurek/PWr-Uczenie-Maszyn/blob/f9561a959c49229f22e489b17ccb23b52e99d2a2/data/shuttle-c0-vs-c4.dat | ------                                                                 |
|vehicle3 | https://github.com/jamesrobertlloyd/dataset-space/blob/d195fd8748ba8def627ae2e727395aee608952ec/data/class/raw/keel/vehicle3.dat | ------                                                                 |
| yeast-2_vs_8 | https://github.com/jamesrobertlloyd/dataset-space/blob/d195fd8748ba8def627ae2e727395aee608952ec/data/class/raw/keel/yeast-2_vs_8.dat | ------                                                                 |


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## References

- Zhu, Y., Li, C., & Dunson, D. B. (2023). "Classification Trees for Imbalanced Data: Surface-to-Volume Regularization." Journal of the American Statistical Association.



