from .tree import tree
from .utils import most_common_label, bootstrap_sample, weight_update
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state



def bootstrap_sample(X, y, rng):
    """
    Creates a bootstrap sample of the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    y : ndarray of shape (n_samples,)
        The target labels.
    rng : numpy.random.Generator
        Random number generator for sampling.

    Returns
    -------
    X_samp : ndarray of shape (n_samples, n_valid_features)
        Bootstrap sample of the data with constant features removed.
    y_samp : ndarray of shape (n_samples,)
        Bootstrap sample of the labels.
    valid_features : list of int
        Indices of non-constant features in the sample.
    """
    n_samples = X.shape[0]

    # Sample rows with replacement using the random generator
    row_indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
    X_samp = X[row_indices]
    y_samp = y[row_indices]

    # Identify valid features (columns with more than one unique value)
    valid_features = [
        i for i in range(X.shape[1]) if len(np.unique(X_samp[:, i])) > 1
    ]
    X_samp = X_samp[:, valid_features]

    return X_samp, y_samp, valid_features


def most_common_label(y):
    """
    Finds the most common label in the data.

    Parameters
    ----------
    y : list or ndarray
        Array of labels.

    Returns
    -------
    most_common : int
        The most frequent label.
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class SPBaDF(BaseEstimator, ClassifierMixin):
    """
    Shape Penalty Bagging Decision Forest (SPBaDF)

    Implements a bagging ensemble using Shape Penalty Regularized Trees for
    imbalanced_spdf classification. Each tree is trained on a bootstrap sample
    and uses a subset of non-constant features.

    Parameters
    ----------
    n_trees : int, default=40
        Number of trees in the ensemble.

    weight : int, default=1
        Weight for the minority class, denoted as λ in the associated research paper.

    pen : int, default=0
        Regularization penalty for controlling the complexity of the decision boundary.

    maximal_leaves : int or float, default=None
        Maximum number of leaves allowed for each tree. If `None`, it defaults to
        `2 * np.sqrt(n_samples) * 0.3333` dynamically for each tree.

    random_state : int, default=23
        Random seed for reproducibility.

    Attributes
    ----------
    trees : list
        List of trained tree estimators.

    considered_features : list of lists
        List of feature subsets used for each tree.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels observed during training.
    """

    def __init__(self, n_trees=40, weight=1, pen=0, maximal_leaves=None, random_state=23):
        self.n_trees = n_trees
        self.weight = weight
        self.pen = pen
        self.maximal_leaves = maximal_leaves
        self.random_state = random_state
        self.trees = []
        self.considered_features = []

    def fit(self, X, y):
        """
        Fits the SPBaDF ensemble on the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for training.
        y : ndarray of shape (n_samples,)
            Target labels for training.

        Returns
        -------
        self : SPBaDF
            The fitted ensemble instance.
        """
        # Validate X and y
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Validate hyperparameters
        if not isinstance(self.n_trees, int) or self.n_trees <= 0:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise ValueError("weight must be a positive number.")
        if self.maximal_leaves is not None and (
                not isinstance(self.maximal_leaves, (int, float)) or self.maximal_leaves <= 0
        ):
            raise ValueError("maximal_leaves must be a positive number or None.")

        # Initialize attributes
        self.trees = []
        self.considered_features = []
        rng = check_random_state(self.random_state)  # Random generator

        for _ in range(self.n_trees):
            # Create a bootstrap sample and remove constant features
            X_samp, y_samp, valid_features = bootstrap_sample(X, y, rng)

            # Compute maximal_leaves dynamically if not explicitly set
            if self.maximal_leaves is None:
                maximal_leaves = int(2 * np.sqrt(X_samp.shape[0]) * 0.3333)
            else:
                maximal_leaves = self.maximal_leaves

            # Train a shape-penalized SVR tree
            tr_svr = tree()
            tr_svr.fit_sv(X_samp, y_samp, self.pen, weight=self.weight, maximal_leaves=maximal_leaves)

            self.trees.append(tr_svr)
            self.considered_features.append(valid_features)  # Store valid features for this tree

        return self

    def predict(self, X):
        """
        Predicts labels for input samples using the trained ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        # Check if the classifier is fitted
        check_is_fitted(self)

        # Validate X
        X = check_array(X)

        tree_preds = []

        for tree, valid_features in zip(self.trees, self.considered_features):
            # Use only valid features for this tree
            X_subset = X[:, valid_features]
            tree_preds.append(tree.predict(X_subset))

        # Aggregate predictions from all trees
        tree_preds = np.array(tree_preds).T  # Shape: (n_samples, n_trees)
        y_pred = np.array([most_common_label(preds) for preds in tree_preds])

        return y_pred


class SPBoDF(BaseEstimator, ClassifierMixin):
    """
    Shape Penalty Boosting Decision Forest (SPBoDF)

    This class implements a boosting ensemble method for imbalanced_spdf data using
    SVR (Surface-to-Volume Regularization) trees. The ensemble is trained to
    optimize decision boundaries, balancing interpretability with generalization
    by penalizing irregular decision surfaces.

    Parameters
    ----------
    n_trees : int, default=40
        The number of trees in the ensemble. Each tree represents one boosting round.

    weight : float, default=1
        Weight multiplier for the minority class to address class imbalance.
        Denoted by λ in the associated research paper.

    pen : float, default=0
        Regularization penalty for controlling the shape of the decision set,
        represented as α_n in the research paper.

    maximal_leaves : int or float, default=None
        Maximum number of leaves allowed for each tree. If `None`, defaults to
        `2 * np.sqrt(n_samples) * 0.3333` dynamically for each boosting round.

    random_state : int, default=23
        Random seed for reproducibility.

    Attributes
    ----------
    estimators_ : list
        List of fitted tree estimators in the ensemble.

    estimator_weights_ : list
        List of weights associated with each tree in the ensemble.

    columns_to_take_ : list
        List of feature subsets selected for each tree, useful for analyzing feature importance.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels observed during training.
    """

    def __init__(self, n_trees=40, weight=1, pen=0, maximal_leaves=None, random_state=23):
        self.n_trees = n_trees
        self.weight = weight
        self.pen = pen
        self.maximal_leaves = maximal_leaves
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train the SPBoDF ensemble on the given dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for training.

        y : ndarray of shape (n_samples,)
            Target labels for training.

        Returns
        -------
        self : SPBoDF
            Returns the instance of the fitted ensemble.
        """
        # Validate X and y
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Validate hyperparameters
        if not isinstance(self.n_trees, int) or self.n_trees <= 0:
            raise ValueError("n_trees must be a positive integer.")
        if not isinstance(self.weight, (int, float)) or self.weight <= 0:
            raise ValueError("weight must be a positive number.")
        if self.maximal_leaves is not None and (
                not isinstance(self.maximal_leaves, (int, float)) or self.maximal_leaves <= 0
        ):
            raise ValueError("maximal_leaves must be a positive number or None.")

        # Initialize attributes
        self.estimators_ = []
        self.estimator_weights_ = []
        self.columns_to_take_ = []
        rng = check_random_state(self.random_state)  # Random generator for reproducibility

        # Initialize sample weights
        n_samples = X.shape[0]
        sample_weight = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_trees):
            # Boost one SVR tree
            sample_weight, estimator_weight, _, selected_columns, estimator = self._boost_svr(
                i, X, y, sample_weight, rng
            )

            # Update ensemble if boosting round is successful
            if sample_weight is None:
                break
            self.estimator_weights_.append(estimator_weight)
            self.columns_to_take_.append(selected_columns)
            self.estimators_.append(estimator)
            sample_weight /= np.sum(sample_weight)

        # warning if the number of trees is less than n_trees
        if len(self.estimators_) < self.n_trees:
            print(f"Warning: Only {len(self.estimators_)} trees were trained, because the error rate was 0 or 1. Try with different random seed or bigger dataset.")

        return self

    def predict(self, X):
        """
        Predict labels for input samples using the trained ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels.
        """
        # Check if the classifier is fitted
        check_is_fitted(self, ["estimators_", "estimator_weights_", "columns_to_take_"])

        if len(self.estimators_) == 0:
            raise ValueError("No trees were trained. Please fit the model first. If already fitted, try with different random seed or bigger dataset.")

        # Validate X
        X = check_array(X)

        prediction = np.zeros(X.shape[0])

        for alpha, tree, cols in zip(self.estimator_weights_, self.estimators_, self.columns_to_take_):
            # Use only the relevant feature subset for this tree
            prediction += alpha * (2 * tree.predict(X[:, cols]) - 1)

        return (np.sign(prediction) + 1) / 2

    def _boost_svr(self, iboost, X, y, sample_weight, rng):
        """
        Perform one boosting step using an SVR tree.

        Parameters
        ----------
        iboost : int
            Boosting round index.

        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Training labels.

        sample_weight : ndarray of shape (n_samples,)
            Sample weights for the current boosting round.

        rng : numpy.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        sample_weight : ndarray
            Updated sample weights.

        estimator_weight : float
            Weight of the trained estimator.

        estimator_error : float
            Error rate of the trained estimator.

        selected_columns : list
            List of feature indices selected during this round.
        """
        # Create a new tree estimator
        estimator = tree()
        # print(f"iboost: {iboost}")

        # Bootstrap sampling with replacement
        row_indices = rng.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True, p=sample_weight)
        X_sampled, y_sampled = X[row_indices], y[row_indices]

        # Identify valid features (non-constant)
        valid_features = [
            i for i in range(X_sampled.shape[1]) if len(np.unique(X_sampled[:, i])) > 1
        ]
        X_sampled = X_sampled[:, valid_features]

        # Compute maximal_leaves dynamically if not explicitly set
        if self.maximal_leaves is None:
            maximal_leaves = int(2 * np.sqrt(len(X_sampled)) * 0.3333)
        else:
            maximal_leaves = self.maximal_leaves

        # Train the SVR tree
        estimator.fit_sv(
            X_sampled, y_sampled, self.pen, weight=self.weight, maximal_leaves=maximal_leaves
        )

        # Predict and calculate error
        y_pred = estimator.predict(X[:, valid_features])
        incorrect = y_pred != y
        estimator_error = np.average(incorrect, weights=sample_weight)

        # Handle edge cases for error
        if estimator_error == 0 or estimator_error >= 1.0 - (1.0 / len(self.classes_)):
            return None, None, None, None, None

        # Compute weight of the estimator
        estimator_weight = 0.5 * np.log((1.0 - estimator_error) / estimator_error)

        # Update sample weights
        sample_weight *= np.exp(estimator_weight * (2 * incorrect - 1)) * (sample_weight > 0)

        return sample_weight, estimator_weight, estimator_error, valid_features, estimator


