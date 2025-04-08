import warnings
from itertools import chain, pairwise
from numbers import Integral, Real

import numpy as np
import scipy.optimize

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
)
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
    _safe_indexing,
    check_random_state,
    column_or_1d,
    gen_batches,
    shuffle,
)
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
    _check_partial_fit_first_call,
    type_of_target,
    unique_labels,
)
from ..utils.optimize import _check_optimize_result
from ..utils.validation import _check_sample_weight, check_is_fitted, validate_data
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer

_STOCHASTIC_SOLVERS = ["sgd", "adam"]


def _pack(coefs_, intercepts_):
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


class MultilayerPerceptron(BaseEstimator):

    _parameter_constraints: dict = {
        "estimator_type": [StrOptions({"classifier", "regressor"})],
        "hidden_layer_sizes": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "activation": [StrOptions({"identity", "logistic", "tanh", "relu"})],
        "solver": [StrOptions({"lbfgs", "sgd", "adam"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "batch_size": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "learning_rate": [StrOptions({"constant", "invscaling", "adaptive"})],
        "learning_rate_init": [Interval(Real, 0, None, closed="neither")],
        "power_t": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "loss": [StrOptions({"log_loss", "squared_error", "poisson"})],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "momentum": [Interval(Real, 0, 1, closed="both")],
        "nesterovs_momentum": ["boolean"],
        "early_stopping": ["boolean"],
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],
        "beta_1": [Interval(Real, 0, 1, closed="left")],
        "beta_2": [Interval(Real, 0, 1, closed="left")],
        "epsilon": [Interval(Real, 0, None, closed="neither")],
        "n_iter_no_change": [
            Interval(Integral, 1, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "max_fun": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        estimator_type="classifier",
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        loss=None,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        self.estimator_type = estimator_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

        if loss is None:
            if self.estimator_type == "classifier":
                self.loss = "log_loss"
            else:
                self.loss = "squared_error"
        else:
            self.loss = loss

        if self.estimator_type == "classifier" and self.loss != "log_loss":
            raise ValueError(
                f"The loss '{self.loss}' is not supported for classification. "
                "Use 'log_loss' for classification."
            )
        elif self.estimator_type == "regressor" and self.loss not in ["squared_error", "poisson"]:
            raise ValueError(
                f"The loss '{self.loss}' is not supported for regression. "
                "Use 'squared_error' or 'poisson' for regression."
            )


    def _validate_input(self, X, y, incremental, reset):

        if self.estimator_type == "classifier":
            X, y = validate_data(
                self,
                X,
                y,
                accept_sparse=["csr", "csc"],
                multi_output=True,
                dtype=(np.float64, np.float32),
                reset=reset,
            )
        else:
            X, y = validate_data(
                self,
                X,
                y,
                accept_sparse=["csr", "csc"],
                multi_output=True,
                y_numeric=True,
                dtype=(np.float64, np.float32),
                reset=reset,
            )

        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        if self.estimator_type == "classifier":

            if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
                self._label_binarizer = LabelBinarizer()
                self._label_binarizer.fit(y)
                self.classes_ = self._label_binarizer.classes_
            else:
                classes = unique_labels(y)
                if self.warm_start:
                    if set(classes) != set(self.classes_):
                        raise ValueError(
                            "warm_start can only be used where `y` has the same "
                            "classes as in the previous call to fit. Previously "
                            f"got {self.classes_}, `y` has {classes}"
                        )
                elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                    raise ValueError(
                        "`y` has classes not in `self.classes_`. "
                        f"`self.classes_` has {self.classes_}. 'y' has {classes}."
                    )

            y = self._label_binarizer.transform(y).astype(bool)

        return X, y

    def _unpack(self, packed_parameters):
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def predict(self, X):
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        y_pred = self._forward_pass_fast(X, check_input=check_input)

        if self.estimator_type == "classifier":
            if self.n_outputs_ == 1:
                y_pred = y_pred.ravel()
            return self._label_binarizer.inverse_transform(y_pred)
        else:  # regressor
            if y_pred.shape[1] == 1:
                return y_pred.ravel()
            return y_pred

    def _score(self, X, y, sample_weight=None):
        if self.estimator_type == "classifier":
            return self._score_with_function(
                X, y, sample_weight=sample_weight, score_function=accuracy_score
            )
        else:  # regressor
            return self._score_with_function(
                X, y, sample_weight=sample_weight, score_function=r2_score
            )

    @available_if(lambda est: est._check_solver())
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None, classes=None):

        if self.estimator_type == "classifier":
            if classes is not None:
                if _check_partial_fit_first_call(self, classes):
                    self._label_binarizer = LabelBinarizer()
                    if type_of_target(y).startswith("multilabel"):
                        self._label_binarizer.fit(y)
                    else:
                        self._label_binarizer.fit(classes)
            elif not hasattr(self, "_label_binarizer"):
                raise ValueError("classes must be passed on the first call to partial_fit.")

        return self._fit(X, y, sample_weight=sample_weight, incremental=True)

    def predict_log_proba(self, X):

        if self.estimator_type != "classifier":
            raise AttributeError(
                "predict_log_proba is not available when estimator_type='regressor'."
            )

        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):

        if self.estimator_type != "classifier":
            raise AttributeError(
                "predict_proba is not available when estimator_type='regressor'."
            )

        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True

        if self.estimator_type == "classifier":
            tags.classifier_tags.multi_label = True

        return tags

    def _more_tags(self):
        return {"_skip_test": True}

    def _is_classifier(self):
        return self.estimator_type == "classifier"

    def _forward_pass(self, activations):

        hidden_activation = ACTIVATIONS[self.activation]

        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]


            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])

        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])

        return activations

    def _forward_pass_fast(self, X, check_input=True):

        if check_input:
            X = validate_data(self, X, accept_sparse=["csr", "csc"], reset=False)

        activation = X

        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def _compute_loss_grad(
        self, layer, sw_sum, activations, deltas, coef_grads, intercept_grads
    ):

        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= sw_sum

        intercept_grads[layer] = np.sum(deltas[layer], axis=0) / sw_sum

    def _loss_grad_lbfgs(
        self,
        packed_coef_inter,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    ):

        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, sample_weight, activations, deltas, coef_grads, intercept_grads
        )
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad

    def _backprop(
        self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads
    ):

        n_samples = X.shape[0]

        activations = self._forward_pass(activations)

        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1], sample_weight)

        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        if sample_weight is None:
            sw_sum = n_samples
        else:
            sw_sum = sample_weight.sum()
        loss += (0.5 * self.alpha) * values / sw_sum


        last = self.n_layers_ - 2

        deltas[last] = activations[-1] - y
        if sample_weight is not None:
            deltas[last] *= sample_weight.reshape(-1, 1)

        self._compute_loss_grad(
            last, sw_sum, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]

        for i in range(last, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, sw_sum, activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads

    def _initialize(self, y, layer_units, dtype):

        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]
        self.n_layers_ = len(layer_units)

        if hasattr(self, 'estimator_type') and self.estimator_type == "regressor":
            if self.loss == "poisson":
                self.out_activation_ = "exp"
            else:
                self.out_activation_ = "identity"
        elif self._label_binarizer.y_type_ == "multiclass":
            self.out_activation_ = "softmax"
        else:
            self.out_activation_ = "logistic"

        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self._best_coefs = [c.copy() for c in self.coefs_]
        self._best_intercepts = [i.copy() for i in self.intercepts_]

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf
                self.validation_scores_ = None
                self.best_validation_score_ = None

    def _init_coef(self, fan_in, fan_out, dtype):
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init

    def _fit(self, X, y, sample_weight=None, incremental=False):
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)
        n_samples, n_features = X.shape
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            self._initialize(y, layer_units, X.dtype)

        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in pairwise(layer_units)
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
            )

        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

    def _fit_lbfgs(
        self,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
    ):
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        opt_res = scipy.optimize.minimize(
            self._loss_grad_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": self.max_fun,
                "maxiter": self.max_iter,
                "iprint": iprint,
                "gtol": self.tol,
            },
            args=(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
            ),
        )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

    def _fit_stochastic(
        self,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
    ):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        if self.early_stopping and incremental:
            raise ValueError("partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = (hasattr(self, 'estimator_type') and self.estimator_type == "classifier") and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            if sample_weight is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
                sample_weight_train = sample_weight_val = None
            else:
                # TODO: incorporate sample_weight in sampling here.
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sample_weight_train,
                    sample_weight_val,
                ) = train_test_split(
                    X,
                    y,
                    sample_weight,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        n_samples = X_train.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    sample_idx = shuffle(sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        batch_idx = sample_idx[batch_slice]
                        X_batch = _safe_indexing(X_train, batch_idx)
                    else:
                        batch_idx = batch_slice
                        X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    if sample_weight is None:
                        sample_weight_batch = None
                    else:
                        sample_weight_batch = sample_weight_train[batch_idx]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch,
                        y_batch,
                        sample_weight_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X_train.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))

                self._update_no_improvement_count(
                    early_stopping, X_val, y_val, sample_weight_val
                )

                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter,
                        ConvergenceWarning,
                    )
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
        if early_stopping:
            val_score = self._score(X, y, sample_weight=sample_weight)

            self.validation_scores_.append(val_score)

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ + self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy() for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, sample_weight=sample_weight, incremental=False)

    def _check_solver(self):
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError(
                "partial_fit is only available for stochastic"
                " optimizers. %s is not stochastic." % self.solver
            )
        return True

    def _score_with_function(self, X, y, sample_weight, score_function):
        y_pred = self._predict(X, check_input=False)

        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return np.nan

        return score_function(y, y_pred, sample_weight=sample_weight)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss="log_loss",
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _validate_input(self, X, y, incremental, reset):
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            classes = unique_labels(y)
            if self.warm_start:
                if set(classes) != set(self.classes_):
                    raise ValueError(
                        "warm_start can only be used where `y` has the same "
                        "classes as in the previous call to fit. Previously "
                        f"got {self.classes_}, `y` has {classes}"
                    )
            elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError(
                    "`y` has classes not in `self.classes_`. "
                    f"`self.classes_` has {self.classes_}. 'y' has {classes}."
                )

        y = self._label_binarizer.transform(y).astype(bool)
        return X, y

    def predict(self, X):

        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        y_pred = self._forward_pass_fast(X, check_input=check_input)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred)

    def _score(self, X, y, sample_weight=None):
        return super()._score_with_function(
            X, y, sample_weight=sample_weight, score_function=accuracy_score
        )

    @available_if(lambda est: est._check_solver())
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None, classes=None):

        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith("multilabel"):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        return self._fit(X, y, sample_weight=sample_weight, incremental=True)

    def predict_log_proba(self, X):

        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):

        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags


class MLPRegressor(RegressorMixin, BaseMultilayerPerceptron):

    _parameter_constraints: dict = {
        **BaseMultilayerPerceptron._parameter_constraints,
        "loss": [StrOptions({"squared_error", "poisson"})],
    }

    def __init__(
        self,
        loss="squared_error",
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss=loss,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def predict(self, X):

        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        """Private predict method with optional input validation"""
        y_pred = self._forward_pass_fast(X, check_input=check_input)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _score(self, X, y, sample_weight=None):
        return super()._score_with_function(
            X, y, sample_weight=sample_weight, score_function=r2_score
        )

    def _validate_input(self, X, y, incremental, reset):
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            y_numeric=True,
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    @available_if(lambda est: est._check_solver)
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None):

        return self._fit(X, y, sample_weight=sample_weight, incremental=True)
