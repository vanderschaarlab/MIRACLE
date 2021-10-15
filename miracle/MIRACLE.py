import random
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.metrics import mean_squared_error

import miracle.logger as log

np.set_printoptions(suppress=True)
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
np.set_printoptions(linewidth=np.inf)


class MIRACLE(object):
    def __init__(
        self,
        lr: float = 0.001,
        batch_size: int = 32,
        num_inputs: int = 1,
        num_outputs: int = 1,
        n_hidden: int = 32,
        ckpt_file: str = "tmp.ckpt",
        reg_lambda: float = 1,
        reg_beta: float = 1,
        DAG_only: bool = False,
        missing_list: list = list(range(1)),
        reg_m: float = 1.0,
        window: int = 10,
        max_steps: int = 400,
    ):
        n_indicators = len(missing_list)
        log.info(f"Number of Indicators = {n_indicators}")
        """
        Assume n_indicators is the number of variables at the RHS of adjacency matrix.
        """
        self.missing_list = missing_list
        self.learning_rate = lr
        self.reg_lambda = reg_lambda
        self.reg_beta = reg_beta
        self.reg_m = reg_m
        self.batch_size = batch_size
        self.num_inputs = num_inputs + num_inputs  # input + indicator
        self.n_hidden = n_hidden
        self.num_outputs = num_outputs
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.X_mask = tf.placeholder("float", [None, self.num_inputs])
        self.rho = tf.placeholder("float", [1, 1])
        self.alpha = tf.placeholder("float", [1, 1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.window = window
        self.max_steps = max_steps
        self.metric = mean_squared_error
        self.missing_value: list = []
        self.count = 0

        # One-hot vector indicating which nodes are trained
        self.sample = tf.placeholder(tf.int32, [self.num_inputs])

        # Store layers weight & bias
        seed = 1
        self.weights = {}
        self.biases = {}

        # Create the input and output weight matrix for each feature
        for i in range(self.num_inputs):
            self.weights["w_h0_" + str(i)] = tf.Variable(
                tf.random_normal([self.num_inputs, self.n_hidden], seed=seed) * 0.01
            )
            self.weights["out_" + str(i)] = tf.Variable(
                tf.random_normal([self.n_hidden, self.num_outputs], seed=seed)
            )

            # self.weights['w_h0_' + str(i)][:,-n_indicators] *= 0.0
        for i in range(self.num_inputs):
            self.biases["b_h0_" + str(i)] = tf.Variable(
                tf.random_normal([self.n_hidden], seed=seed) * 0.01
            )
            self.biases["out_" + str(i)] = tf.Variable(
                tf.random_normal([self.num_outputs], seed=seed)
            )

        # reg_lambda The first and second layers are shared
        self.weights.update(
            {
                "w_h1": tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden])),
                "w_h2": tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden])),
            }
        )

        self.biases.update(
            {
                "b_h1": tf.Variable(tf.random_normal([self.n_hidden])),
                "b_h2": tf.Variable(tf.random_normal([self.n_hidden])),
            }
        )

        self.hidden_h0: dict = {}
        self.hidden_h1: dict = {}
        self.hidden_h2: dict = {}

        self.layer_1: dict = {}
        self.layer_1_dropout: dict = {}
        self.out_layer: dict = {}

        self.Out_0: list = []

        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.elu

        indices = [self.num_inputs - n_indicators] * self.n_hidden
        indicator_mask = tf.transpose(
            tf.one_hot(
                indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1
            )
        )
        for i in range(self.num_inputs - n_indicators + 1, self.num_inputs):
            indices = [i] * self.n_hidden
            indicator_mask *= tf.transpose(
                tf.one_hot(
                    indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1
                )
            )

        for i in range(self.num_inputs):
            indices = [i] * self.n_hidden
            self.mask[str(i)] = (
                tf.transpose(
                    tf.one_hot(
                        indices,
                        depth=self.num_inputs,
                        on_value=0.0,
                        off_value=1.0,
                        axis=-1,
                    )
                )
                * indicator_mask
            )

            self.weights["w_h0_" + str(i)] = (
                self.weights["w_h0_" + str(i)] * self.mask[str(i)]
            )
            self.hidden_h0["nn_" + str(i)] = self.activation(
                tf.add(
                    tf.matmul(self.X, self.weights["w_h0_" + str(i)]),
                    self.biases["b_h0_" + str(i)],
                )
            )
            self.hidden_h1["nn_" + str(i)] = self.activation(
                tf.add(
                    tf.matmul(self.hidden_h0["nn_" + str(i)], self.weights["w_h1"]),
                    self.biases["b_h1"],
                )
            )
            self.out_layer["nn_" + str(i)] = (
                tf.matmul(self.hidden_h1["nn_" + str(i)], self.weights["out_" + str(i)])
                + self.biases["out_" + str(i)]
            )
            self.Out_0.append(self.out_layer["nn_" + str(i)])

        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0, axis=1)
        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Residuals
        self.R = self.X - self.Out
        self.R_proxy = []
        self.R_proxy = tf.square(
            tf.multiply(self.X[:, :-n_indicators], self.X_mask[:, :-n_indicators])
            - tf.multiply(self.Out[:, :-n_indicators], self.X_mask[:, :-n_indicators])
        )
        self.supervised_loss = tf.reduce_sum(self.R_proxy)
        self.reg_moment = 0
        self.X_replace: list = []

        for dim in range(n_indicators):
            index = missing_list[dim]
            prob = (
                tf.nn.sigmoid(self.Out[:, self.num_inputs - n_indicators + dim]) + 1.01
            ) / 1.02
            weight = 1.0 / (prob)

            a = tf.reduce_mean(self.Out[:, index])  # * self.X_mask[:, index])/num_obs
            b = tf.reduce_sum(
                self.X[:, index] * self.X_mask[:, index] * weight
            ) / tf.reduce_sum(weight)
            self.reg_moment += tf.square(a - b)

        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.X[:, -n_indicators:], logits=self.Out[:, -n_indicators:]
        )
        self.R_proxy = tf.concat([self.R_proxy, bce_loss], axis=1)

        self.regularization_loss = 0

        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(
                tf.math.sqrt(
                    tf.reduce_sum(
                        tf.square(self.weights["w_h0_" + str(i)]), axis=1, keepdims=True
                    )
                )
            )

        self.W = tf.concat(self.W_0, axis=1)

        # truncated power series
        d = tf.cast(self.X.shape[1], tf.float32)
        coff = 1.0
        Z = tf.multiply(self.W, self.W)

        dag_l = tf.cast(d, tf.float32)

        Z_in = tf.eye(d)
        for i in range(1, 25):

            Z_in = tf.matmul(Z_in, Z)

            dag_l += 1.0 / coff * tf.linalg.trace(Z_in)
            coff = coff * (i + 1)

        self.h = dag_l - tf.cast(d, tf.float32)

        # group lasso
        L1_loss = 0.0
        for i in range(self.num_inputs):
            if i < self.num_inputs - n_indicators:
                w_1 = tf.slice(self.weights["w_h0_" + str(i)], [0, 0], [i, -1])
                w_2 = tf.slice(
                    self.weights["w_h0_" + str(i)],
                    [i + 1, 0],
                    [self.num_inputs - n_indicators - (i + 1), -1],
                )
                L1_loss += tf.reduce_sum(tf.norm(w_1, axis=1)) + tf.reduce_sum(
                    tf.norm(w_2, axis=1)
                )
            else:
                w_1 = tf.slice(
                    self.weights["w_h0_" + str(i)],
                    [0, 0],
                    [self.num_inputs - n_indicators, -1],
                )
                L1_loss += tf.reduce_sum(tf.norm(w_1, axis=1))

        # Divide the residual into untrain and train subset
        _, subset_R = tf.dynamic_partition(
            tf.transpose(self.R_proxy), partitions=self.sample, num_partitions=2
        )
        subset_R = tf.transpose(subset_R)

        self.mse_loss_subset = tf.reduce_mean(tf.reduce_sum(subset_R, axis=1))

        self.regularization_loss_subset = (
            self.mse_loss_subset
            + self.reg_beta * L1_loss
            + 0.5 * self.rho * self.h * self.h
            + self.alpha * self.h
            + self.reg_m * self.reg_moment
        )

        if DAG_only:
            self.loss_op_dag = self.optimizer_subset.minimize(
                self.regularization_loss_subset
            )
        else:
            self.loss_op_dag = self.optimizer_subset.minimize(
                self.regularization_loss_subset
                + self.Lambda * self.rho * self.supervised_loss
            )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.tmp = ckpt_file

    def __del__(self) -> None:
        try:
            tf.reset_default_graph()
            log.debug("Destructor Called... Cleaning up")
            self.sess.close()
            del self.sess
        except BaseException:
            pass

    def _fit(
        self,
        X_missing: np.ndarray,
        X_mask: np.ndarray,
        X_seed: Optional[np.ndarray] = None,
        early_stopping: bool = False,
    ) -> np.ndarray:
        X = X_missing.copy()
        num_nodes = np.shape(X_missing)[1]

        log.info(
            f"""Using hyperparameters
            reg_lambda = {self.reg_lambda},
            reg_beta = {self.reg_beta},
            reg_m = {self.reg_m}, lr = {self.learning_rate}
            """
        )

        rho_i = np.array([[1]])
        alpha_i = np.array([[1.0]])
        best_loss = 1e9

        one_hot_sample = [0] * self.num_inputs
        subset_ = random.sample(range(self.num_inputs), num_nodes)
        for j in subset_:
            one_hot_sample[j] = 1

        if X_seed is None:
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
        elif len(X_seed) == 1:
            self.missing_value = [
                int((np.nanmax(X) - np.nanmin(X)) * 2)
            ] * self.num_inputs
            inds = np.where(np.isnan(X))
            X = np.array(X)
            X[inds] = np.take(self.missing_value, inds[1])
        else:
            X = X_seed

        avg_seed: list = []

        for step in range(1, self.max_steps):

            if step == 0 or (step % 1 == 0 and step > self.window):

                X_pred = np.mean(np.array(avg_seed), axis=0)
                X = X * X_mask + X_pred * (1 - X_mask)
                if step % 10 == 0:
                    log.info(f"step {step}, of {self.max_steps} steps")

            h_value, loss = self.sess.run(
                [self.h, self.supervised_loss],
                feed_dict={
                    self.X: X,
                    self.X_mask: X_mask,
                    self.keep_prob: 1,
                    self.rho: rho_i,
                    self.alpha: alpha_i,
                    self.is_train: True,
                    self.noise: 0,
                },
            )
            log.debug(
                "Step " + str(step) + ", Loss= " + f"{loss:.4f}",
                " h_value:",
                h_value,
            )

            idxs = np.arange(X.shape[0])
            random.shuffle(idxs)
            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):
                batch_x = X[
                    idxs[
                        step1 * self.batch_size : step1 * self.batch_size
                        + self.batch_size
                    ]
                ]
                batch_x_masks = X_mask[
                    step1 * self.batch_size : step1 * self.batch_size + self.batch_size
                ]
                self.sess.run(
                    self.loss_op_dag,
                    feed_dict={
                        self.X: batch_x,
                        self.X_mask: batch_x_masks,
                        self.sample: one_hot_sample,
                        self.keep_prob: 1,
                        self.rho: rho_i,
                        self.alpha: alpha_i,
                        self.Lambda: self.reg_lambda,
                        self.is_train: True,
                        self.noise: 0,
                    },
                )

            if early_stopping:
                if loss < best_loss and step > 10:
                    self.saver.save(self.sess, self.tmp)
                    best_loss = loss

            if len(avg_seed) >= self.window:
                avg_seed.pop(0)
            avg_seed.append(self._transform(X))

        if early_stopping:
            self.saver.restore(self.sess, self.tmp)

        log.debug(self._transform(X))

        X_pred = np.mean(np.array(avg_seed), axis=0)
        return X * X_mask + X_pred * (1 - X_mask), X_pred, avg_seed

    def fit(
        self, X_missing: np.ndarray, X_seed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if X_seed is None:
            X_seed = X_missing.copy()
            col_mean = np.nanmean(X_seed, axis=0)
            inds = np.where(np.isnan(X_seed))
            X_seed[inds] = np.take(col_mean, inds[1])

        df_mask = pd.DataFrame(X_missing)
        df_mask = df_mask.where(df_mask.isnull(), 0)
        df_mask = df_mask.mask(df_mask.isnull(), 1)

        indicators = df_mask[df_mask.columns[df_mask.isin([1]).any()]].values
        indicators = 1 - indicators

        X_MASK = np.ones(X_missing.shape)
        X_MASK[np.isnan(X_missing)] = 0

        X_MISSING_c = np.concatenate([X_missing, indicators], axis=1)
        X_seed_c = np.concatenate([X_seed, indicators], axis=1)
        num_input_missing = indicators.shape[1]
        X_MASK_c = np.concatenate(
            [X_MASK, np.ones((X_MASK.shape[0], num_input_missing))], axis=1
        )

        transformed, _, _ = self._fit(X_MISSING_c, X_MASK_c, X_seed=X_seed_c)

        return transformed[:, : X_missing.shape[1]]

    def rmse_loss(
        self, ori_data: np.ndarray, imputed_data: np.ndarray, data_m: np.ndarray
    ) -> np.ndarray:
        numerator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
        denominator = np.sum(1 - data_m)
        return np.sqrt(numerator / float(denominator))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.sess.run(
            self.Out,
            feed_dict={
                self.X: X,
                self.keep_prob: 1,
                self.is_train: False,
                self.noise: 0,
            },
        )
