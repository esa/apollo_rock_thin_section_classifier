# coding=utf-8
# Copyright 2020 The SimCLR Authors
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

import re
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

EETA_DEFAULT = 0.001


# Keep in mind this is OptimizerV2, may provide some versioning issues
class LARSOptimizer(tf.keras.optimizers.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    exclude_from_weight_decay: Optional[List[str]]
    exclude_from_layer_adaptation: Optional[List[str]]

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        use_nesterov: bool = False,
        weight_decay: float = 0.0,
        exclude_from_weight_decay: Optional[List[str]] = None,
        exclude_from_layer_adaptation: Optional[List[str]] = None,
        classic_momentum: bool = True,
        eeta: float = EETA_DEFAULT,
        name: str = "LARSOptimizer",
    ) -> None:
        """Constructs a LARSOptimizer.

        Args:
          learning_rate: A `float` for learning rate.
          momentum: A `float` for momentum.
          use_nesterov: A 'Boolean' for whether to use nesterov momentum.
          weight_decay: A `float` for weight decay.
          exclude_from_weight_decay: A list of `string` for variable screening, if
              any of the string appears in a variable's name, the variable will be
              excluded for computing weight decay. For example, one could specify
              the list like ['batch_normalization', 'bias'] to exclude BN and bias
              from weight decay.
          exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
              for layer adaptation. If it is None, it will be defaulted the same as
              exclude_from_weight_decay.
          classic_momentum: A `boolean` for whether to use classic (or popular)
              momentum. The learning rate is applied during momentum update in
              classic momentum, but after momentum for popular momentum.
          eeta: A `float` for scaling of learning rate when computing trust ratio.
          name: The name for the scope.
        """
        super(LARSOptimizer, self).__init__(
            name="LARSOptimizer", learning_rate=learning_rate
        )

        # self._set_hyper("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation is not None:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slots(self, var_list: list) -> None:
        """Create slots for the optimizer.

        Args:
            var_list: List of variables to create slots for.
        """
        for v in var_list:
            self.add_slot(v, "Momentum")

    def _resource_apply_dense(
        self,
        grad: tf.Tensor,
        param: tf.Variable,
        apply_state: Optional[Dict[Tuple[str, tf.DType], Dict[str, tf.Tensor]]] = None,
    ) -> tf.Operation:
        """Apply gradients to variables.

        Args:
            grad: The gradient
            param: The variable
            apply_state: Additional optimizer state from apply_gradients

        Returns:
            An operation that applies the gradient
        """
        if grad is None or param is None:
            return tf.no_op()

        var_device, var_dtype = param.device, param.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        learning_rate = coefficients["lr_t"]

        param_name = param.name

        v = self.get_slot(param, "Momentum")

        if self._use_weight_decay(param_name):
            grad += self.weight_decay * param

        if self.classic_momentum:
            trust_ratio = 1.0
            if self._do_layer_adaptation(param_name):
                w_norm = tf.norm(param, ord=2)
                g_norm = tf.norm(grad, ord=2)
                trust_ratio = tf.where(
                    tf.greater(w_norm, 0),
                    tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
                    1.0,
                )
            scaled_lr = learning_rate * trust_ratio

            next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
            if self.use_nesterov:
                update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
            else:
                update = next_v
            next_param = param - update
        else:
            next_v = tf.multiply(self.momentum, v) + grad
            if self.use_nesterov:
                update = tf.multiply(self.momentum, next_v) + grad
            else:
                update = next_v

            trust_ratio = 1.0
            if self._do_layer_adaptation(param_name):
                w_norm = tf.norm(param, ord=2)
                v_norm = tf.norm(update, ord=2)
                trust_ratio = tf.where(
                    tf.greater(w_norm, 0),
                    tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
                    1.0,
                )
            scaled_lr = trust_ratio * learning_rate
            next_param = param - scaled_lr * update

        return tf.group(
            *[
                param.assign(next_param, use_locking=False),
                v.assign(next_v, use_locking=False),
            ]
        )

    def _use_weight_decay(self, param_name: str) -> bool:
        """Whether to use L2 weight decay for `param_name`.

        Args:
            param_name: Name of the parameter.
        """
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name: str) -> bool:
        """Whether to do layer-wise learning rate adaptation for `param_name`.

        Args:
            param_name: Name of the parameter.
        """
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def get_config(self) -> dict:
        """Get config of the optimizer.

        Returns:
            A dictionary containing the configuration of the optimizer.
        """
        config = super(LARSOptimizer, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "classic_momentum": self.classic_momentum,
                "weight_decay": self.weight_decay,
                "eeta": self.eeta,
                "use_nesterov": self.use_nesterov,
            }
        )
        return config
