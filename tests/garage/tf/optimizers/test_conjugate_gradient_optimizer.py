"""Tests for garage.tf.optimizers.conjugateGradientOptimizer"""
import numpy as np
import tensorflow as tf

from garage.tf.optimizers.conjugate_gradient_optimizer import (
    cg, FiniteDifferenceHvp, PerlmutterHvp)
from garage.tf.policies import Policy


class OneParamPolicy(Policy):
    """Helper class for testing hvp classes"""

    def __init__(self, name='OneParamPolicy'):
        super().__init__(name, None)
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            _ = tf.Variable([0.])

    def get_action(self, observation):
        pass

    def get_actions(self, observations):
        pass


class TestConjugateGradientOptimizer:
    """Test class for ConjugateGradientOptimizer and HVP classes"""

    def test_cg(self):
        """Solve Ax = b using Conjugate gradient method."""
        a = np.random.randn(5, 5)
        a = a.T.dot(a)  # make sure a is positive semi-definite
        b = np.random.randn(5)
        x = cg(a.dot, b, cg_iters=5)
        assert np.allclose(a.dot(x), b)


class TestPermmutterHvp:
    """Test class for PermmutterHvp"""

    def test_perm_mutter_hvp(self):
        """Test if hessian product calculations are correct"""
        policy = OneParamPolicy()
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.Variable([0.0])
        f = a * (x**2)
        expected_hessian = 2 * a_val
        vector = np.array([10.0])
        expected_hvp = expected_hessian * vector
        reg_coeff = 1e-5
        hvp = PerlmutterHvp()

        sess = tf.compat.v1.Session()
        sess.__enter__()
        sess.run(tf.global_variables_initializer())

        hvp.update_opt(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        computed_hvp = hx(vector)
        np.allclose(computed_hvp, expected_hvp)


class TestFiniteDifferenceHvp:
    """Test class for FiniteDifferenceHvp"""

    def test_finite_difference_hvp(self):
        """Test if hessian product calculations are correct"""
        policy = OneParamPolicy()
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.Variable([0.0])
        f = a * (x**2)
        expected_hessian = 2 * a_val
        vector = np.array([10.0])
        expected_hvp = expected_hessian * vector
        reg_coeff = 1e-5
        hvp = FiniteDifferenceHvp()

        sess = tf.compat.v1.Session()
        sess.__enter__()
        sess.run(tf.global_variables_initializer())

        hvp.update_opt(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        computed_hvp = hx(vector)
        np.allclose(computed_hvp, expected_hvp)
