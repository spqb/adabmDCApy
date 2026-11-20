"""Steered sampling: hard alphabet and Hamming constraints, linear distance biases and weighted callbacks.

:mod:`adabmDCA.steering.sampling` holds :class:`SteeringConfig` and the MCMC and
PTT samplers, :mod:`adabmDCA.steering.support` the constraints and the exact
constrained profile, and :mod:`adabmDCA.steering.kernels` the constrained Gibbs
updates. The anchor restricted to Hamming windows is
:class:`adabmDCA.anchor.window.WindowedAnchor`.
"""
