"""Steered sampling: a user potential added to the DCA Hamiltonian.

A steering potential ``V(x, s)`` of strength ``s`` changes the sampled
distribution from ``p(x) ∝ exp(-beta * H(x))`` to

    p_s(x) ∝ exp(-beta * H(x) - V(x, s)),

so sequences with low ``V`` are favoured. Samples of ``p_s`` are turned back
into averages under ``p`` with the importance weights ``w(x) ∝ exp(V(x, s))``.

The potential is a black box evaluated on whole sequences, so it cannot enter
the site-by-site conditional distributions of the fast DCA kernels. Instead,
:class:`SteeredKernel` proposes a block of ``proposal_steps`` Gibbs updates
under ``H`` alone and accepts the block with probability
``min(1, exp(-[V(x') - V(x)]))``.

For this Metropolis-Hastings correction to be exact, the proposal must be
reversible with respect to ``exp(-beta * H)``. A Gibbs update of one site is;
a sequence of them in a fixed site order is not, but a palindromic one
(``s1 s2 ... sm ... s2 s1``) is, for every choice of sites. The sites of each
block are therefore drawn at random, shared by all chains so the block runs
as one fused kernel, and visited in palindromic order. (Sharing a
non-palindromic random order across chains would leave each chain correct on
average but bias the population as a whole.) The potential is called once
per chain and block.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import torch

from adabmDCA.exceptions import InputValidationError

SteeringInput = Literal["sequences", "onehot"]
SteeringPotential = Callable[[Sequence[str] | torch.Tensor, float], Sequence[float] | np.ndarray | torch.Tensor]
STEERING_INPUTS = ("sequences", "onehot")

# Adaptive proposal blocks aim at this acceptance band, up to this many sweeps' worth of updates.
_ADAPT_LOW, _ADAPT_HIGH = 0.2, 0.5
_MAX_BLOCK_SWEEPS = 16


class Steering:
    """A validated steering potential, evaluated on batches of chains.

    Args:
        potential: ``potential(batch, strength)`` returning one value per sequence
            of ``batch``. With ``steering_input="sequences"`` the batch is a list
            of aligned sequence strings; with ``"onehot"`` it is a tensor of shape
            ``(n, L, q)`` on the model's device and precision. It must return
            ``0`` for every sequence when ``strength`` is ``0``.
        tokens: Ordered alphabet of the model, used to decode sequences.
        steering_input: ``"sequences"`` or ``"onehot"``.
    """

    def __init__(self, potential: SteeringPotential, *, tokens: str, steering_input: SteeringInput = "sequences"):
        if not callable(potential):
            raise InputValidationError("steering_potential must be callable as potential(batch, strength).")
        if steering_input not in STEERING_INPUTS:
            raise InputValidationError(f"steering_input must be one of {STEERING_INPUTS}.")
        self.potential = potential
        self.tokens = tokens
        self.steering_input = steering_input
        self._token_array = np.array(list(tokens), dtype="<U1")
        self.evaluations = 0

    def decode(self, states: torch.Tensor) -> list[str]:
        """Decode categorical chains ``(n, L)`` to aligned sequence strings."""
        length = states.shape[1]
        characters = np.ascontiguousarray(self._token_array[states.detach().cpu().numpy()])
        return characters.view(f"<U{length}").ravel().tolist()

    def __call__(self, chains: torch.Tensor, strength: float, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Evaluate the potential on one-hot ``(n, L, q)`` or categorical ``(n, L)`` chains.

        Returns:
            Float64 tensor of shape ``(n,)`` on the chains' device; zeros for ``strength == 0``.
        """
        n = chains.shape[0]
        if strength == 0.0 or n == 0:
            return torch.zeros(n, dtype=torch.float64, device=chains.device)
        return self._evaluate(chains, strength, dtype=dtype)

    def _evaluate(self, chains: torch.Tensor, strength: float, *, dtype: torch.dtype | None) -> torch.Tensor:
        n = chains.shape[0]
        if self.steering_input == "sequences":
            states = chains.argmax(-1) if chains.ndim == 3 else chains
            batch = self.decode(states)
        else:
            if chains.ndim == 3:
                batch = chains if dtype is None else chains.to(dtype)
            else:
                batch = torch.nn.functional.one_hot(chains.long(), len(self.tokens)).to(dtype or torch.float32)
        values = self.potential(batch, float(strength))
        self.evaluations += n
        if isinstance(values, torch.Tensor):
            values = values.detach().to(device=chains.device, dtype=torch.float64)
        else:
            try:
                values = torch.as_tensor(np.asarray(values, dtype=np.float64), device=chains.device)
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    "steering_potential must return one number per sequence (a list, array or tensor)."
                ) from exc
        values = values.reshape(-1) if values.ndim > 1 and values.numel() == n else values
        if values.shape != (n,):
            raise InputValidationError(
                f"steering_potential returned shape {tuple(values.shape)}; expected ({n},), one value per sequence."
            )
        if not torch.isfinite(values).all():
            raise InputValidationError("steering_potential returned non-finite values.")
        return values

    def check_zero_strength(self, chains: torch.Tensor, *, dtype: torch.dtype | None = None) -> None:
        """Raise unless the potential vanishes at strength 0, as the steering ladder requires."""
        values = self._evaluate(chains, 0.0, dtype=dtype)
        if values.abs().max() > 1e-9:
            raise InputValidationError(
                "steering_potential(batch, 0) must be 0 for every sequence: strength 0 means no steering. "
                "Multiply the potential by the strength, e.g. lambda seqs, s: s * score(seqs)."
            )


class SteeredKernel:
    """Metropolis-Hastings kernel for ``exp(-beta * H - V(·, strength))``.

    Each proposal is ``proposal_steps`` Gibbs updates under the DCA
    Hamiltonian, at random sites shared by all chains and visited in
    palindromic order; it is accepted with probability ``min(1, exp(-ΔV))``.
    One call with ``nsweeps`` performs ``nsweeps * L`` site updates; its last
    block is shortened when needed.

    With ``proposal_steps=None`` the block length adapts while ``adapting`` is
    true (doubling when acceptance exceeds 0.5, halving below 0.2, at most 16
    sweeps); call :meth:`freeze` before collecting samples so the kernel is a
    fixed, exactly invariant Markov chain.

    Args:
        steering: The validated potential.
        strength: Steering strength passed to the potential.
        length: Sequence length ``L``.
        device: Device of the chains, which selects the Triton step on CUDA.
        proposal_steps: Site updates per proposal, or ``None`` to adapt.
    """

    def __init__(self, steering: Steering, strength: float, *, length: int, device: torch.device,
                 proposal_steps: int | None = None):
        if proposal_steps is not None and (isinstance(proposal_steps, bool) or not isinstance(proposal_steps, int)
                                           or proposal_steps < 1):
            raise InputValidationError("steering_proposal_steps must be a positive integer or None.")
        self.steering = steering
        self.strength = float(strength)
        self.length = length
        self.adapting = proposal_steps is None
        self.proposal_steps = proposal_steps or max(1, length // 10)
        self._fused = _fused_gibbs(str(device))
        self.proposals = 0
        self.accepted = 0
        self._cache_chains = None
        self._cache_values = None

    @property
    def acceptance(self) -> float:
        """Fraction of accepted block proposals since the kernel was frozen (or created)."""
        return self.accepted / self.proposals if self.proposals else float("nan")

    def freeze(self) -> None:
        """Stop adapting the block length and reset the acceptance counters."""
        self.adapting = False
        self.proposals = self.accepted = 0

    def values(self, chains: torch.Tensor) -> torch.Tensor:
        """Potential of one-hot ``chains``, reused when they are the kernel's last output."""
        if chains is self._cache_chains:
            return self._cache_values
        return self.steering(chains, self.strength, dtype=chains.dtype)

    def remember(self, chains: torch.Tensor, values: torch.Tensor) -> None:
        """Record the potential of ``chains`` so the next call need not evaluate it again."""
        self._cache_chains, self._cache_values = chains, values

    @torch.no_grad()
    def run(self, chains: torch.Tensor, params: dict[str, torch.Tensor], nsweeps: int, beta: float = 1.0,
            values: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one-hot chains; return them with their potential values."""
        current = self.values(chains) if values is None else values
        updates = nsweeps * self.length
        done = 0
        while done < updates:
            # The last block of a call is shortened to the requested work: every
            # block is an exact Metropolis-Hastings step for p_s, whatever its length.
            steps = min(self.proposal_steps, updates - done)
            proposal = self._propose(chains, params, steps, float(beta))
            proposed = self.steering(proposal, self.strength, dtype=chains.dtype)
            log_u = torch.rand(len(chains), device=chains.device, dtype=torch.float64).log()
            accept = log_u < -(proposed - current)
            chains = torch.where(accept[:, None, None], proposal, chains)
            current = torch.where(accept, proposed, current)
            rate = float(accept.double().mean())
            self.proposals += 1
            self.accepted += rate
            done += steps
            if self.adapting:
                if rate > _ADAPT_HIGH:
                    self.proposal_steps = min(_MAX_BLOCK_SWEEPS * self.length, 2 * self.proposal_steps)
                elif rate < _ADAPT_LOW:
                    self.proposal_steps = max(1, self.proposal_steps // 2)
        self.remember(chains, current)
        return chains, current

    def _propose(self, chains: torch.Tensor, params: dict[str, torch.Tensor], steps: int,
                 beta: float) -> torch.Tensor:
        """``steps`` Gibbs updates at random shared sites in palindromic order."""
        half = (steps + 1) // 2
        first = torch.randint(0, self.length, (half,), device=chains.device)
        sites = torch.cat([first, first[:steps - half].flip(0)])
        if self._fused is not None and params["bias"].dtype in (torch.float32, torch.float64):
            states = self._fused(chains.argmax(-1).to(torch.int32), params, 0, beta, sites=sites)
            return torch.nn.functional.one_hot(states.long(), chains.shape[-1]).to(chains.dtype)
        return _gibbs_at_sites(chains.clone(), params, sites, beta)

    def __call__(self, chains: torch.Tensor, params: dict[str, torch.Tensor], nsweeps: int,
                 beta: float = 1.0) -> torch.Tensor:
        """Sampler interface of :mod:`adabmDCA.sampling`: return the advanced one-hot chains."""
        return self.run(chains, params, nsweeps, beta)[0]


def _fused_gibbs(device: str) -> Callable[..., torch.Tensor] | None:
    """A compiled categorical Gibbs sampler with explicit sites: Triton on CUDA, Numba on CPU."""
    if device == "cpu":
        from adabmDCA.numba_kernels import categorical_sampler, is_numba_available

        return categorical_sampler("gibbs") if is_numba_available() else None
    if not device.startswith("cuda"):
        return None
    try:
        from adabmDCA.sampling_triton import gibbs_sampling_categorical_triton, is_triton_available
    except ImportError:
        return None
    return gibbs_sampling_categorical_triton if is_triton_available() else None


def _gibbs_at_sites(chains: torch.Tensor, params: dict[str, torch.Tensor], sites: torch.Tensor,
                    beta: float) -> torch.Tensor:
    """Gibbs updates of one-hot ``chains`` in place, at ``sites`` shared by all chains."""
    n, length, q = chains.shape
    flat = chains.view(n, length * q)
    couplings = params["coupling_matrix"].reshape(length, q, length * q)
    for site in sites.tolist():
        logits = beta * (params["bias"][site] + flat @ couplings[site].T)
        new = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
        chains[:, site] = torch.nn.functional.one_hot(new, q).to(chains.dtype)
    return chains


def importance_warning(ess: float, n: int) -> str | None:
    """A warning when the importance weights are too concentrated to reweight reliably."""
    if ess < max(10.0, 0.01 * n):
        return (f"The importance weights have an effective sample size of {ess:.0f} out of {n}: the steered "
                "and unsteered distributions barely overlap, so reweighted averages are unreliable. "
                "Use a weaker steering_strength to estimate unsteered averages.")
    return None


def importance_summary(potentials: np.ndarray, log_z_ratio: float | None = None) -> tuple[np.ndarray, float]:
    """Log importance weights back to the unsteered model and their effective sample size.

    Args:
        potentials: ``V(x, s)`` of each steered sample.
        log_z_ratio: ``log Z_s - log Z_0`` when known (PTT); otherwise the
            weights are self-normalized to mean 1.

    Returns:
        ``(log_weights, effective_sample_size)``.
    """
    potentials = np.asarray(potentials, dtype=np.float64)
    if log_z_ratio is None:
        log_weights = potentials - (np.logaddexp.reduce(potentials) - math.log(len(potentials)))
    else:
        log_weights = potentials + log_z_ratio
    normalized = np.exp(log_weights - log_weights.max())
    ess = float(normalized.sum() ** 2 / (normalized ** 2).sum())
    return log_weights, ess
