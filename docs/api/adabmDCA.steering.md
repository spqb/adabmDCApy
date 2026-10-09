<!-- markdownlint-disable -->

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `adabmDCA.steering`
Steered sampling: a user potential added to the DCA Hamiltonian.

A steering potential ``V(x, s)`` of strength ``s`` changes the sampled distribution from ``p(x) ∝ exp(-beta * H(x))`` to

 p_s(x) ∝ exp(-beta * H(x) - V(x, s)),

so sequences with low ``V`` are favoured. Samples of ``p_s`` are turned back into averages under ``p`` with the importance weights ``w(x) ∝ exp(V(x, s))``.

The potential is a black box evaluated on whole sequences, so it cannot enter the site-by-site conditional distributions of the fast DCA kernels. Instead, :class:`SteeredKernel` proposes a block of ``proposal_steps`` Gibbs updates under ``H`` alone and accepts the block with probability ``min(1, exp(-[V(x') - V(x)]))``.

For this Metropolis-Hastings correction to be exact, the proposal must be reversible with respect to ``exp(-beta * H)``. A Gibbs update of one site is; a sequence of them in a fixed site order is not, but a palindromic one (``s1 s2 ... sm ... s2 s1``) is, for every choice of sites. The sites of each block are therefore drawn at random, shared by all chains so the block runs as one fused kernel, and visited in palindromic order. (Sharing a non-palindromic random order across chains would leave each chain correct on average but bias the population as a whole.) The potential is called once per chain and block.

**Global Variables**
---------------
- **STEERING_INPUTS**

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `importance_warning`

```python
importance_warning(ess: 'float', n: 'int') → str | None
```

A warning when the importance weights are too concentrated to reweight reliably.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `importance_summary`

```python
importance_summary(
    potentials: 'ndarray',
    log_z_ratio: 'float | None' = None
) → tuple[ndarray, float]
```

Log importance weights back to the unsteered model and their effective sample size.



**Args:**

 - <b>`potentials`</b>:  ``V(x, s)`` of each steered sample.
 - <b>`log_z_ratio`</b>:  ``log Z_s - log Z_0`` when known (PTT); otherwise the  weights are self-normalized to mean 1.



**Returns:**
 ``(log_weights, effective_sample_size)``.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Steering`
A validated steering potential, evaluated on batches of chains.



**Args:**

 - <b>`potential`</b>:  ``potential(batch, strength)`` returning one value per sequence  of ``batch``. With ``steering_input="sequences"`` the batch is a list  of aligned sequence strings; with ``"onehot"`` it is a tensor of shape  ``(n, L, q)`` on the model's device and precision. It must return  ``0`` for every sequence when ``strength`` is ``0``.
 - <b>`tokens`</b>:  Ordered alphabet of the model, used to decode sequences.
 - <b>`steering_input`</b>:  ``"sequences"`` or ``"onehot"``.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    potential: 'SteeringPotential',
    tokens: 'str',
    steering_input: 'SteeringInput' = 'sequences'
)
```








---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_zero_strength`

```python
check_zero_strength(chains: 'Tensor', dtype: 'dtype | None' = None) → None
```

Raise unless the potential vanishes at strength 0, as the steering ladder requires.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decode`

```python
decode(states: 'Tensor') → list[str]
```

Decode categorical chains ``(n, L)`` to aligned sequence strings.


---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SteeredKernel`
Metropolis-Hastings kernel for ``exp(-beta * H - V(·, strength))``.

Each proposal is ``proposal_steps`` Gibbs updates under the DCA Hamiltonian, at random sites shared by all chains and visited in palindromic order; it is accepted with probability ``min(1, exp(-ΔV))``. One call with ``nsweeps`` performs ``nsweeps * L`` site updates; its last block is shortened when needed.

With ``proposal_steps=None`` the block length adapts while ``adapting`` is true (doubling when acceptance exceeds 0.5, halving below 0.2, at most 16 sweeps); call :meth:`freeze` before collecting samples so the kernel is a fixed, exactly invariant Markov chain.



**Args:**

 - <b>`steering`</b>:  The validated potential.
 - <b>`strength`</b>:  Steering strength passed to the potential.
 - <b>`length`</b>:  Sequence length ``L``.
 - <b>`device`</b>:  Device of the chains, which selects the Triton step on CUDA.
 - <b>`proposal_steps`</b>:  Site updates per proposal, or ``None`` to adapt.

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    steering: 'Steering',
    strength: 'float',
    length: 'int',
    device: 'device',
    proposal_steps: 'int | None' = None
)
```






---

#### <kbd>property</kbd> acceptance

Fraction of accepted block proposals since the kernel was frozen (or created).



---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `freeze`

```python
freeze() → None
```

Stop adapting the block length and reset the acceptance counters.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remember`

```python
remember(chains: 'Tensor', values: 'Tensor') → None
```

Record the potential of ``chains`` so the next call need not evaluate it again.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    chains: 'Tensor',
    params: 'dict[str, Tensor]',
    nsweeps: 'int',
    beta: 'float' = 1.0,
    values: 'Tensor | None' = None
) → tuple[Tensor, Tensor]
```

Advance one-hot chains; return them with their potential values.

---

<a href="https://github.com/spqb/adabmDCApy/blob/main/adabmDCA/steering.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `values`

```python
values(chains: 'Tensor') → Tensor
```

Potential of one-hot ``chains``, reused when they are the kernel's last output.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
