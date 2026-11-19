import random
from bisect import bisect_right
from math import comb, nan
from typing import Tuple, Optional
import torch


def _normalize_weights(data: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must have nonempty shape (N, L, q).")
    dtype = data.dtype if data.is_floating_point() else torch.get_default_dtype()
    if weights is None:
        data = data.to(dtype)
        return data, data.new_full((len(data), 1), 1 / len(data))
    if weights.numel() != len(data):
        raise ValueError("Provide one weight per sequence.")
    dtype = torch.promote_types(dtype, weights.dtype)
    weights = weights.reshape(-1, 1).to(device=data.device, dtype=dtype)
    total = weights.sum()
    if not (torch.isfinite(total) & (total != 0)):
        raise ValueError("Sequence weights must have a finite, nonzero sum.")
    return data.to(dtype), weights / total


def _get_freq_single_point(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:    
    _, _, q = data.shape
    frequencies = (data * weights).sum(dim=0)
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(frequencies, min=0.0)

    return (1. - pseudo_count) * frequencies + (pseudo_count / q)


def get_freq_single_point(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Computes the single point frequencies of the input MSA.
    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Weights of the sequences.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    data, norm_weights = _normalize_weights(data, weights)
    return _get_freq_single_point(data, norm_weights.unsqueeze(-1), pseudo_count)


def _get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:
    
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    
    fij = (data_oh * weights).T @ data_oh
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(fij, min=0.0)
    # Apply the pseudo count
    fij = (1. - pseudo_count) * fij + (pseudo_count / q**2)
    # Within a site, two different one-hot states cannot occur together.
    fi = _get_freq_single_point(data, weights.unsqueeze(-1), pseudo_count)
    blocks = fij.reshape(L, q, L, q)
    sites = torch.arange(L, device=data.device)
    blocks[sites, :, sites, :] = torch.diag_embed(fi)
    return blocks


def get_freq_two_points(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """
    Computes the 2-points statistics of the input MSA.

    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
    
    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
    """
    data, norm_weights = _normalize_weights(data, weights)
    return _get_freq_two_points(data, norm_weights, pseudo_count)


def generate_unique_triplets(
    L: int,
    ntriplets: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates a set of unique triplets of positions. Used to compute the 3-points statistics.
    
    Args:
        L (int): Length of the sequences.
        ntriplets (int): Number of triplets to be generated.
        device (torch.device, optional): Device to perform computations on. Defaults to "cpu".
    
    Returns:
        torch.Tensor: Indices of up to ntriplets unique triplets, with shape (count, 3).
    """    
    if L < 0 or ntriplets < 0:
        raise ValueError("L and ntriplets must be nonnegative.")
    total = comb(L, 3)
    count = min(total, ntriplets)
    if count == 0:
        return torch.empty((0, 3), dtype=torch.long, device=device)

    # Sample lexicographic ranks and locate the first two sites by their prefix counts.
    seed = torch.randint(0, 2**63 - 1, (), device="cpu").item()
    ranks = random.Random(seed).sample(range(total), count)
    triplet_prefix = [total - comb(L - i, 3) for i in range(L - 1)]
    pair_count = comb(L, 2)
    pair_prefix = [pair_count - comb(L - i, 2) for i in range(L)]
    triplets = []
    for rank in ranks:
        i = bisect_right(triplet_prefix, rank) - 1
        pair_rank = rank - triplet_prefix[i] + pair_prefix[i + 1]
        j = bisect_right(pair_prefix, pair_rank) - 1
        k = j + 1 + pair_rank - pair_prefix[j]
        triplets.append((i, j, k))
    return torch.tensor(triplets, dtype=torch.long, device=device)


def _get_C_ijk(triplet: list[int], centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the third central moment of three sites in already centered data."""
    i, j, k = triplet
    return torch.einsum("ma,mb,mc->abc", centered[:, i] * weights, centered[:, j], centered[:, k])


def get_freq_three_points(
    nat: torch.Tensor,
    gen: torch.Tensor,
    ntriplets: int = 10000,
    triplets: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the 3-body connected correlation statistics of the input MSAs.

    Args:
        nat (torch.Tensor): Input MSA representing natural data in one-hot encoding.
        gen (torch.Tensor): Input MSA representing generated data in one-hot encoding.
        ntriplets (int): Number of triplets to test. Defaults to 10000.
        triplets (Optional[torch.Tensor], optional): Predefined triplets to use for the computation. Defaults to None.
        weights (Optional[torch.Tensor], optional): Importance weights for the natural sequences. Defaults to None.
        device (torch.device, optional): Device to perform computations on. Defaults to "cpu".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Natural and generated 3-points connected correlation for ntriplets randomly extracted triplets.
    """
    nat, norm_weights = _normalize_weights(nat, weights)
    gen, uniform_weights = _normalize_weights(gen, None)
    if nat.shape[1:] != gen.shape[1:]:
        raise ValueError("The two MSAs must have the same length and state count.")
    if triplets is None:
        triplets = generate_unique_triplets(L=nat.shape[1], ntriplets=ntriplets, device=device)
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError("triplets must have shape (count, 3).")
    if len(triplets) == 0:
        return nat.new_empty(0), gen.new_empty(0)
    if triplets.is_floating_point() or triplets.is_complex() or triplets.dtype == torch.bool:
        raise ValueError("Triplet site indices must be integers.")

    nat_mean = nat.mean(dim=0) if weights is None else (nat * norm_weights.unsqueeze(-1)).sum(dim=0)
    nat_centered = nat - nat_mean
    gen_centered = gen - gen.mean(dim=0)
    indices = triplets.tolist()
    return (
        torch.stack([_get_C_ijk(t, nat_centered, norm_weights) for t in indices]).flatten(),
        torch.stack([_get_C_ijk(t, gen_centered, uniform_weights) for t in indices]).flatten(),
    )


def get_covariance_matrix(
    data: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
) -> torch.Tensor:
    """Computes the weighted covariance matrix of the input multi sequence alignment.

    Args:
        data (torch.Tensor): Input MSA in one-hot variables.
        weights (torch.Tensor | None, optional): Importance weights of the sequences.
        pseudo_count (float, optional): Pseudo count. Defaults to 0.0.

    Returns:
        torch.Tensor: Covariance matrix.
    """
    _, L, q = data.shape
    fi = get_freq_single_point(data, weights, pseudo_count)
    fij = get_freq_two_points(data, weights, pseudo_count)
    cov_matrix = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    
    return cov_matrix.reshape(L * q, L * q)


def extract_Cij_from_freq(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the frequencies.

    Args:
        fij (torch.Tensor): Two-point frequencies of the natural data.
        pij (torch.Tensor): Two-point frequencies of the generated data.
        fi (torch.Tensor): Single-point frequencies of the natural data.
        pi (torch.Tensor): Single-point frequencies of the generated data.
        mask (Optional[torch.Tensor], optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted covariance matrix entries of the natural data and generated data.
    """
    idx_row, idx_col = torch.tril_indices(fi.shape[0], fi.shape[0], offset=-1, device=fij.device)
    cov_data = fij[idx_row, :, idx_col, :] - fi[idx_row, :, None] * fi[idx_col, None, :]
    cov_chains = pij[idx_row, :, idx_col, :] - pi[idx_row, :, None] * pi[idx_col, None, :]
    if mask is not None:
        selected = torch.broadcast_to(mask, fij.shape)[idx_row, :, idx_col, :]
        return cov_data[selected], cov_chains[selected]
    return cov_data.flatten(), cov_chains.flatten()


def extract_Cij_from_seqs(
    data: torch.Tensor,
    chains: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pseudo_count: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts the lower triangular part of the covariance matrices of the natural data and generated data starting from the sequences.

    Args:
        data (torch.Tensor): Natural data sequences.
        chains (torch.Tensor): Generated data sequences.
        weights (torch.Tensor | None, optional): Weights of the sequences. Defaults to None.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.
        mask (torch.Tensor | None, optional): Mask for comparing just a subset of the couplings. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Selected connected correlations of the natural and generated data.
    """
    fi = get_freq_single_point(data, weights=weights, pseudo_count=pseudo_count)
    pi = get_freq_single_point(chains, weights=None)
    fij = get_freq_two_points(data, weights=weights, pseudo_count=pseudo_count)
    pij = get_freq_two_points(chains, weights=None)
    
    return extract_Cij_from_freq(fij, pij, fi, pi, mask)


def get_correlation_two_points(
    fij: torch.Tensor,
    pij: torch.Tensor,
    fi: torch.Tensor,
    pi: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Compute Pearson correlation and absolute regression slope for connected correlations.

    Args:
        fij (torch.Tensor): Two-point frequencies of the natural data.
        pij (torch.Tensor): Two-point frequencies of the generated data.
        fi (torch.Tensor): Single-point frequencies of the natural data.
        pi (torch.Tensor): Single-point frequencies of the generated data.
        mask (Optional[torch.Tensor], optional): Mask to select the couplings to use for the correlation coefficient. Defaults to None. 

    Returns:
        Tuple[float, float]: Pearson correlation and absolute least-squares slope. Undefined values are NaN.
    """
    
    x, y = extract_Cij_from_freq(fij, pij, fi, pi, mask)
    if x.numel() < 2:
        return nan, nan
    dtype = torch.promote_types(torch.promote_types(x.dtype, y.dtype), torch.float32)
    x, y = x.to(dtype), y.to(dtype)
    x, y = x - x[0], y - y[0]
    x, y = x - x.mean(), y - y.mean()
    xy = x @ y
    xx = x @ x
    yy = y @ y
    pearson, slope = torch.stack([(xy / (xx.sqrt() * yy.sqrt())).clamp(-1, 1), (xy / xx).abs()]).tolist()
    return pearson, slope


def _one_hot_rows(sequences: torch.Tensor, q: int, chunk: int, dtype: torch.dtype):
    """Yield ``(start, rows)`` with ``rows`` the flattened one-hot encoding ``(n, L * q)`` of a chunk."""
    for start in range(0, len(sequences), chunk):
        block = sequences[start:start + chunk]
        yield start, torch.nn.functional.one_hot(block.long(), q).to(dtype).flatten(1)


def _weighted_moments(sequences, weights, q, chunk, dtype):
    """Weighted one-hot means and covariance ``(L q, L q)``, with weights normalized to one."""
    weights = (weights / weights.sum()).to(dtype)
    mean = None
    for start, rows in _one_hot_rows(sequences, q, chunk, dtype):
        part = weights[start:start + len(rows)] @ rows
        mean = part if mean is None else mean + part
    covariance = torch.zeros(len(mean), len(mean), dtype=dtype, device=mean.device)
    for start, rows in _one_hot_rows(sequences, q, chunk, dtype):
        centered = rows - mean
        covariance += (centered * weights[start:start + len(rows), None]).T @ centered
    return weights, mean, covariance


def _third_moments(sequences, weights, mean, u, v, q, chunk, dtype):
    """For seed state pairs ``(u, v)``: weighted mean of ``c_u c_v c_w`` and of its square, for every state ``w``."""
    first = second = None
    for start, rows in _one_hot_rows(sequences, q, chunk, dtype):
        centered = rows - mean
        pair = centered[:, u] * centered[:, v] * weights[start:start + len(rows), None]
        part1 = pair.T @ centered
        part2 = (pair * centered[:, u] * centered[:, v]).T @ (centered * centered)
        first = part1 if first is None else first + part1
        second = part2 if second is None else second + part2
    return first, second


def _triplet_values(sequences, weights, triplets, q, chunk, dtype):
    """Connected correlations of explicit state triplets ``(K, 3)`` (flat ``site * q + state`` indices)."""
    weights = (weights / weights.sum()).to(dtype)
    columns = torch.unique(triplets)
    position = torch.searchsorted(columns, triplets)
    mean = None
    for start, rows in _one_hot_rows(sequences, q, chunk, dtype):
        part = weights[start:start + len(rows)] @ rows[:, columns]
        mean = part if mean is None else mean + part
    total = torch.zeros(len(triplets), dtype=dtype, device=mean.device)
    for start, rows in _one_hot_rows(sequences, q, chunk, dtype):
        centered = rows[:, columns] - mean
        product = centered[:, position[:, 0]] * centered[:, position[:, 1]] * centered[:, position[:, 2]]
        total += weights[start:start + len(rows)] @ product
    return total


def compare_three_point_correlations(
    data: torch.Tensor,
    samples: torch.Tensor,
    *,
    q: int,
    data_weights: Optional[torch.Tensor] = None,
    gap_index: Optional[int] = 0,
    n_seeds: int = 3000,
    max_triplets: int = 2000,
    max_pair_correlation: float = 0.1,
    frequency_range: Tuple[float, float] = (0.05, 0.95),
    z_threshold: float = 5.0,
    seed: int = 0,
    chunk: int = 4096,
) -> dict:
    """Connected three-point correlations of data and samples on resolvable, not pairwise-implied triplets.

    A connected correlation ``C_ijk(a, b, c)`` of one-hot indicators is not a
    three-body interaction: pairwise models produce it too. Two selections keep
    the comparison informative:

    - no strong pair: the correlation coefficient of each two of the three
      states, in the data, is below ``max_pair_correlation`` in absolute value.
      Near-redundant states would turn ``C_ijk`` into a rescaled two-point
      correlation (for indicators ``x^2 = x``, so ``x_u = x_v`` gives
      ``C_uvw = (1 - 2 f_u) C_uw``). States are non-gap and have data
      frequencies within ``frequency_range``;
    - resolvable: ``z = max(|C_data| / SE_data, |C_samples| / SE_samples)``
      is at least ``z_threshold``. The standard errors are the weighted
      standard deviation of the product of the three centred indicators over
      the square root of the effective number of sequences: the sum of the
      data weights (as given, so without reweighting every sequence counts as
      independent and the error is optimistic) and the number of samples.
      Taking the larger ratio does not reward agreement between data and
      samples. With about 10^7 candidates, about five pure-noise ones reach
      ``z = 5``.

    The search cannot enumerate all triplets. It draws ``n_seeds`` random state
    pairs that pass the selection and computes their correlation with every
    third state in one matrix product per data chunk. Of the triplets above
    the threshold it keeps a random sample of ``max_triplets``, so that the
    comparison covers moderate correlations as well as the largest. The data are also split at random into two
    halves: the Pearson correlation of the selected triplets between them is
    the ``noise_ceiling``, the agreement the data allow (slightly optimistic,
    since the selection used all the data).

    Args:
        data: Integer-encoded data ``(N, L)``.
        samples: Integer-encoded ``(M, L)`` or one-hot ``(M, L, q)`` samples.
        q: Number of states.
        data_weights: Sequence weights of the data; ``None`` weighs them equally.
        gap_index: Index of the gap state, excluded; ``None`` keeps all states.
        n_seeds: Random seed pairs.
        max_triplets: Largest number of triplets returned.
        max_pair_correlation: Largest absolute pair correlation coefficient.
        frequency_range: Allowed data frequencies of the three states.
        z_threshold: Smallest signal-to-noise ratio.
        seed: Seed of the seed pairs and of the half split.
        chunk: Sequences per one-hot block.

    Returns:
        ``triplets`` (``(K, 3, 2)`` array of ``(site, state)``), ``data``,
        ``samples`` and ``z`` (one value per triplet), ``pearson``, ``slope``,
        ``r2`` (:func:`r2_about_identity`), ``noise_ceiling`` and
        ``noise_ceiling_r2`` (the same two measures between the data halves), ``n_seeds``, ``n_candidates`` (triplets passing the
        pair and frequency selection), ``n_resolved`` (of them, above the
        threshold), ``n_eff_data``, ``n_samples`` and the selection settings.
    """
    if samples.ndim == 3:
        samples = samples.argmax(-1)
    if data.ndim != 2 or samples.ndim != 2 or data.shape[1] != samples.shape[1]:
        raise ValueError("data and samples must be integer-encoded alignments of the same length.")
    device = data.device
    samples = samples.to(device)
    dtype = torch.float32
    length = data.shape[1]
    states = length * q
    raw_weights = (torch.ones(len(data), device=device) if data_weights is None
                   else data_weights.reshape(-1).to(device=device, dtype=dtype))
    n_eff_data = float(raw_weights.sum())
    wd, fd, cov_d = _weighted_moments(data, raw_weights, q, chunk, dtype)
    wg, fg, _ = _weighted_moments(samples, torch.ones(len(samples), device=device), q, chunk, dtype)
    site = torch.arange(states, device=device) // q
    variance = fd * (1 - fd)
    correlation = cov_d / torch.sqrt(variance[:, None] * variance[None, :]).clamp_min(1e-12)
    usable = (fd >= frequency_range[0]) & (fd <= frequency_range[1])
    if gap_index is not None:
        usable &= torch.arange(states, device=device) % q != gap_index
    weak = correlation.abs() < max_pair_correlation

    # Seeds: random pairs of usable states at different sites with a weak correlation.
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.nonzero(usable).flatten().cpu()
    seeds = torch.empty((0, 2), dtype=torch.long)
    for _ in range(50):
        if len(seeds) >= n_seeds or len(candidates) < 2:
            break
        draw = candidates[torch.randint(len(candidates), (4 * n_seeds, 2), generator=generator)]
        ok = (site.cpu()[draw[:, 0]] != site.cpu()[draw[:, 1]]) & weak.cpu()[draw[:, 0], draw[:, 1]]
        seeds = torch.cat([seeds, draw[ok]])[:n_seeds]
    u, v = seeds[:, 0].to(device), seeds[:, 1].to(device)

    result = {"max_pair_correlation": max_pair_correlation, "frequency_range": list(frequency_range),
              "z_threshold": z_threshold, "n_seeds": len(u), "n_eff_data": n_eff_data, "n_samples": len(samples)}
    empty = {"triplets": torch.empty((0, 3, 2), dtype=torch.long).numpy(), "data": [], "samples": [], "z": [],
             "pearson": None, "slope": None, "noise_ceiling": None, "r2": None, "noise_ceiling_r2": None,
             "n_candidates": 0, "n_resolved": 0}
    if len(u) == 0:
        return {**result, **empty}

    mean_d, square_d = _third_moments(data, wd, fd, u, v, q, chunk, dtype)
    mean_g, square_g = _third_moments(samples, wg, fg, u, v, q, chunk, dtype)
    error_d = torch.sqrt((square_d - mean_d ** 2).clamp_min(0) / n_eff_data).clamp_min(1e-12)
    error_g = torch.sqrt((square_g - mean_g ** 2).clamp_min(0) / len(samples)).clamp_min(1e-12)
    z = torch.maximum(mean_d.abs() / error_d, mean_g.abs() / error_g)
    allowed = (usable[None, :] & weak[u] & weak[v]
               & (site[None, :] != site[u][:, None]) & (site[None, :] != site[v][:, None]))
    # The same triplet appears once per choice of its seed pair: count and keep it once.
    flat = torch.nonzero(allowed)
    keys = torch.sort(torch.stack([u[flat[:, 0]], v[flat[:, 0]], flat[:, 1]], 1), 1).values
    keys, first = _unique_rows(keys)
    flat = flat[first]
    scores = z[flat[:, 0], flat[:, 1]]
    resolved = scores >= z_threshold
    result.update(n_candidates=len(flat), n_resolved=int(resolved.sum()))
    # A random sample of the resolved triplets: the largest z alone would leave out every
    # correlation of moderate size and turn the scatter into two clouds of opposite sign.
    count = int(resolved.sum())
    order = (torch.randperm(count, generator=generator)[:max_triplets].to(device) if count > max_triplets
             else torch.arange(count, device=device))
    chosen = flat[resolved][order]
    if len(chosen) < 2:
        return {**result, **{key: value for key, value in empty.items() if key not in result}}
    triplets = keys[resolved][order]
    values_d = mean_d[chosen[:, 0], chosen[:, 1]]
    values_g = mean_g[chosen[:, 0], chosen[:, 1]]

    halves = torch.randperm(len(data), generator=generator).to(device)
    first_half, second_half = halves[: len(data) // 2], halves[len(data) // 2:]
    half_a = _triplet_values(data[first_half], raw_weights[first_half], triplets, q, chunk, dtype)
    half_b = _triplet_values(data[second_half], raw_weights[second_half], triplets, q, chunk, dtype)

    a, b = values_d.double().cpu(), values_g.double().cpu()
    centered = a - a.mean()
    result.update(
        triplets=torch.stack([triplets // q, triplets % q], -1).cpu().numpy(),
        data=a.numpy(), samples=b.numpy(), z=scores[resolved][order].double().cpu().numpy(),
        pearson=float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        slope=float(centered @ (b - b.mean()) / (centered @ centered)),
        noise_ceiling=float(torch.corrcoef(torch.stack([half_a.double().cpu(), half_b.double().cpu()]))[0, 1]),
        r2=r2_about_identity(a, b),
        noise_ceiling_r2=r2_about_identity(half_a.double().cpu(), half_b.double().cpu()),
    )
    return result


def r2_about_identity(reference, generated) -> float:
    """``1 - sum (generated - reference)^2 / sum (reference - mean)^2``.

    Unlike the Pearson correlation it penalizes a wrong scale (a slope away
    from 1) and an offset as well as scatter: it is 1 only when the points lie
    on the diagonal, and can be negative.
    """
    reference = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    generated = torch.as_tensor(generated, dtype=torch.float64).reshape(-1)
    spread = float(((reference - reference.mean()) ** 2).sum())
    return float(1 - ((generated - reference) ** 2).sum() / spread) if spread > 0 else float("nan")


def _unique_rows(rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unique rows and the index of the first occurrence of each."""
    unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((len(unique),), len(rows), dtype=torch.long, device=rows.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(rows), device=rows.device), reduce="amin")
    return unique, first
