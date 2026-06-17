from collections import deque
from typing import Deque, Dict, Optional, Tuple
import torch
import numpy as np


class Timer:
    """Track recent ``(time, pearson)`` points and predict when a target Pearson is reached.

    The prediction assumes a power-law relation in log-space:

    ``log(1 - pearson) = a * log(time) + b``.

    Fitting starts only after ``burnout`` updates have been observed and once at
    least ``min_points`` buffered points are available.
    """

    def __init__(
        self,
        target_pearson: float,
        memory_size: int = 50,
        burnout: int = 80,
        min_points: int = 20,
    ):
        if memory_size <= 0:
            raise ValueError("memory_size must be > 0")
        if burnout < 0:
            raise ValueError("burnout must be >= 0")
        if min_points <= 0:
            raise ValueError("min_points must be > 0")
        if min_points > memory_size:
            raise ValueError("min_points must be <= memory_size")
        target_pearson = float(target_pearson)
        if target_pearson >= 1.0:
            raise ValueError("target_pearson must be < 1")
        if target_pearson <= -1.0:
            raise ValueError("target_pearson must be > -1")
        self.target_pearson = target_pearson
        self.memory_size = int(memory_size)
        self.burnout = int(burnout)
        self.min_points = int(min_points)
        self.n_observations = 0
        self.time: Deque[float] = deque(maxlen=self.memory_size)
        self.pearson: Deque[float] = deque(maxlen=self.memory_size)

    def update(self, time: float, pearson: float) -> None:
        """Append a new observation.

        Args:
            time (float): Elapsed training time. Must be > 0.
            pearson (float): Current Pearson correlation. Must be < 1.
        """
        time = float(time)
        pearson = float(pearson)
        if time <= 0:
            raise ValueError("time must be > 0")
        if pearson >= 1.0:
            raise ValueError("pearson must be < 1")
        self.n_observations += 1
        if self.n_observations > self.burnout:
            self.time.append(time)
            self.pearson.append(pearson)

    def predict(self) -> Optional[float]:
        """Predict the total training time needed to reach the configured target Pearson.

        Returns:
            Optional[float]: Predicted total time to reach the target Pearson. Returns ``None``
                during burnout, when there are not enough valid points, or when the fit
                is not predictive.
        """
        # Start estimating only after the burn-in period.
        if self.n_observations < self.burnout:
            return None
        if len(self.time) < self.min_points or len(self.pearson) < self.min_points:
            return None
        target_pearson = self.target_pearson

        t = np.asarray(self.time, dtype=float)
        p = np.asarray(self.pearson, dtype=float)

        # Keep only finite values in the valid fit domain.
        mask = np.isfinite(t) & np.isfinite(p) & (t > 0.0) & (p < 1.0)
        t = t[mask]
        p = p[mask]
        if t.size < self.min_points:
            return None

        # If target already reached in observed data, return the first reached time.
        reached_idx = np.where(p >= target_pearson)[0]
        if reached_idx.size > 0:
            return float(t[reached_idx[0]])

        if target_pearson <= np.max(p):
            raise ValueError("target_pearson must be greater than all stored pearson values unless already reached")

        x = np.log(t)
        y = np.log(1.0 - p)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return None

        slope, intercept = np.polyfit(x, y, 1)

        # For convergence, 1-pearson should decrease with time, hence negative slope.
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope >= 0.0:
            return None

        y_target = np.log(1.0 - target_pearson)
        x_target = (y_target - intercept) / slope
        if x_target >= 12:
            return None
        t_target = float(np.exp(x_target))

        return t_target


def init_parameters(fi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Initialize the parameters of the DCA model. The bias terms are initialized
    from the single-point frequencies 'fi', while the coupling matrix is initialized
    to zero.

    Args:
        fi (torch.Tensor): Single-point frequencies of the data.

    Returns:
        Dict[str, torch.Tensor]: 
            "bias" (torch.Tensor): Bias terms.
            "coupling_matrix" (torch.Tensor): Coupling matrix.
    """
    L, q = fi.shape
    params = {}
    params["bias"] = torch.log(fi)
    params["coupling_matrix"] = torch.zeros((L, q, L, q), device=fi.device, dtype=fi.dtype)
    
    return params


def init_chains(
    num_chains: int,
    L: int,
    q: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    fi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Initialize the Markov chains of the DCA model. If 'fi' is provided, the chains are sampled from the
    profile model, otherwise they are sampled uniformly at random.

    Args:
        num_chains (int): Number of parallel chains.
        L (int): Length of the MSA.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the chains.
        dtype (torch.dtype, optional): Data type of the chains. Defaults to torch.float32.
        fi (Optional[torch.Tensor], optional): Single-point frequencies. Defaults to None.

    Returns:
        torch.Tensor: Initialized Markov chains in one-hot encoding format, shape (num_chains, L, q).
    """
    if fi is None:
        chains = torch.randint(low=0, high=q, size=(num_chains, L), device=device)
    else:
        chains = torch.multinomial(fi, num_samples=num_chains, replacement=True).to(device=device).T
    
    return torch.nn.functional.one_hot(chains, num_classes=q).to(dtype)


def get_mask_save(L: int, q: int, device: torch.device) -> torch.Tensor:
    """Returns the mask to save the upper-triangular part of the coupling matrix.
    
    Args:
        L (int): Length of the MSA.
        q (int): Number of values that each residue can assume.
        device (torch.device): Device where to store the mask.
        
    Returns:
        torch.Tensor: Mask.
    """
    mask_save = torch.ones((L, q, L, q), dtype=torch.bool, device=device)
    idx1_rm, idx2_rm = torch.tril_indices(L, L, offset=0)
    mask_save[idx1_rm, :, idx2_rm, :] = 0
    
    return mask_save


@torch.jit.script
def systematic_resampling(
    chains: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Performs the systematic resampling of the chains according to their relative weight.

    Args:
        chains (torch.Tensor): Chains.
        weights (torch.Tensor): Weights of the chains.

    Returns:
        torch.Tensor: Resampled chains.
    """
    num_chains = len(chains)
    device = chains.device
    # Normalize the weights
    weights = weights.view(-1) / weights.sum()
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    chains = torch.repeat_interleave(chains, counts, dim=0)

    return chains


def resample_sequences(
    data: torch.Tensor,
    weights: torch.Tensor,
    nextract: int,
) -> torch.Tensor:
    """Extracts nextract sequences from data with replacement according to the weights.
    
    Args:
        data (torch.Tensor): Data array.
        weights (torch.Tensor): Weights of the sequences.
        nextract (int): Number of sequences to be extracted.

    Returns:
        torch.Tensor: Extracted sequences.
    """
    weights = weights.view(-1)
    indices = torch.multinomial(weights, nextract, replacement=True)
    
    return data[indices]


def get_device(device: str, message: bool = True) -> torch.device:
    """Returns the device where to store the tensors.
    
    Args:
        device (str): Device to be used. Possible values are 'cpu', 'cuda', 'mps'.
        message (bool, optional): Print the device. Defaults to True.
        
    Returns:
        torch.device: Device.
    """
    if "mps" in device:
        if message:
            print(f"Running on M chip GPU, Metal Performance Shaders (MPS)")
        return torch.device(device)
    if "cuda" in device and torch.cuda.is_available():
        if message:
            print(f"Running on {torch.cuda.get_device_name(torch.device(device))}")
        return torch.device(device)
    else:
        if message:
            print("Running on CPU")
        return torch.device("cpu")
    
    
def get_dtype(dtype: str) -> torch.dtype:
    """Returns the data type of the tensors.
    
    Args:
        dtype (str): Data type. Possible values are 'float32' and 'float64'.
        
    Returns:
        torch.dtype: Data type.
    """
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Data type {dtype} not supported.")
    
    
def parse_log_file(log_path: str) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Parse a DCA training log file.
    
    Args:
        log_path (str): Path to the log file.
        
    Returns:
        Tuple[Dict[str, str], Dict[str, np.ndarray]]: Dictionary containing metadata and training data.
    """
    metadata = {}
    data = {
        'Epochs': [],
        'Pearson': [],
        'Slope': [],
        'LL_train': [],
        'LL_val': [],
        'Pearson_val': [],
        'Slope_val': [],
        'ESS': [],
        'Entropy': [],
        'Density': [],
        'Time': []
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Parse metadata
    i = 0
    while i < len(lines) and lines[i].strip():
        line = lines[i].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
            i += 1
        else:
            break
    
    # Find the header line
    while i < len(lines):
        if lines[i].strip().startswith('Epochs'):
            header = lines[i].strip().split()
            i += 1
            break
        i += 1
    
    # Parse data lines
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this is a new section (e.g., "Decimation")
        if not line[0].isdigit() and '.' not in line.split()[0]:
            # Skip section headers
            i += 1
            continue
            
        try:
            values = line.split()
            if len(values) >= len(header):
                for j, key in enumerate(header):
                    if key in data:
                        data[key].append(float(values[j]))
        except (ValueError, IndexError):
            pass
        
        i += 1
    
    # Convert lists to numpy arrays
    parsed_data = {key: np.array(values) for key, values in data.items()}
    
    return metadata, parsed_data