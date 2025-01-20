import torch
import torch.nn as nn

from tqdm import tqdm

from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points

def get_entropic_order(fi: torch.Tensor) -> torch.Tensor:
    """Returns the entropic order of the sites in the MSA.

    Args:
        fi (torch.Tensor): Single-site frequencies of the MSA.

    Returns:
        torch.Tensor: Entropic order of the sites.
    """
    site_entropy = -torch.sum(fi * torch.log(fi + 1e-10), dim=1)
   
    return torch.argsort(site_entropy, descending=False)

# Define the loss function
def loss_fn(
    model: nn.Module,
    X: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
) -> torch.Tensor:
    """Computes the negative log-likelihood of the model.
    
    Args:
        model (nn.Module): arDCA model.
        X (torch.Tensor): Input MSA one-hot encoded.
        fi_target (torch.Tensor): Single-site frequencies of the MSA.
        fij_target (torch.Tensor): Pairwise frequencies of the MSA.
    """
    n_samples, L, q = X.shape
    log_likelihood = 0
    for i in range(1, model.L):
        energy_i = (fi_target[i] @ model.h[i]) + (model.J[i, :, :i, :] * fij_target[i, :, :i, :]).sum()
        logZ_i = torch.logsumexp(model.h[i] + X[:, :i, :].view(n_samples, -1) @ model.J[i, :, :i, :].view(q, -1).mT, dim=-1).mean()
        log_likelihood += energy_i - logZ_i
    
    return - log_likelihood

# Define he model
class arDCA(nn.Module):
    def __init__(
        self,
        L: int,
        q: int,
    ):
        """Initializes the arDCA model. Either fi or L and q must be provided.

        Args:
            L (int): Number of residues in the MSA.
            q (int): Number of states for the categorical variables.

        """
        super(arDCA, self).__init__()
        self.L = L
        self.q = q
        self.h = nn.Parameter(torch.randn(L, q))
        self.J = nn.Parameter(torch.randn(self.L, self.q, self.L, self.q))
        self.revert_entropic_order = None

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the probability of next token given the previous ones.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            
        Returns:
            torch.Tensor: Probability of the next token.
        """
        # X has to be a 3D tensor of shape (n_samples, l, q) with l < L
        if X.dim() != 3:
            raise ValueError("X must be a 3D tensor")
        if X.shape[1] >= self.L:
            raise ValueError("X must have a second dimension smaller than L")
        n_samples, residue_idx = X.shape[0], X.shape[1]

        J_ar = self.J[residue_idx, :, :residue_idx, :].view(self.q, -1)
        X_ar = X.view(n_samples, -1)
        logit_i = self.h[residue_idx] + torch.einsum("ij,nj->ni", J_ar, X_ar)
        prob_i = torch.softmax(logit_i, dim=-1)
        
        return prob_i
    
    def fit(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 1000,
        target_pearson: float = 0.95,
        pseudo_count: float = 0.0,
        n_samples: int = 10000,
        use_entropic_order: bool = True,
        fix_first_residue: bool = False,
    ) -> None:
        """Fits the model to the data.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            weights (torch.Tensor): Weights of the sequences in the MSA.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 1000.
            target_pearson (float, optional): Target Pearson correlation. Defaults to 0.95.
            pseudo_count (float, optional): Pseudo-count for the frequencies. Defaults to 0.0.
            n_samples (int, optional): Number of samples to generate for computing the pearson. Defaults to 10000.
            use_entropic_order (bool, optional): Whether to use the entropic order. Defaults to True.
            fix_first_residue (bool, optional): Fix the position of the first residue so that it is not sorted by entropy.
                Used when the first residue encodes for the label. Defaults to False.
        """
        fi = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij = get_freq_two_points(X, weights=weights, pseudo_count=pseudo_count)
        if use_entropic_order:
            if fix_first_residue:
                entropic_order = get_entropic_order(fi[1:])
                entropic_order = torch.cat([torch.tensor([0], device=entropic_order.device), entropic_order + 1])
            else:
                entropic_order = get_entropic_order(fi)
            self.revert_entropic_order = torch.argsort(entropic_order)
            X = X[:, entropic_order, :]
        else:
            self.revert_entropic_order = None
        # Target frequencies, if entropic order is used, the frequencies are sorted
        fi_target = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij_target = get_freq_two_points(X, weights=weights, pseudo_count=pseudo_count)
        self.h = nn.Parameter(torch.log(fi_target + 1e-10))
        
        # Training loop
        pbar = tqdm(
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]",
        )
        pbar.set_description(f"Epochs: 0 - LL: nan")
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self, X, fi_target, fij_target)
            loss.backward()
            optimizer.step()
            
            samples = self.sample(n_samples=n_samples)
            pi = get_freq_single_point(samples)
            pij = get_freq_two_points(samples)
            pearson, _ = get_correlation_two_points(fi=fi, fij=fij, pi=pi, pij=pij)
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Epochs: {epoch} - LL: {-loss.item():.2f}")
            if pearson >= target_pearson:
                break
        pbar.close()
    
    def sample(
        self,
        n_samples: int,
    ) -> torch.Tensor:
        """Samples from the model.
        
        Args:
            n_samples (int): Number of samples to generate.
            
        Returns:
            torch.Tensor: Generated samples.
        """       
        X = torch.zeros(n_samples, self.L, self.q, dtype=self.h.dtype, device=self.h.device)
        X_init = torch.multinomial(torch.softmax(self.h[0], dim=-1), num_samples=n_samples, replacement=True) # (n_samples,)
        X[:, 0, :] = nn.functional.one_hot(X_init, self.q).to(dtype=self.h.dtype)
        
        for i in range(1, self.L):
            prob_i = self.forward(X[:, :i, :])
            sample_i = torch.multinomial(prob_i, num_samples=1).squeeze()
            X[:, i] = nn.functional.one_hot(sample_i, self.q).to(dtype=self.h.dtype)
        if self.revert_entropic_order is not None:
            X = X[:, self.revert_entropic_order, :]
            
        return X
    
    def sample_conditioned(
        self,
        prompt: torch.Tensor,
    ) -> torch.Tensor:
        """Generates samples conditioned on the prompt as the first residues.
        
        Args:
            prompt (torch.Tensor): Prompt one-hot encoded.
            
        Returns:
            torch.Tensor: Generated samples.
        """
        n_samples, _ = prompt.shape
        X = torch.zeros(n_samples, self.L, self.q, dtype=self.h.dtype, device=self.h.device)
        # create the first column of the one-hot MSA using the prompt
        X_0 = torch.zeros(n_samples, self.q, dtype=self.h.dtype, device=self.h.device)
        # if the prompt is shorter than L, fill the rest with zeros
        num_categories = prompt.shape[1]
        X_0[:, :num_categories] = prompt
        X[:, 0, :] = X_0
        # generate the rest of the MSA
        for i in range(1, self.L):
            prob_i = self.forward(X[:, :i, :])
            sample_i = torch.multinomial(prob_i, num_samples=1).squeeze()
            X[:, i] = nn.functional.one_hot(sample_i, self.q).to(dtype=self.h.dtype)
        if self.revert_entropic_order is not None:
            X = X[:, self.revert_entropic_order, :]
            
        return X