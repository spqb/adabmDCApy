import torch


def _get_ortho(mat: torch.Tensor):
    """Orthonormalize the column vectors of a matrix.

    Args:
        mat (torch.Tensor): Matrix to orthonormalized. (a, b)

    Returns:
        torch.Tensor: Orthonormalized matrix. (a, b)
    """
    res = mat.clone()
    d = mat.shape[1]

    u0 = mat[:, 0] / mat[:, 0].norm()
    res[:, 0] = u0
    for i in range(1, d):
        ui = mat[:, i]
        for j in range(i):
            ui -= (ui @ res[:, j]) * res[:, j]
        res[:, i] = ui / ui.norm()
        
    return res


def _compute_U(
    M: torch.Tensor,
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the first 'num_directions' right eigenvector of the dataset.

    Args:
        M (torch.Tensor): Dataset (num_samples, num_variables).
        num_directions (int): Number of principal components to compute.
        device (torch.device): Device to use.
        dtype (torch.dtype): Data type to use.
    
    Returns:
        torch.Tensor: Principal components (num_variables, num_directions).
    """
    num_samples, num_variables = M.shape
    max_iter = 100
    err_threshold = 1e-15
    curr_v = torch.rand(num_samples, num_directions, device=device, dtype=dtype) * 2 - 1
    u = torch.rand(num_variables, num_directions, device=device, dtype=dtype)
    curr_id_mat = torch.rand(num_directions, num_directions, device=device, dtype=dtype) * 2 - 1
    for n in range(max_iter):
        v = curr_v.clone()
        curr_v = M @ u
        if num_samples < num_variables:
            id_mat = (v.T @ curr_v) / num_samples
            curr_v = _get_ortho(curr_v)
        curr_u = M.T @ curr_v
        if num_variables <= num_samples:
            id_mat = (u.T @ curr_u) / num_samples
            curr_u = _get_ortho(curr_u)
        u = curr_u.clone()
        if (id_mat - curr_id_mat).norm() < err_threshold:
            break
        curr_id_mat = id_mat.clone()
    u = _get_ortho(u)

    return u

class Pca():
    def __init__(self):
        self.num_directions = 0
        self.U = None

    def fit(
        self,
        M: torch.Tensor,
        num_directions: int = 2,
    ) -> None:
        """Fit the PCA model to the data.

        Args:
            M (torch.Tensor): Data matrix (num_samples, num_variables).
            num_directions (int): Number of principal components to compute.
        """
        # Check that the input matrix is a 2D tensor
        if M.dim() != 2:
            raise ValueError("Input matrix must be a 2D tensor.")
        self.num_directions = num_directions
        self.U = _compute_U(M, self.num_directions, M.device, M.dtype)

    def transform(
        self,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """Projects the data onto the principal components.

        Args:
            M (torch.Tensor): Data matrix (num_samples, num_variables).

        Returns:
            torch.Tensor: Projected data matrix (num_samples, num_directions).
        """
        return M @ self.U

    def fit_transform(
        self,
        M: torch.Tensor,
        num_directions: int = 2,
    ) -> torch.Tensor:
        """Fit the PCA model to the data and project the data onto the principal components.

        Args:
            M (torch.Tensor): Data matrix (num_samples, num_variables).
            num_directions (int): Number of principal components to compute.

        Returns:
            torch.Tensor: Projected data matrix (num_samples, num_directions).
        """
        self.fit(M, num_directions)
        return self.transform(M)