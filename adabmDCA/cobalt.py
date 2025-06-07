import torch
from tqdm.autonotebook import trange

@torch.jit.script
def has_neighbours(seq: torch.Tensor, db: torch.Tensor, threshold: float) -> bool:
    """
    Check if a sequence 'seq' has neighbours in the database 'db', that is, sequences
    that have pairwise sequence identity greater than 'threshold'.
    """
    if len(db) == 0:
        return False
    else:
        seqid = (seq == db).float().sum(dim=1) / seq.shape[0]
        return bool((seqid > threshold).any().item())


def split_train_test(
    headers: list[str],
    X: torch.Tensor,
    seqid_th: float,
    rnd_gen: torch.Generator | None = None,
) -> tuple[list, torch.Tensor, list, torch.Tensor]:
    """Splits X into two sets, T and S, such that no sequence in S has more than
    'seqid_th' fraction of its residues identical to any sequence in T.
    
    Args:
        headers (list[str]): List of sequence headers.
        X (torch.Tensor): Encoded input MSA.
        seqid_th (float): Threshold sequence identity.
        rnd_gen (torch.Generator, optional): Random number generator. Defaults to None.

    Returns:
        tuple[list, torch.Tensor, list, torch.Tensor]: Training and test sets.
    """
    T_mask = torch.tensor([False] * len(X), dtype=torch.bool, device=X.device)
    S_mask = torch.tensor([False] * len(X), dtype=torch.bool, device=X.device)
    perm = torch.randperm(len(X), generator=rnd_gen, device=X.device)
    X = X[perm]
    for i in trange(len(X), desc="Train/test splitting", leave=False):
        has_neighbours_in_S = has_neighbours(X[i], X[S_mask], seqid_th)
        has_neighbours_in_T = has_neighbours(X[i], X[T_mask], seqid_th)
        if torch.rand(1, generator=rnd_gen, device=X.device).item() < 0.5:
            if not has_neighbours_in_T:
                S_mask[i] = True
            elif not has_neighbours_in_S:
                T_mask[i] = True
        else:
            if not has_neighbours_in_S:
                T_mask[i] = True
            elif not has_neighbours_in_T:
                S_mask[i] = True
    
    T_mask = T_mask.cpu().numpy()
    S_mask = S_mask.cpu().numpy()
    T_headers = headers[T_mask]
    T = X[T_mask]
    S_headers = headers[S_mask]
    S = X[S_mask]
    
    if len(T) > len(S):
        return T_headers, T, S_headers, S
    else:
        return S_headers, S, T_headers, T
    
    
def prune_redundant_sequences(
    headers: list[str],
    X: torch.Tensor,
    seqid_th: float,
    rnd_gen: torch.Generator | None = None,
) -> tuple[list, torch.Tensor]:
    """Prunes sequences from X such that no sequence has more than 'seqid_th' fraction of its residues identical to any other sequence in the set.

    Args:
        headers (list[str]): List of sequence headers.
        X (torch.Tensor): Encoded input MSA.
        seqid_th (float): Threshold sequence identity.
        rnd_gen (torch.Generator, optional): Random generator. Defaults to None.

    Returns:
        tuple[list, torch.Tensor]: Pruned sequences.
    """
    perm = torch.randperm(len(X), generator=rnd_gen, device=X.device)
    X = X[perm]
    U_mask = torch.tensor([False] * len(X), dtype=torch.bool, device=X.device)
    for i in trange(len(X), desc="Pruning redundant sequences", leave=False):
        has_neighbours_in_U = has_neighbours(X[i], X[U_mask], seqid_th)
        if not has_neighbours_in_U:
            U_mask[i] = True
    U_mask = U_mask.cpu().numpy()
    
    return headers[U_mask], X[U_mask]


def run_cobalt(
    headers: list[str],
    X: torch.Tensor,
    t1: float,
    t2: float,
    t3: float,
    max_train: int | None,
    max_test: int | None,
    rnd_gen: torch.Generator | None = None,
) -> tuple[list, torch.Tensor, list, torch.Tensor]:
    """
    Runs the Cobalt algorithm to split the input MSA into training and test sets.
    
    Args:
        headers (list[str]): List of sequence headers.
        X (torch.Tensor): Encoded input MSA.
        t1 (float): No sequence in S has more than this fraction of its residues identical to any sequence in T.
        t2 (float): No pair of test sequences has more than this value fractional identity.
        t3 (float): No pair of training sequences has more than this value fractional identity.
        max_train (int | None): Maximum number of sequences in the training set.
        max_test (int | None): Maximum number of sequences in the test set.
        rnd_gen (torch.Generator, optional): Random number generator. Defaults to None.

    Returns:
        tuple[list, torch.Tensor, list, torch.Tensor]: Training and test sets.
    """
    # Cobalt step 1
    headers_train, train, headers_test, test = split_train_test(headers, X, t1, rnd_gen)
    # Cobalt step 2
    if len(test) > 0:
        headers_test, test = prune_redundant_sequences(headers_test, test, t2, rnd_gen)
    # Cobalt step 3
    if len(train) > 0:
        headers_train, train = prune_redundant_sequences(headers_train, train, t3, rnd_gen)
    
    if max_train is not None and len(train) > max_train:
        train_ids = torch.randperm(len(train), generator=rnd_gen)[:max_train]
        headers_train = headers_train[train_ids]
        train = train[train_ids]
    if max_test is not None and len(test) > max_test:
        test_ids = torch.randperm(len(test), generator=rnd_gen)[:max_test]
        headers_test = headers_test[test_ids]
        test = test[test_ids]
        
    return headers_train, train, headers_test, test