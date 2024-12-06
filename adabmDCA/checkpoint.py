from typing import Dict
from abc import ABC, abstractmethod
import torch
import time
import h5py

from adabmDCA.io import save_chains, save_params
from adabmDCA.statmech import _get_acceptance_rate


class Checkpoint(ABC):
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        max_epochs: int,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
    ):
        self.file_paths = file_paths
        self.tokens = tokens
        if params is not None:
            self.params = {key: value.clone() for key, value in params.items()}
        else:
            self.params = None
        if chains is not None:
            self.chains = chains.clone()
        else:
            self.chains = None
        self.max_epochs = max_epochs
        self.updates = 0
        
    
    @abstractmethod
    def check(
        self,
        updates: int,
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            updates (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        pass
        

    @abstractmethod
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        pass
    
    
class LinearCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        max_epochs: int,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
        checkpt_interval: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            max_epochs=max_epochs,
            params=params,
            chains=chains,
        )
        self.checkpt_interval = checkpt_interval
        self.max_epochs = max_epochs
        
    
    def check(
        self,
        updates: int,
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            updates (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        self.updates = updates
        return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph.
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        template = "{epochs:10} {pearson:10} {slope:10} {log_likelihood:10} {entropy:10} {density:10} {time:10}\n"
        with open(self.file_paths["log"], "a") as f:
            f.write(template.format(
                epochs = kwargs["epochs"],
                pearson = kwargs["pearson"],
                slope = kwargs["slope"],
                log_likelihood = kwargs["log_likelihood"],
                entropy = kwargs["entropy"],
                density = kwargs["density"],
                time = (time.time() - kwargs["time_start"]),
                )
            )
            
            
class AcceptanceCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: Dict,
        tokens: str,
        max_epochs: int,
        params: Dict[str, torch.Tensor] | None = None,
        chains: Dict[str, torch.Tensor] | None = None,
        target_acc_rate: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(
            file_paths=file_paths,
            tokens=tokens,
            max_epochs=max_epochs,
            params=params,
            chains=chains,
        )
        self.target_acc_rate = target_acc_rate
        self.num_saved_models = 0
        # Create a .h5 archive for storing the history of the parameters
        self.file_paths["params_history"] = self.file_paths["params"].with_suffix(".h5")
        with h5py.File(self.file_paths["params_history"], "w") as f:
            f["alphabet"] = self.tokens
            f.create_group(f"update_{self.updates}")
            for key, value in params.items():
                f[f"update_{self.updates}"].create_dataset(key, data=value.cpu().numpy())
        
        
        
    def check(
        self,
        updates: int,
        curr_params: Dict[str, torch.Tensor],
        curr_chains: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached by computing the acceptance rate of swapping the 
        configurations of the present model and the last saved model.
        
        Args:
            updates (int): Number of gradient updates performed.
            curr_params (Dict[str, torch.Tensor]): Current parameters of the model.
            curr_chains (Dict[str, torch.Tensor]): Current chains of the model.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        acc_rate = _get_acceptance_rate(
            prev_params=self.params,
            curr_params=curr_params,
            prev_chains=self.chains,
            curr_chains=curr_chains,
        )
        self.updates = updates
        return (acc_rate < self.target_acc_rate) or (updates == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Saves the chains and the parameters of the model and appends the current parameters to the
        file containing the parameters history.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph.
            chains (Dict[str, torch.Tensor]): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        self.num_saved_models += 1
        # Store the current parameters and chains
        self.params = {key: value.clone() for key, value in params.items()}
        self.chains = chains.clone()
        # Append the current parameters to the history
        with h5py.File(self.file_paths["params_history"], "a") as f:
            f.create_group(f"update_{self.updates}")
            for key, value in params.items():
                f[f"update_{self.updates}"].create_dataset(key, data=value.cpu().numpy())
        # Save the current parameters and chains
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        # Update the log file
        template = "{epochs:10} {pearson:10} {slope:10} {log_likelihood:10} {entropy:10} {density:10} {time:10}\n"
        with open(self.file_paths["log"], "a") as f:
            f.write(template.format(
                epochs = kwargs["epochs"],
                pearson = kwargs["pearson"],
                slope = kwargs["slope"],
                log_likelihood = kwargs["log_likelihood"],
                entropy = kwargs["entropy"],
                density = kwargs["density"],
                time = (time.time() - kwargs["time_start"]),
                )
            )
        
            
def get_checkpoint(chpt: str) -> Checkpoint:
    if chpt == "linear":
        return LinearCheckpoint
    elif chpt == "acceptance":
        return AcceptanceCheckpoint
    else:
        raise ValueError(f"Checkpoint type {chpt} not recognized.")        