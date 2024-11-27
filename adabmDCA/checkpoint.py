from typing import Dict
from abc import ABC, abstractmethod
import torch
import time
import h5py

from adabmDCA.io import save_chains, save_params
from adabmDCA.statmech import get_acceptance_rate


class Checkpoint(ABC):
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        max_epochs: int,
        params: Dict[str, torch.Tensor] = None,
        chains: Dict[str, torch.Tensor] = None,
    ):
        self.file_paths = file_paths
        self.tokens = tokens
        self.params = {key: value.clone() for key, value in params.items()}
        self.chains = chains.clone()
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
        params: Dict[str, torch.Tensor] = None,
        chains: Dict[str, torch.Tensor] = None,
        checkpt_interval: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(file_paths, tokens, params, chains)
        self.checkpt_interval = checkpt_interval
        
    
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
        template = "{0:10} {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}\n"
        with open(self.file_paths["log"], "a") as f:
            f.write(template.format(
                f"{kwargs["epochs"]}",
                f"{kwargs["pearson"]:.3f}",
                f"{kwargs["slope"]:.3f}",
                f"{kwargs["log_likelihood"]:.3f}",
                f"{kwargs["entropy"]:.3f}",
                f"{kwargs["density"]:.3f}",
                f"{(time.time() - kwargs["time_start"]):.1f}",
                )
            )
            
            
class AcceptanceCheckpoint(Checkpoint):
    def __init__(
        self,
        file_paths: Dict,
        tokens: str,
        max_epochs: int,
        params: Dict[str, torch.Tensor] = None,
        chains: Dict[str, torch.Tensor] = None,
        target_acc_rate: float = 0.25,
        *args,
        **kwargs,
    ):
        super().__init__(file_paths, tokens, max_epochs, params, chains)
        self.target_acc_rate = target_acc_rate
        self.num_saved_models = 0
        # Convert the output file of the parameters into a .h5 archive
        self.file_paths["params"] = self.file_paths["params"].with_suffix(".h5")
        # Create the output file for the parameters
        with h5py.File(self.file_paths["params"], "w") as f:
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
        acc_rate = get_acceptance_rate(
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
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        self.num_saved_models += 1
        self.params = {key: value.clone() for key, value in params.items()}
        self.chains = chains.clone()
        with h5py.File(self.file_paths["params"], "a") as f:
            f.create_group(f"update_{self.updates}")
            for key, value in params.items():
                f[f"update_{self.updates}"].create_dataset(key, data=value.cpu().numpy())
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)
        template = "{0:10} {1:10} {2:10} {3:10} {4:10} {5:10} {6:10}\n"
        with open(self.file_paths["log"], "a") as f:
            f.write(template.format(
                f"{kwargs["epochs"]}",
                f"{kwargs["pearson"]:.3f}",
                f"{kwargs["slope"]:.3f}",
                f"{kwargs["log_likelihood"]:.3f}",
                f"{kwargs["entropy"]:.3f}",
                f"{kwargs["density"]:.3f}",
                f"{(time.time() - kwargs["time_start"]):.1f}",
                )
            )
        
            
def get_checkpoint(chpt: str) -> Checkpoint:
    if chpt == "linear":
        return LinearCheckpoint
    elif chpt == "acceptance":
        return AcceptanceCheckpoint
    else:
        raise ValueError(f"Checkpoint type {chpt} not recognized.")        