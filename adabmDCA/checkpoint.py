from typing import Dict, Tuple
from abc import ABC, abstractmethod
import torch
from adabmDCA.io import save_chains, save_params
import time


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
        self.params = params
        self.chains = chains
        self.max_epochs = max_epochs
        
    
    @abstractmethod
    def check(
        self,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.

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
        **kwargs,
    ):
        super().__init__(file_paths, tokens, params, chains)
        self.checkpt_interval = checkpt_interval
        
    
    def check(
        self,
        upds: int,
        **kwargs,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            upds (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        return (upds % self.checkpt_interval == 0) or (upds == self.max_epochs)
    
    
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: Dict[str, torch.Tensor],
        log_weights: torch.Tensor,
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
            
def get_checkpoint_fn(chpt: str) -> Checkpoint:
    if chpt == "linear":
        return LinearCheckpoint
    else:
        raise ValueError(f"Checkpoint type {chpt} not recognized.")        