from typing import Dict, Any, Optional
import torch
import wandb
from adabmDCA.io import save_chains, save_params


class Checkpoint:
    """Helper class to save the model's parameters and chains at regular intervals during training and to log the
    progress of the training.
    """
    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        params: Optional[Dict[str, torch.Tensor]] = None,
        chains: Optional[torch.Tensor] = None,
        use_wandb: bool = False,
    ):
        """Initializes the Checkpoint class.

        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the arguments of the training.
            params (Optional[Dict[str, torch.Tensor]], optional): Parameters of the model. Defaults to None.
            chains (Optional[torch.Tensor], optional): Chains. Defaults to None.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
        """
            
        self.file_paths = file_paths
        self.tokens = tokens
        
        self.wandb = use_wandb
        if self.wandb:
            wandb.init(project="adabmDCA", config=args)
            
        if params is not None:
            self.params = {key: value.clone() for key, value in params.items()}
        else:
            self.params = None
        if chains is not None:
            self.chains = chains.clone()
        else:
            self.chains = None
        self.max_epochs = args["nepochs"]
        self.checkpt_interval = 50
        
        self.logs = {
            "Epochs": 0,
            "Pearson": 0.0,
            "Slope": 0.0,
            "LL_train": 0.0,
            "LL_test": 0.0,
            "ESS": 0.0,
            "Entropy": 0.0,
            "Density": 0.0,
            "Time": 0.0,
        }
        
        template = "{0:<20} {1:<50}\n"  
        with open(file_paths["log"], "w") as f:
            if args["label"] is not None:
                f.write(template.format("label:", args["label"]))
            else:
                f.write(template.format("label:", "N/A"))
            
            f.write(template.format("model:", str(args["model"])))
            f.write(template.format("input MSA:", str(args["data"])))
            f.write(template.format("alphabet:", args["alphabet"]))
            f.write(template.format("sampler:", args["sampler"]))
            f.write(template.format("nchains:", args["nchains"]))
            f.write(template.format("nsweeps:", args["nsweeps"]))
            f.write(template.format("lr:", args["lr"]))
            f.write(template.format("pseudo count:", args["pseudocount"]))
            f.write(template.format("data type:", args["dtype"]))
            f.write(template.format("target Pearson Cij:", args["target"]))
            if args["model"] == "eaDCA":
                f.write(template.format("gsteps:", args["gsteps"]))
                f.write(template.format("factivate:", args["factivate"]))
            f.write(template.format("random seed:", args["seed"]))
            f.write("\n")
            # write the header of the log file
            header_string = " ".join([f"{key:<10}" for key in self.logs.keys()])
            f.write(header_string + "\n")
        
        
    def log(
        self,
        record: Dict[str, Any],
    ) -> None:
        """Adds a key-value pair to the log dictionary

        Args:
            record (Dict[str, Any]): Key-value pairs to be added to the log dictionary.
        """
        for key, value in record.items():
            if key not in self.logs.keys():
                raise ValueError(f"Key {key} not recognized.")
        
            if isinstance(value, torch.Tensor):
                self.logs[key] = value.item()
            else:
                self.logs[key] = value
                
        if self.wandb:
            wandb.log(self.logs)        
        out_string = " ".join([f"{value:<10.3f}" if isinstance(value, float) else f"{value:<10}" for value in self.logs.values()])
        with open(self.file_paths["log"], "a") as f:
            f.write(out_string + "\n")
    
    
    def check(
        self,
        updates: int,
    ) -> bool:
        """Checks if a checkpoint has been reached.
        
        Args:
            updates (int): Number of gradient updates performed.

        Returns:
            bool: Whether a checkpoint has been reached.
        """
        return (updates % self.checkpt_interval == 0) or (updates == self.max_epochs)
        
        
    def save(
        self,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph
            chains (torch.Tensor): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """            
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights)