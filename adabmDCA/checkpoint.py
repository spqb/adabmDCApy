from typing import Any

import torch
import wandb

from adabmDCA.io import save_chains, save_params
from adabmDCA.training_config import DEFAULT_CHECKPOINT_INTERVAL, TrainingConfig


class Checkpoint:
    """Helper class to save the model's parameters and chains at regular intervals during training and to log the
    progress of the training.
    """

    def __init__(
        self,
        file_paths: dict,
        tokens: str,
        args: dict,
        use_wandb: bool = False,
        config: TrainingConfig | None = None,
    ):
        """Initializes the Checkpoint class.

        Args:
            file_paths (dict): Dictionary containing the paths of the files to be saved.
            tokens (str): Alphabet to be used for encoding the sequences.
            args (dict): Dictionary containing the arguments of the training.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
        """

        self.file_paths = file_paths
        self.tokens = tokens
        self.config = config

        self.wandb = use_wandb
        if self.wandb:
            wandb.init(project="adabmDCA", config=args)

        if config is None:
            self.max_epochs = args["nepochs"]
            self.checkpt_interval = args.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)
        else:
            limits = config.limits
            self.max_epochs = limits.max_gradient_steps if config.model_type == "bmDCA" else limits.max_structure_steps
            self.checkpt_interval = config.resolved_checkpoint_interval

        self.logs = {
            "Epochs": 0,
            "Pearson": 0.0,
            "Slope": 0.0,
            "LL_train": 0.0,
            "LL_val": 0.0,
            "Pearson_val": 0.0,
            "Slope_val": 0.0,
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
            if args.get("val") is not None:
                f.write(template.format("validation MSA:", str(args["val"])))
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
            if args["model"] == "edDCA":
                f.write(template.format("target density:", args["target_density"]))
                f.write(template.format("decimation rate:", args["decimation_rate"]))
            f.write(template.format("max gradient steps:", str(args.get("max_gradient_steps"))))
            f.write(template.format("max structure steps:", str(args.get("max_structure_steps"))))
            f.write(template.format("checkpoint interval:", self.checkpt_interval))
            f.write(template.format("random seed:", args["seed"]))
            f.write("\n")
            # write the header of the log file
            header_string = " ".join([f"{key:<15}" for key in self.logs])
            f.write(header_string + "\n")

    def log(
        self,
        record: dict[str, Any],
    ) -> None:
        """Adds a key-value pair to the log dictionary

        Args:
            record (Dict[str, Any]): Key-value pairs to be added to the log dictionary.
        """
        for key, value in record.items():
            if key not in self.logs:
                raise ValueError(f"Key {key} not recognized.")

            if isinstance(value, torch.Tensor):
                self.logs[key] = value.item()
            else:
                self.logs[key] = value

        if self.wandb:
            wandb.log(self.logs)
        out_string = " ".join(
            [f"{value:<15.3f}" if isinstance(value, float) else f"{value:<15}" for value in self.logs.values()]
        )
        with open(self.file_paths["log"], "a") as f:
            f.write(out_string + "\n")

    def begin_stage(self, stage: str, metadata: dict[str, Any]) -> None:
        """Record a training phase boundary in the human-readable log."""
        with open(self.file_paths["log"], "a") as f:
            f.write(f"\n[{stage.upper()}]\n")
            f.writelines(f"{name + ':':<20} {value}\n" for name, value in metadata.items())
            f.write("\n")
            header = " ".join(f"{key:<15}" for key in self.logs)
            f.write(header + "\n")

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
        params: dict[str, torch.Tensor],
        mask: torch.Tensor,
        chains: torch.Tensor,
        log_weights: torch.Tensor,
    ) -> None:
        """Saves the chains and the parameters of the model.

        Args:
            params (Dict[str, torch.Tensor]): Parameters of the model.
            mask (torch.Tensor): Mask of the model's coupling matrix representing the interaction graph.
            chains (torch.Tensor): Chains.
            log_weights (torch.Tensor): Log of the chain weights. Used for AIS.
        """
        save_params(fname=self.file_paths["params"], params=params, mask=mask, tokens=self.tokens)
        save_chains(
            fname=self.file_paths["chains"], chains=chains.argmax(dim=-1), tokens=self.tokens, log_weights=log_weights
        )
