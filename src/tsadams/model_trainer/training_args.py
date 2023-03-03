from dataclasses import asdict, dataclass, field

#TODO: early stopping
#TODO: scheduler with epochs


@dataclass
class TrainingArguments:
    """
    Parameters
    ----------
    output_dir: str
        Directory to store model checkpoints.
    overwrite_output_dir: bool
        Overwrite current content in output directory.
    train_batch_size: int
        Batch size for training.
    eval_batch_size: int
        Batch size for evaluation.
    learning_rate: float
        Initial learning rate for training.
    weight_decay: float
        Weight decay.
    adam_beta1: float
        Beta1 parameter for Adam optimizer.
    adam_beta2: float
        Beta2 parameter for Adam optimizer.
    adam_epsilon: float
        Epslion parameter for Adam optimizer.
    max_grad_norm: float
        Max gradient norm.
    max_epochs: int
        Max epochs during training.
    max_steps: int
        Max steps during training. If speficied, overrides max_epochs.
    lr_scheduler_steps: int
        Steps for learning rate scheduler. If training with max_epochs, one step
        is one epoch, if training with max_steps, one step is one step.
    lr_scheduler_gamma: float
        Gamma for learning rate scheduler.
    eval_freq: int
        Number of epochs or steps between evaluations for logging.
    seed: int
        Random seed for sampling. Default -1, will not set seed.
    """
    output_dir: str = field(metadata={"help": "Output directory"})
    overwrite_output_dir: bool = field(default=False, metadata={"help":"Overwrite the current content of the output directory."})
    train_batch_size: int = field(default=1, metadata={"help": "Training batch size."})
    eval_batch_size: int = field(default=1, metadata={"help": "Eval batch size."})
    learning_rate: float = field(default=0.001, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for Adam optimizer"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    max_epochs: int = field(default=1, metadata={"help": "Number of epochs for training."})
    max_steps: int = field(default=-1, metadata={"help": "Number of steps for training."})
    lr_scheduler_steps: int = field(default=-1, metadata={"help": "Number of steps for scheduler. If -1, no scheduler steps."})
    lr_scheduler_gamma: float = field(default=0.5, metadata={"help": "Gamma for scheduler steps."})
    eval_freq: int = field(default=1, metadata={"help": "Number of epochs or steps between evaluations for logging."})
    seed: int = field(default=-1, metadata={"help": "Numpy random seed for sampling."})

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}
        attrs_as_str = [f"\t{k}={v},\n" for k, v in sorted(self_as_dict.items())]

        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


