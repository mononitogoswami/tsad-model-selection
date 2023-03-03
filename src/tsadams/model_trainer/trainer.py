
import time
from turtle import forward
import numpy as np
import torch as t
import tqdm

from typing import List, Tuple, Optional

from ..datasets.dataset import Dataset
from ..loaders.loader import Loader
from ..models.base_model import PyMADModel
from .training_args import TrainingArguments
from ..utils.utils import de_unfold

#TODO: callbacks, to replace trajectories
#TODO: checkpoints
#TODO: device
#TODO: tqdm

class Trainer(object):
    def __init__(self,
                 model: PyMADModel,
                 args: TrainingArguments,
                 train_dataset: Dataset,
                 eval_dataset: Optional[Dataset] = None,
                 optimizers: Tuple[t.optim.Optimizer, t.optim.lr_scheduler.LambdaLR] = (None, None),
                 verbose: bool = False) -> 'Loader':
        """
        Parameters
        ----------
        model: nn.module
            Model to be trained. Must have training_step method.
        args: TrainingArguments
            Training arguments.
        train_dataset: Dataset
            Training dataset.
        eval_dataset: Dataset
            Evaluation dataset.
        optimizers: Tuple[t.optim.Optimizer, t.optim.lr_scheduler.LambdaLR]
            Optional torch optimizer and lr scheduler for training.
            If not provided, will use Adam optimizer
        verbose:
            Boolean for printing details.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.verbose = verbose

        if self.args.max_steps > 0:
            self.train_strategy = 'step'
        else:
            self.train_strategy = 'epoch'

        if verbose:
            print(42*'-')
            print(self)
            print(42*'-')

    def _instantiate_loaders(self):
        if self.verbose:
            print('Instantiating train loader...')

        sample_with_replace = True if self.train_strategy=='step' else False
        train_dataloader = Loader(dataset=self.train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  window_size=self.model.window_size,
                                  window_step=self.model.window_step,
                                  shuffle=True,
                                  padding_type='None',
                                  sample_with_replace=sample_with_replace,
                                  verbose=self.verbose, 
                                  mask_position='None', 
                                  n_masked_timesteps=0)

        if self.eval_dataset:
            if self.verbose:
                print('Instantiating eval loader...')
            eval_dataloader = Loader(dataset=self.eval_dataset,
                                     batch_size=self.args.eval_batch_size,
                                     window_size=self.model.window_size,
                                     window_step=self.model.window_step,
                                     shuffle=False,
                                     padding_type='None',
                                     sample_with_replace=False,
                                     verbose=self.verbose,
                                     mask_position='None', 
                                     n_masked_timesteps=0)
        else:
            eval_dataloader = None

        return train_dataloader, eval_dataloader

    def _instantiate_optimizers(self):
        optimizer = t.optim.Adam(self.model.parameters(),
                                lr=self.args.learning_rate,
                                betas=[self.args.adam_beta1, self.args.adam_beta2])

        if self.args.lr_scheduler_steps>0:
            lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_scheduler_steps,
                                                        gamma=self.args.lr_scheduler_gamma)
        else:
            lr_scheduler = None
            
        return optimizer, lr_scheduler

    def train(self):

        if self.model.training_type == 'sgd':
            self.train_sgd()

        elif self.model.training_type == 'direct':
            self.train_direct()

    def train_direct(self):
        # Set seed for Dataloader
        if self.args.seed>0:
                    np.random.seed(self.args.seed)

        # Dataloaders (TODO: think if we can move this to model.fit)
        train_dataloader, eval_dataloader = self._instantiate_loaders()

        # Fit model
        self.model.fit(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)


    def train_sgd(self):
        # Set seed for Dataloader
        if self.args.seed>0:
            np.random.seed(self.args.seed)       

        # Dataloaders
        train_dataloader, eval_dataloader = self._instantiate_loaders()

        # Optimizers
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self._instantiate_optimizers()

        if self.train_strategy=='epoch':
            n_train_epochs = self.args.max_epochs
        else:
            n_train_epochs = int(np.ceil(self.args.max_steps/train_dataloader.n_batch_in_epochs))

        progress_bar = tqdm.tqdm(total=n_train_epochs)

        # Training loop
        total_steps = 0
        tr_loss = 0
        counter_loss = 0

        start_time = time.time()
        self.trajectories = {'epoch': [], 'iteration':[], 'train_loss':[], 'eval_loss':[], 'time': []} #TODO: replace with callback and logger

        for epoch in range(n_train_epochs):

            for step, batch in enumerate(train_dataloader):

                # Training step
                self.model.train()
                loss = self.model.training_step(input=batch)

                if isinstance(loss, (list, tuple)):
                    if not np.isnan(float(loss[0])):
                        loss[0].backward(retain_graph=True)
                        loss[1].backward()
                else:
                    if not np.isnan(float(loss)):
                        loss.backward()
                    
                if hasattr(self.optimizer, "clip_grad_norm"):
                    # Gradient clipping specific for optimizer
                    self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                elif hasattr(self.model, "clip_grad_norm_"):
                    # Gradient clipping specific for model
                    self.model.clip_grad_norm_(self.args.max_grad_norm)
                else:
                    # Normal clipping otherwise
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if isinstance(loss, (list, tuple)):
                    tr_loss += loss[0].detach().cpu().numpy()
                else:
                    tr_loss += loss.detach().cpu().numpy()

                counter_loss += 1

                # Break conditions
                total_steps += 1
                if (self.train_strategy=='step') and (total_steps > self.args.max_steps):
                    break

                if isinstance(loss, (list, tuple)):
                    if np.isnan(float(loss[0])):
                        break
                else:
                    if np.isnan(float(loss)):
                        break

                # Evaluation, logger and printing
                print_step = (self.train_strategy=='step') and (total_steps % self.args.eval_freq==0)
                print_epoch = (self.train_strategy=='epoch') and \
                              (step == (train_dataloader.n_batch_in_epochs-1)) and \
                              (epoch % self.args.eval_freq == 0)

                if print_step or print_epoch:
                    self.trajectories['epoch'].append(epoch)
                    self.trajectories['iteration'].append(total_steps)
                    self.trajectories['train_loss'].append(tr_loss/counter_loss)
                    self.trajectories['time'].append(time.time()-start_time)
                    
                    if self.eval_dataset is not None:
                        self.model.eval()
                        eval_loss = self.evaluate_model(loader=eval_dataloader).detach().cpu().numpy()
                        self.trajectories['eval_loss'].append(eval_loss)

                    if self.verbose:
                        display_string = "Epoch: {}, Step: {}, Time: {:3.3f}, Train: {:.5f}".format(epoch,
                                                                                                    total_steps,
                                                                                                    time.time()-start_time,
                                                                                                    tr_loss/counter_loss)
                        if self.eval_dataset:
                            display_string += ", Eval: {:.5f}".format(eval_loss)

                        print(display_string)

                    tr_loss, counter_loss = 0, 0
            
            progress_bar.update(1)

    def evaluate_model(self, loader):
        eval_loss = 0

        for step, batch in enumerate(loader):
            loss = self.model.eval_step(x=batch)
            eval_loss += loss

        return eval_loss/(step+1)

    def predict(self, dataset: Dataset, warmup_dataset: Dataset=None, batch_size: int=1, padding_type: str='right',
                return_detail: bool=False, verbose: bool=False, mask_position: str='None', 
                n_masked_timesteps: int=0):

        assert padding_type in ['None', 'left', 'right'], f'padding_type {padding_type} not recongnized.'
        assert mask_position in ['None', 'mid', 'right'], f'mask_position {mask_position} not recongnized.'

        if self.model.window_size < 0:
            padding_type = 'None'
        
        self.model.eval()

        # Loop between entities of dataseta
        anomaly_scores = {}
        for i, entity in enumerate(dataset):
            # Instantiate Loader
            dataloader = Loader(dataset=entity,
                                batch_size=batch_size,
                                window_size=self.model.window_size,
                                window_step=self.model.window_step,
                                shuffle=False,
                                padding_type=padding_type,
                                sample_with_replace=False,
                                verbose=verbose,
                                mask_position=mask_position, 
                                n_masked_timesteps=n_masked_timesteps)

            # Loop on entity
            entity_scores = np.zeros((len(dataloader), dataset.n_features, dataloader.window_size))
            step = 0

            for batch in dataloader:
                batch_anomaly_score = self.model.window_anomaly_score(input=batch, return_detail=True).detach().cpu().numpy()
                entity_scores[ step : (step+len(batch_anomaly_score) )] = batch_anomaly_score
                step += len(batch_anomaly_score)

            # Final scores
            entity_scores = self.model.final_anomaly_score(input=entity_scores, return_detail=return_detail)
            entity_scores = entity_scores.detach().cpu().numpy()

            # Remove extra padding
            if padding_type == 'right':
                if return_detail: entity_scores = entity_scores[:, :-dataloader.padding_size]
                else: entity_scores = entity_scores[:-dataloader.padding_size]
            elif padding_type == 'left':
                if return_detail: entity_scores = entity_scores[:, dataloader.padding_size:]
                else: entity_scores = entity_scores[dataloader.padding_size:]
            
            assert len(entity_scores) == entity.n_time, f'Length of scores {len(entity_scores)} is not equal to length of entity {entity.n_time}'

            anomaly_scores[entity.name] = entity_scores
            
        return anomaly_scores

    def reconstruct(self, dataset: Dataset, warmup_dataset: Dataset=None, batch_size: int=1,
                    padding_type: str='right', verbose: bool=False, mask_position: str='None', 
                    n_masked_timesteps: int=0):
        
        assert padding_type in ['None', 'left', 'right'], 'padding_type not recongnized.'
        assert mask_position in ['None', 'mid', 'right'], 'mask_position not recongnized.'

        self.model.eval()

        # Loop between entities of dataset
        actual = {}
        reconstruction = {}
        for i, entity in enumerate(dataset):
            if verbose:
                print(entity.name)

            # Instantiate Loader
            dataloader = Loader(dataset=entity,
                                batch_size=batch_size,
                                window_size=self.model.window_size,
                                window_step=self.model.window_step,
                                shuffle=False,
                                padding_type=padding_type,
                                sample_with_replace=False,
                                verbose=verbose, 
                                mask_position=mask_position, 
                                n_masked_timesteps=n_masked_timesteps)

            # Loop on entity
            actual_values = t.zeros((len(dataloader), dataset.n_features, dataloader.window_size))
            entity_reconstruction = t.zeros((len(dataloader), dataset.n_features, dataloader.window_size))
            step = 0
            for batch in dataloader:
                Y, Y_hat, *_ = self.model(input=batch)
                actual_values[ step : (step+len(Y_hat) )] = Y
                entity_reconstruction[ step : (step+len(Y_hat) )] = Y_hat
                step += len(Y_hat)

            actual_values = de_unfold(windows=actual_values, window_step=dataloader.window_step)
            actual_values = actual_values.cpu().detach().numpy()

            entity_reconstruction = de_unfold(windows=entity_reconstruction, window_step=dataloader.window_step)
            entity_reconstruction = entity_reconstruction.cpu().detach().numpy()

            # Remove extra padding
            if padding_type == 'right':
                actual_values = actual_values[:, :-dataloader.padding_size]
                entity_reconstruction = entity_reconstruction[:, :-dataloader.padding_size]
            elif padding_type == 'left':
                actual_values = actual_values[:, dataloader.padding_size:]
                entity_reconstruction = entity_reconstruction[:, dataloader.padding_size:]
            
            assert entity_reconstruction.shape[-1] == entity.n_time, f'Length of reconstruction {len(entity_reconstruction)} is \
                                                                       not equal to length of entity {entity.n_time}'

            actual[entity.name] = actual_values  
            reconstruction[entity.name] = entity_reconstruction  
        
        return actual, reconstruction

    def __str__(self):
        info_dict = {'model': self.model.__class__.__name__,
                     'device': self.model.device,
                     'train_dataset': self.train_dataset.name,
                     'eval_dataset': self.eval_dataset.name if self.eval_dataset is not None else None,
                     'train_strategy': self.train_strategy,
                     'eval_freq': self.args.eval_freq,
                     'optimizer': self.optimizer.__class__.__name__ if self.optimizer is not None else None}

        info_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in info_dict.items()}
        attrs_as_str = [f"\t{k}={v},\n" for k, v in info_dict.items()]

        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
