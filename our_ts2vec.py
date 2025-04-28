import pathlib
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from models import TSEncoder
from models import SpatioTSEncoder
from models import WeatherDataModule

# from models.losses import hierarchical_contrastive_loss

from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
import pytorch_lightning as pl
    

class OurSpatioTemporalTS2Vec(pl.LightningModule):
    '''The TS2Vec model'''

    def __init__(
        self,
        input_dims,
        hist_length,
        data_handler,
        criterion,
        hidden_dims,
        cell_output_dims,
        region_output_dims,
        kernel_size,
        conv1d_kernel_size,
        num_layers,
        lr,
        batch_size,
        resize_shape,
        device='cuda',
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.

        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''

        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self.input_dims = input_dims

        # self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self._net = SpatioTSEncoder(input_dims=input_dims, hist_length=hist_length, hidden_dims=hidden_dims, cell_output_dims=cell_output_dims, region_output_dims=region_output_dims, kernel_size=kernel_size, conv1d_kernel_size = conv1d_kernel_size, num_layers = num_layers, device = device).to(self.device)

        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.data_handler = data_handler
        self.criterion = criterion

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.train_losses = []
        self.val_losses = []

        self.n_epochs = 0
        self.n_iters = 0

    def training_step(self, batch, batch_idx):
        
        # x: [B, T, 2R, H, W]
        # 2R - batch already contains both original and augmented versions of regions
        x = batch[0]
        
        # Time augmentations
#         ts_l = x.size(1)
#         crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
#         crop_left = np.random.randint(ts_l - crop_l + 1)
#         crop_right = crop_left + crop_l
#         crop_eleft = np.random.randint(crop_left + 1)
#         crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
#         crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
        
        orig_data = x[:, :, :self.input_dims, :, :]
        augs = x[:, :, self.input_dims:, :, :]
        
        # input for self._net: [B, T, R, H, W]
        z_orig = self._net(orig_data)
        # z_orig: [R, T, Co]
        
        # input for self._net: [B, T, R, H, W]
        z_augs = self._net(augs)
        # z_augs: [R, T, Co]

        train_loss = self.criterion(z_orig, z_augs)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.net.update_parameters(self._net)

        return train_loss

    def validation_step(self, batch, batch_idx):
        
        # x: [B, T, 2R, H, W]
        # 2R - batch already contains both original and augmented versions of regions
        x = batch[0]
        
        orig_data = x[:, :, :self.input_dims, :, :]
        augs = x[:, :, self.input_dims:, :, :]
        
        # input for self._net: [B, T, R, H, W]
        z_orig = self._net(orig_data)
        # z_orig: [R, T, Co]
        
        # input for self._net: [B, T, R, H, W]
        z_augs = self._net(augs)
        # z_augs: [R, T, Co]

        val_loss = self.criterion(z_orig, z_augs)
        self.log("val_loss", val_loss, prog_bar=False)

        return {"val_loss": val_loss}

    def training_epoch_end(self, outputs : list) -> None:
        """Print loss at the end of the training epoch

        :param outputs: list of losses, collected from each batch
        """

        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print("[Epoch %d] Train: loss=%.3f"% (self.current_epoch + 1, loss))
        self.train_losses.append(float(loss))

    def validation_epoch_end(self, outputs : list) -> None:
        """Print loss at the end of the validation epoch

        :param outputs: list of losses, collected from each batch
        """
        loss = sum(output['val_loss'] for output in outputs) / len(outputs)
        print("[Epoch %d] Val: loss=%.3f"% (self.current_epoch + 1, loss))
        self.val_losses.append(float(loss))

    def configure_optimizers(self):
        """Set parameters for optimizer.

        :return: optimizer
        """
        optimizer =  torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}

    def train_dataloader(self):
        """Set DataLoader for training data.

        :return: DataLoader for training data
        """
        train_dataloader = self.data_handler.train_dataloader()
        return train_dataloader

    def val_dataloader(self):
        """Set DataLoader for validation data.

        :return: DataLoader for validating data
        """
        val_dataloader = self.data_handler.val_dataloader()
        return val_dataloader
    
    def get_train_losses(self):
        return self.train_losses
    
    def get_val_losses(self):
        return self.val_losses