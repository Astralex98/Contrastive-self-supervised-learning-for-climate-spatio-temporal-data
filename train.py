import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
# from ts2vec import TS2Vec
from ts2vec import SpatioTemporalTS2Vec
from our_ts2vec import OurSpatioTemporalTS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout, get_free_gpu

import pickle
import pathlib
import datetime
from omegaconf import OmegaConf

import pytorch_lightning as pl
from models import WeatherDataModule
from models.losses import TS2Vec_loss
from models.losses import Our_TS2Vec_loss

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
#     parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    

######################################### Models hyperparameters #########################################
    
    # Read config list with hyperparameters
    oc_file = "cfg.yaml"
    oc_from_file = OmegaConf.load(open(oc_file, "r"))
    oc = OmegaConf.create()
    oc = OmegaConf.merge(oc, oc_from_file)
    
    # this allows to use tuples in cfg.yaml
    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("tuple", resolve_tuple)
    
    # Data
    data_dir = oc.data.data_dir
    dataset_name = oc.data.name
    add_features = oc.data.add_features
    hist_length = oc.data.history
    is_our_loss = oc.data.is_our_loss
    step_size = oc.data.step_size
    
    # Optimization
    device = oc.optim.device
    batch_size = oc.optim.batch_size
    seed = oc.optim.seed
    lr = oc.optim.lr
    n_epochs = oc.optim.max_epochs
    pat = oc.optim.patience
    alpha = oc.optim.alpha
    
    # Model
    hidden_dims = oc.model.hidden_dims
    cell_output_dims = oc.model.cell_output_dims
    region_output_dims = oc.model.region_output_dims
    max_train_length = None
    kernel_size = oc.model.kernel_size
    conv1d_kernel_size = oc.model.conv1d_kernel_size
    num_layers = oc.model.num_layers
    resize_shape = oc.model.resize_shape
    blur_kernel_size = oc.model.blur_kernel_size
    pool_win_size = oc.model.pool_win_size
    
    
    
    config = dict(
        batch_size=batch_size,
        lr=lr,
        cell_output_dims=cell_output_dims,
        max_train_length=max_train_length
    )

#########################################################################################################

    device = init_dl_program(args.gpu, seed=seed, max_threads=None)
    # device = 'cpu'
    
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Load data
    celled_data_path = pathlib.Path(data_dir, dataset_name)
    data = torch.load(celled_data_path)
    
    data_handler = WeatherDataModule(mode = "ssl",
                                     is_our_loss = is_our_loss,
                                 data_dir = data_dir,
                                 dataset_name = dataset_name,
                                 right_border = data.shape[1],
                                 up_border = data.shape[2],
                                 train_val_test_split = (0.7, 0.3, 0),
                                 periods_forward = 1,
                                 history_length = hist_length,
                                 boundaries = [-2],
                                 batch_size = batch_size,
                                 additional_features = add_features,
                                 step_size = step_size,
                                 resize_shape = resize_shape,
                                 blur_kernel_size = blur_kernel_size,
                                 pool_win_size = pool_win_size)
    
    data_handler.setup()
    
    # Criterion and ts2vec module
    criterion = None
    # ts2vec = None
    
    if (is_our_loss == True):
        criterion = Our_TS2Vec_loss(temporal_unit=0, alpha = alpha)
    else:
        criterion = TS2Vec_loss(temporal_unit=0, alpha = alpha)
        
    ts2vec = OurSpatioTemporalTS2Vec(
            input_dims = len(add_features) + 1,
            hist_length =  hist_length,
            data_handler = data_handler,
            criterion = criterion,
            hidden_dims = hidden_dims,
            cell_output_dims = cell_output_dims,
            region_output_dims = region_output_dims,
            kernel_size = kernel_size,
            conv1d_kernel_size = conv1d_kernel_size,
            num_layers = num_layers,
            device = device,
            lr = lr,
            batch_size = batch_size,
            resize_shape = resize_shape,
            max_train_length = max_train_length)
        
        
#         ts2vec = SpatioTemporalTS2Vec(
#             input_dims = len(add_features) + 1,
#             hist_length =  hist_length,
#             data_handler = data_handler,
#             criterion = criterion,
#             hidden_dims = hidden_dims,
#             cell_output_dims = cell_output_dims,
#             region_output_dims = region_output_dims,
#             kernel_size = kernel_size,
#             conv1d_kernel_size = conv1d_kernel_size,
#             num_layers = num_layers,
#             device = device,
#             lr = lr,
#             batch_size = batch_size,
#             resize_shape = resize_shape,
#             max_train_length = max_train_length)
    
    
    current_time = datetime.datetime.now().strftime("%m%d%Y_%H:%M:%S")
    experiment_name = 'TS2Vec' + '_' + current_time

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'training/{experiment_name}',
        filename='{epoch:02d}-{val_loss:.3f}',
        save_weights_only = True,
        mode='min')
    
    earlystopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor = 'val_loss', patience = pat)
    
    trainer = pl.Trainer(max_epochs=n_epochs, devices = [0], accelerator="gpu", 
                         enable_progress_bar=False, benchmark=True, check_val_every_n_epoch=1,
                         callbacks=[earlystopping_callback])
    
    trainer.fit(model=ts2vec)
    
    path_to_model_save = os.path.join("training", "TS2Vec.pth")
    torch.save(ts2vec.state_dict(), path_to_model_save)
    
    train_losses = ts2vec.get_train_losses()
    val_losses = ts2vec.get_val_losses()
    
    losses_dict = {"train_loss" : train_losses, "val_losses" : val_losses}
        
    with open('losses.pkl', 'wb') as f:
        pickle.dump(losses_dict, f)