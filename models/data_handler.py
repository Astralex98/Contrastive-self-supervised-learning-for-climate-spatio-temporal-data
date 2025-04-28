import pathlib
from typing import Optional, Tuple, List
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import math
import torchvision


def create_celled_data(
    data_path,
    dataset_name,
    time_col: str = "time",
    event_col: str = "val",
    x_col: str = "x",
    y_col: str = "y",
):
    data_path = pathlib.Path(
        data_path,
        dataset_name,
    )

    df = pd.read_csv(data_path)
    df.sort_values(by=[time_col], inplace=True)
    df = df[[event_col, x_col, y_col, time_col]]

    indicies = range(df.shape[0])
    start_date = int(df[time_col][indicies[0]])
    finish_date = int(df[time_col][indicies[-1]])
    n_cells_hor = df[x_col].max() - df[x_col].min() + 1
    n_cells_ver = df[y_col].max() - df[y_col].min() + 1
    celled_data = torch.zeros([finish_date - start_date + 1, n_cells_hor, n_cells_ver])

    for i in tqdm.tqdm(indicies):
        x = int(df[x_col][i])
        y = int(df[y_col][i])
        celled_data[int(df[time_col][i]) - start_date, x, y] = df[event_col][i]

    return celled_data

# Spatial augmentation
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class Dataset_RNN(Dataset):
    """
    Simple Torch Dataset for many-to-many RNN
        celled_data: source of data,
        start_date: start date index,
        end_date: end date index,
        periods_forward: number of future periods for a target,
        history_length: number of past periods for an input,
        transforms: input data manipulations
    """

    def __init__(
        self,
        celled_data: torch.Tensor,
        celled_features_list: List[torch.Tensor],
        start_date: int,
        end_date: int,
        periods_forward: int,
        history_length: int,
        boundaries: Optional[List[None]],
        step_size: int,
        resize_shape: int,
        blur_kernel_size: int,
        pool_win_size: int,
        mode,
        is_our_loss
    ):
        self.data = celled_data[start_date:end_date, :, :]
        self.features = [
            feature[start_date:end_date, :, :]
            for feature in celled_features_list
        ]
        self.periods_forward = periods_forward
        self.history_length = history_length
        self.step_size = step_size
        self.mode = mode
        self.is_our_loss = is_our_loss
        self.target = self.data
        
        self.resize_shape = resize_shape
        self.blur_kernel_size = blur_kernel_size
        self.pool_win_size = pool_win_size

        # bins for pdsi
        self.boundaries = boundaries
        if self.mode == "classification":
            # 1 is for drought
            self.target = 1 - torch.bucketize(self.target, self.boundaries)

    def __len__(self):
        return len(self.data) - self.periods_forward - self.history_length * self.step_size

    def __getitem__(self, idx):
        
        # Spatial augmentations
        padding = math.floor(self.pool_win_size/2)
        
        # Отдельно определяем общий размер, к которому приводим все исходные карты признаков
        # resize = torchvision.transforms.Resize(size=(self.resize_shape, self.resize_shape))
        
        # Пространственные аугментации
        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.GaussianBlur(kernel_size = self.blur_kernel_size),
            
                        # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
                        AddGaussianNoise(0, 1),

                        torch.nn.AvgPool2d(self.pool_win_size, padding=padding, stride=1)
         ])
        
        # Data: [T, H, W] 
        # input_tensor = resize(self.data[idx : idx + self.history_length])
        input_tensor = self.data[idx : idx + (self.history_length * self.step_size) : self.step_size]
        # Input_tensor: [T, S, S], S = resize_shape
        
        # Input_tensor: [T, S, S]
        for feature in self.features:
              input_tensor = torch.stack((input_tensor, feature[idx : idx + self.history_length]), dim=1)
#             input_tensor = torch.stack(
#                 (input_tensor, resize(feature[idx : idx + self.history_length])), dim=1
#             )

        # Input_tensor: [T, R, S, S], R = regions_num

        # Применяем пространственные аугментации
        # Augs: [T, R, S, S]
        augs = transforms(input_tensor)
        
        target = self.target[
            idx + self.history_length : idx + self.history_length + self.periods_forward
        ]
        
        # input_tensor: [T, 2R, S, S]
        input_tensor = torch.cat((input_tensor, augs), dim = 1)
        
        return (
            input_tensor,
            target,
        )
    
class WeatherDataModule(pl.LightningDataModule):
    """LightningDataModule for Weather dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        mode: str = "regression",
        is_our_loss: bool = True,
        data_dir: str = "datasets",
        dataset_name: str = "dataset_name",
        left_border: int = 0,
        down_border: int = 0,
        right_border: int = 2000,
        up_border: int = 2500,
        time_col: str = "time",
        event_col: str = "value",
        x_col: str = "x",
        y_col: str = "y",
        train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1),
        periods_forward: int = 1,
        history_length: int = 1,
        data_start: int = 0,
        data_len: int = 100,
        feature_to_predict: str = "pdsi",
        num_of_additional_features: int = 0,
        additional_features: Optional[List[str]] = None,
        step_size: int = 1,
        boundaries: Optional[List[str]] = None,
        patch_size: int = 8,
        normalize: bool = False,
        batch_size: int = 64,
        resize_shape: int = 112,
        blur_kernel_size: int = 5,
        pool_win_size: int = 5,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.mode = mode
        self.is_our_loss = is_our_loss
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.left_border = left_border
        self.right_border = right_border
        self.down_border = down_border
        self.up_border = up_border
        self.h = 0
        self.w = 0
        self.time_col = time_col
        self.event_col = event_col
        self.x_col = x_col
        self.y_col = y_col

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_val_test_split = train_val_test_split
        self.periods_forward = periods_forward
        self.history_length = history_length
        self.data_start = data_start
        self.data_len = data_len
        self.feature_to_predict = feature_to_predict
        self.num_of_features = num_of_additional_features + 1
        self.additional_features = additional_features
        self.step_size = step_size
        self.boundaries = torch.Tensor(boundaries)

        self.patch_size = patch_size
        self.normalize = normalize

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.resize_shape = resize_shape
        self.blur_kernel_size = blur_kernel_size
        self.pool_win_size = pool_win_size

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # celled_data_path = pathlib.Path(self.data_dir, "celled", self.dataset_name)
        celled_data_path = pathlib.Path(self.dataset_name)

        if not celled_data_path.is_file():
            celled_data = create_celled_data(
                self.data_dir,
                self.dataset_name,
                self.time_col,
                self.event_col,
                self.x_col,
                self.y_col,
            )
            torch.save(celled_data, celled_data_path)

        data_dir_geo = self.dataset_name.split(self.feature_to_predict)[1]
        for feature in self.additional_features:
            celled_feature_path = pathlib.Path(
                self.data_dir, "celled", feature + data_dir_geo
            )
            if not celled_feature_path.is_file():
                celled_feature = create_celled_data(
                    self.data_dir,
                    feature + data_dir_geo,
                    self.time_col,
                    self.event_col,
                    self.x_col,
                    self.y_col,
                )
                torch.save(celled_feature, celled_feature_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            celled_data_path = pathlib.Path(self.data_dir, self.dataset_name)
            celled_data = torch.load(celled_data_path)

            # loading features
            celled_features_list = []

            for feature in self.additional_features:

                celled_feature_path = pathlib.Path(self.data_dir, feature)
                celled_feature = torch.load(celled_feature_path)

                celled_features_list.append(celled_feature)

            train_start = 0
            train_end = int(self.train_val_test_split[0] * celled_data.shape[0])

            self.data_train = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_start,
                train_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.step_size,
                self.resize_shape,
                self.blur_kernel_size,
                self.pool_win_size,
                self.mode,
                self.is_our_loss,
            )
            # valid_end = int(
            #     (self.train_val_test_split[0] + self.train_val_test_split[1])
            #     * celled_data.shape[0]
            # )
            valid_end = celled_data.shape[0]
            self.data_val = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_end - self.history_length,
                valid_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.step_size,
                self.resize_shape,
                self.blur_kernel_size,
                self.pool_win_size,
                self.mode,
                self.is_our_loss,
            )
            test_end = celled_data.shape[0]
            self.data_test = Dataset_RNN(
                celled_data,
                celled_features_list,
                train_end - self.history_length,
                test_end,
                self.periods_forward,
                self.history_length,
                self.boundaries,
                self.step_size,
                self.resize_shape,
                self.blur_kernel_size,
                self.pool_win_size,
                self.mode,
                self.is_our_loss,
            )
            print(f"train dataset shape {len(self.data_train)}")
            print(f"val dataset shape {len(self.data_val)}")
            print(f"test dataset shape {len(self.data_test)}")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )