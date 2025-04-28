# Spatio-temporal TS2Vec

This repository contains the generalization of the spatio-temporal TS2Vec from time series to the spatio-temporal domain.

## Requirements

The recommended requirements for TS2Vec are specified as follows:
* Bottleneck
* torch==1.10.1
* scipy==1.6.1
* numpy==1.19.2
* statsmodels==0.12.2
* pandas==1.0.1
* scikit_learn==0.24.2
* pytorch-lightning==1.9.1
* torchmetrics
* torchvision
* omegaconf

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [Drought data](https://github.com/Astralex98/long-term-drought-prediction) should be put into `datasets/` so that each data file can be located by `datasets/<dataset_name>.csv`.
* [ERA-5 data](https://mediatum.ub.tum.de/1524895) should be put into `datasets/` so that each data file can be located by `datasets/<dataset_name>.pt`. To transform raw data to .pt you can use [notebook](https://colab.research.google.com/drive/1tV7iFRAP3zIzCfX6uGjwrG7BjWvd4jHe?usp=sharing).


## Usage

To train and evaluate TS2Vec on a dataset, run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 

**Scripts:** The scripts for reproduction are provided in `scripts/` folder.


## Code Example

```python
from ts2vec import TS2Vec
import datautils

# Load the ECG200 dataset from UCR archive
train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

# Train a TS2Vec model
model = TS2Vec(
    input_dims=1,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)

# Compute timestamp-level representations for test set
test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims

# Compute instance-level representations for test set
test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims

# Sliding inference for test set
test_repr = model.encode(
    test_data,
    causal=True,
    sliding_length=1,
    sliding_padding=50
)  # n_instances x n_timestamps x output_dims
# (The timestamp t's representation vector is computed using the observations located in [t-50, t])
```
