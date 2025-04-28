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
* [ERA-5 data](https://mediatum.ub.tum.de/1524895) should be put into `datasets/` so that each data file can be located by `datasets/<dataset_name>.pt`. To transform raw data to `.pt` you can use [notebook](https://colab.research.google.com/drive/1tV7iFRAP3zIzCfX6uGjwrG7BjWvd4jHe?usp=sharing).


## Usage

To train and evaluate TS2Vec on a dataset, run the following command in the project directory (`ts2vec/`):

```train & evaluate
docker build . -t=ts2vec_image
docker run -it --rm --shm-size=256m --memory=64g --memory-swap=64g --cpuset-cpus=0-5 --gpus '"device=0,1"' -v $(pwd):/prj/main -p 8001:8001 --gpus 1 --name "your_name" ts2vec_image bash
cd main
/prj/ts2vec/bin/python train.py your_project_name --gpu 0
```

After training and evaluation, the weights of the trained encoder (`TS2Vec.pth`) can be found in `ts2vec/training` folder. 

**Downstream task:** To evaluate the weights of the pretrained encoder you can use [Downstream problem evaluation notebook](https://colab.research.google.com/drive/14XChfYfhBx9xq_f9UdVQEtVIq0Y4rdsS?usp=sharing). To use this jupyter notebook you need to store weights from the encoder and download them during the evaluation. For example, you can store the weights in the Google Drive and download them using `gdown` like it was done in the notebook **Downstream problem evaluation notebook**.
