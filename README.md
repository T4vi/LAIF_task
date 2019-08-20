# TPA-LSTM

Original Implementation of [''Temporal Pattern Attention for Multivariate Time Series Forecasting''](https://arxiv.org/abs/1809.04206).

## Requirements
Data requirements:
- We need to know how the data looks like and how users would create similar datasets. What's the format of annotation files, what is important.
- It needs to work if all data is placed in a single folder on disk

Inference requirements:
- We need to be able to load a set of saved weights (from arbitrary local path)
- We need to be able to initialize historical data
- Return predictions for the next K samples

Train requirements:
- We need to be able to load initialization weights (if none provided, start from random) before training
- We need to be able to specify an arbitrary number of epochs to run
- We need to be able to compute metrics on a validation dataset (provided in paper)
- We would like to have a certain granularity within this API:
    - prepare_train(train_split, test_split)
    - run_single_epoch()  # Uses train_split, outputs loss
    - run_validation()  # Uses test_split, outputs metrics
    - save_weights(path)
    - cleanup_train()

## Dependencies

* python3.6.6

You can check and install other dependencies in `requirements.txt`.

```
$ pip install -r requirements.txt
# to install TensorFlow, you can refer to https://www.tensorflow.org/install/
```

## Usage

The following example usage shows how to train and test a TPA-LSTM model on MuseData with settings used in this work.

### Training

```
$ python main.py --mode train \
    --attention_len 16 \
    --batch_size 32 \
    --data_set muse \
    --dropout 0.2 \
    --learning_rate 1e-5 \
    --model_dir ./models/model \
    --num_epochs 40 \
    --num_layers 3 \
    --num_units 338
```

### Testing

```
$ python main.py --mode test \
    --attention_len 16 \
    --batch_size 32 \
    --data_set muse \
    --dropout 0.2 \
    --learning_rate 1e-5 \
    --model_dir ./models/model \
    --num_epochs 40 \
    --num_layers 3 \
    --num_units 338
```
