# TPA-LSTM

Original Implementation of [''Temporal Pattern Attention for Multivariate Time Series Forecasting''](https://arxiv.org/abs/1809.04206).

[PDF](https://arxiv.org/pdf/1809.04206.pdf)

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