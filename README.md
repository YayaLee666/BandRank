# BandRank

This repository contains implementation for paper [**"Ranking on Dynamic Graphs: An Effective and Robust Band-Pass Disentangled Approach"**](https://dl.acm.org/doi/10.1145/3696410.3714943).

## Experiment Reproduce

Code is written in Python and the proposed model is implemented using **Pytorch**.

To start training, use the following command:
```
python train.py --data_dir "your data dir" --data enron --prefix bandrank --verbose 1
```

Make sure to adjust the '--data_dir' path to where your datasets are stored.

## Dataset Access

- Reddit - http://snap.stanford.edu/jodie/reddit.csv

- Wikipedia - http://snap.stanford.edu/jodie/wikipedia.csv

- MOOC - http://snap.stanford.edu/jodie/mooc.csv

- LastFM - http://snap.stanford.edu/jodie/lastfm.csv

Preprocessed Enron and UCI datasets are taken from - https://github.com/snap-stanford/CAW

## Main Baseline Links

These following repositories represent the primary baseline models used in our experiments.

- TGRank - https://github.com/susheels/tgrank

- TATKC - https://github.com/ZJUT-DBLab/TATKC

- FreeDyG - https://github.com/Tianxzzz/FreeDyG

## Requirements

```
python >= 3.8
torch  == 2.0.1
scikit-learn == 1.2.2
pandas == 2.0.1
numpy == 1.24.3
tqdm == 4.63.0
```

