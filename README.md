# BandRank


## Dataset Access

\begin{itemize}
\item Reddit - http://snap.stanford.edu/jodie/reddit.csv

\item Wikipedia - http://snap.stanford.edu/jodie/wikipedia.csv

\item MOOC - http://snap.stanford.edu/jodie/mooc.csv

\item LastFM - http://snap.stanford.edu/jodie/lastfm.csv
\end{itemize}

Preprocessed Enron and UCI datasets are taken from - https://github.com/snap-stanford/CAW

## Main Baseline Links

TGRank - https://github.com/susheels/tgrank

TATKC - https://github.com/ZJUT-DBLab/TATKC

FreeDyG - https://github.com/Tianxzzz/FreeDyG

## Experiment Reproduce

This repository contains implementation for paper "Ranking on Dynamic Graphs: An Effective and Robust Band-Pass Disentangled Approach"

Code is written in Python and the proposed model is implemented using **Pytorch**.

To start training, use the following command:
```
python train.py --data_dir "your data dir" --data enron --prefix tgrank-listwise --verbose 1
```

Make sure to adjust the --data_dir path to where your datasets are stored.

## Requirements

```
python >= 3.8
torch  == 2.0.1
scikit-learn == 1.2.2
pandas == 2.0.1
numpy == 1.24.3
tqdm == 4.63.0
```
