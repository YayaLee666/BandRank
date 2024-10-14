# BandRank


## Dataset Links
 
Reddit - http://snap.stanford.edu/jodie/reddit.csv

Wikipedia - http://snap.stanford.edu/jodie/wikipedia.csv

MOOC - http://snap.stanford.edu/jodie/mooc.csv

LastFM - http://snap.stanford.edu/jodie/lastfm.csv

Preprocessed Enron and UCI datasets are taken from - https://github.com/snap-stanford/CAW

## Main Baseline Links

TGRank - https://github.com/susheels/tgrank

TATKC - https://github.com/ZJUT-DBLab/TATKC

FreeDyG - https://github.com/Tianxzzz/FreeDyG



This repository contains implementation for paper "Ranking on Dynamic Graphs: An Effective and Robust Band-Pass Disentangled Approach"

Code is written in Python and the proposed model is implemented using Pytorch.

Running the code:
```
python train.py --data_dir "your data dir" --data enron --prefix tgrank-listwise --verbose 1
```
