# Chinese-LSTM
Chinese Word Segmentation with Bi-LSTMs

## 1. Download the dataset
## 2. Initial preprocessing of dataset:
Inludes:
1. Translation from Traditional Chinese datasets to Simplified Chinese using Hanzi Convert. Creates a basis of an original file.
2. Creating an Input File from the translated original file --> no spaces
3. Label File, line by line using the BIES format

```
$ python3 preprocess.py [Dataset type]
Either 'training','dev','test'
```
## 3. Creating dataset of each subset 'msr','cityu','as','pku':
