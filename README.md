# Chinese-LSTM
Chinese Word Segmentation with Bi-LSTMs

These are to be run inside of code folder
To get the data preprocessed
```
$ python3 preprocess.py [Dataset type]
Either 'training','dev'
```
To run the model
```
$ python3 model.py
```
To predict
```
$ python3 predict.py input_path output_path resources_path
```
To Score
```
$ python3 score.py prediction_file gold_file
```


