from argparse import ArgumentParser

import json
import numpy as np
import os
import tensorflow.keras as K

from createdataset import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """


    #print(input_path, output_path, resources_path)
    #loading dictionary contiaining vocabularies of unigrams and bigrams
    with open(os.path.join(resources_path, 'vocabs/pku_msr_30.json')) as f:
        loaded_data = json.load(f)
    # processing the Input file into feature vectors given the dictionary
    A = CreateDataset(LabelFile_path = None,
                      InputFile_path = input_path,
                      PaddingSize = None,
                      set_type = 'TEST',
                      TrainingVocab = [loaded_data['UNIGRAM_DICT'],
                                       loaded_data['BIGRAM_DICT']])
    X_test_uni, X_test_bi = A.DateGenTest()

    padded = lambda x: np.array([xi+[1]*(max(map(len, x))-len(xi)) for xi in x])
    #keeping original lengths
    lengths = [len(i) for i in X_test_uni]
    #pad to max line length

    ##################
    # Model loading #
    #################
    loaded_model = K.models.load_model(os.path.join(resources_path, 'models/model_2019-04-24_12:12:00_-0500pku_msr.h5'))
    loaded_model.load_weights(os.path.join(resources_path, 'models/model_weights_2019-04-24_12:12:00_-0500pku_msr.h5'))
    loaded_model.compile(loss = 'categorical_crossentropy',
                  optimizer = K.optimizers.Adam(lr=0.0015, clipnorm=1., clipvalue=0.5),
                  metrics = ['acc', K.metrics.Precision()])

    ## PREDICTION
    predicted_one_hot = loaded_model.predict([padded(X_test_uni), padded(X_test_bi)])
    #de - one hot encode
    predicted = np.argmax(predicted_one_hot, axis=2)

    predictedBIES = []
    BIES = {0 : 'B', 1 : 'I', 2 : 'E', 3 : 'S'}
    for i in range(len(predicted)):
        #un-padding to original line lengths
        to_bies = list(predicted[i][:lengths[i]])
        #decoding into BIES format from numerical classes for each line
        BIES_letters = ''.join([BIES[i] for i in to_bies])
        #appending to list of lines in BIES
        predictedBIES.append(BIES_letters)

    #saving output file to be used in score function
    with open(output_path, 'w') as outfile:
        outfile.write('\n'.join(predictedBIES))
        print("Saved predictions at: {}".format(output_path))



if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
