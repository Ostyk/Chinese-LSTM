import os
import numpy as np
from typing import Tuple, List, Dict

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator

class CreateDataset(object):
    '''makes feed files of combined unigrams and bigrams'''
    def __init__(self, LabelFile_path, InputFile_path, PaddingSize, set_type, TrainingVocab):
        self.Label_File = LabelFile_path
        self.Input_File = InputFile_path
        self.PaddingSize = PaddingSize
        self.set_type = set_type
        self.TrainingVocab = TrainingVocab
    
    def DateGen(self):
        '''creates labels from the label file'''
        
        features_vectors, word_to_index = self.FeatureGenerator() 
        labels = self.BIESToNumerical()
        #Optimal_Line_Length = int(np.mean([len(i) for i in features_vectors])) #length of longest line
        Optimal_Line_Length = self.PaddingSize
        
        #print("MAXLEN: {}".format(Optimal_Line_Length)) 
        padded_labels = pad_sequences(labels, truncating='pre', padding='post', maxlen = Optimal_Line_Length)
        
        y =  K.utils.to_categorical(padded_labels, num_classes=4)
        X = pad_sequences(features_vectors, truncating='pre', padding='post', maxlen = Optimal_Line_Length)
        
        info = {"MAXLEN": Optimal_Line_Length,
                "VocabSize": len(word_to_index)}
        return X, y, info, word_to_index
    
    def BIESToNumerical(self):
        '''Converts Label File from BIES encoding to numerical classes'''
        BIES = {'B' : 0, 'I' : 1, 'E' : 2, 'S' : 3}
        #numerical BIES class given to a line 
        labels = []
        with open(self.Label_File, 'r', encoding ='utf8') as f1:
            count = 0
            for line in f1:
                l = line.rstrip()
                labels.append([BIES[i] for i in l])
        return labels
    
    def FeatureGenerator(self):
        '''Generates features based on '''
        features_vectors = []
        if self.set_type == 'training':
            word_to_index = self.generateVocab()
        else:
            word_to_index = self.TrainingVocab

        with open(self.Input_File, 'r', encoding ='utf8') as f1:
            for line in f1:
                l = line.rstrip()
                grams = self.split_into_grams(l, 'uni_grams') + self.split_into_grams(l,'bi_grams')
                #difference is creating by grams line by line
                features_vectors.append([word_to_index.get(i, 0) for i in grams])
        return features_vectors, word_to_index
    
    def generateVocab(self):
        '''
        Generates vocabulary based on file
        args: Inputfile, returns: word_to_index dict
        '''
        big_line = ''
        with open(self.Input_File, 'r', encoding ='utf8') as f1:
            for line in f1:
                big_line+=line.rstrip()
        final = self.split_into_grams(big_line, type_ = 'bi_grams') + self.split_into_grams(big_line, type_ = 'uni_grams')
        vocab = set(final)
        word_to_index = dict()
        word_to_index['<UNK>'] = 0
        word_to_index.update({value:key+1 for key,value in enumerate(vocab)})
        
        
        return word_to_index
        
    
    @staticmethod
    def split_into_grams(sentence: str, type_ = 'uni_grams') -> List[str]:
        """
        :param sentence Sentence as str
        :type_: uni_grams or _bigrams
        :return bigrams List of unigrams or bigrams
        """
        n = 1 if type_ == 'uni_grams' else 2
        grams = []
        for i in range(len(sentence)-1):
            gram = sentence[i:i+n]
            grams.append(gram)
        return grams
    
def ChooseDataset(set_type, subset):
    '''returns paths to Label and Input file for a specific dataset
    args: set_type
    return: Label_file, Input_file
    '''
    datasets = {"training":'../icwb2-data/training',
                "dev":'../icwb2-data/gold',
                "testing":'../icwb2-data/testing'}
        
    def get_file_names(path, type_='LabelFile'):
        x = []
        dev = True if path.split("/")[-1] == 'gold' else False #checks for dev
        for i in os.listdir(path):
            if dev and i.split("_")[1][:4]=='test': #eliminates 'training' from 'gold' 
                if os.path.splitext(i)[0].split("_")[-1] == type_:
                    x.append(os.path.join(path, i))
            elif not dev:
                if os.path.splitext(i)[0].split("_")[-1] == type_:
                    x.append(os.path.join(path, i))
        return x
    
    Label_files = get_file_names(path = datasets[set_type], type_ = 'LabelFile')
    Input_files = get_file_names(path = datasets[set_type], type_ = 'InputFile')
    names = ['msr','cityu','as','pku']
    choose = lambda i: i.split(".utf8")[0].split('/')[-1].split("_")[0]
    e, r = False, False
    chosen = False
    while not chosen:
        #x = input("Choose from the following: {}".format(names))
        x = subset
        for i in range(len(Label_files)):
            if choose(Input_files[i]) == x: 
                Input_file = Input_files[i]
                e = True
            if choose(Label_files[i]) == x:
                Label_file = Label_files[i]
                r = True
            if e and r:
                chosen = True
            
    return Label_file, Input_file

    