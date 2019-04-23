import os
import numpy as np
from typing import Tuple, List, Dict

import tensorflow.keras as K
from tensorflow.keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator

def ChooseDataset(set_type, subset):
    '''returns paths to Label and Input file for a specific dataset
    args: set_type - List, if len(List) > 1 then it creates a new file based on a concatenation of the given files
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
    
    def concat_subsets(filenames, subset, f_type):
        current = os.path.join(*filenames[0].split('/')[:-1], '_'.join(subset) + f_type + '.utf8')
        if os.path.isfile(current):
            print("loading file")
            return current
        else:
            print("creating file")
            with open(current, 'w') as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
            return current

    Label_files = sorted(get_file_names(path = datasets[set_type], type_ = 'LabelFile'))
    Input_files = sorted(get_file_names(path = datasets[set_type], type_ = 'InputFile'))
    names = ['msr','cityu','as','pku']
    choose = lambda i: i.split(".utf8")[0].split('/')[-1].split("_")[0]
    
    Label_files = [i for i in Label_files if choose(i) in subset]
    Input_files = [i for i in Input_files if choose(i) in subset]
    #if more than one subset is chosen we create another file
    if len(Label_files) > 1:
        Label_file = concat_subsets(Label_files, subset, f_type = 'LabelFile')
        Input_file = concat_subsets(Input_files, subset, f_type = 'InputFile')
    else:
        Label_file, Input_file = Label_files[0], Input_files[0]
        
    return Label_file, Input_file

class CreateDataset(object):
    '''makes feed files of combined unigrams and bigrams'''
    def __init__(self, LabelFile_path, InputFile_path, PaddingSize, set_type, TrainingVocab):
        self.Label_File = LabelFile_path
        self.Input_File = InputFile_path
        self.PaddingSize = PaddingSize
        self.set_type = set_type
        self.TrainingVocab = TrainingVocab
    
    def DateGen(self):
        '''Data generator given the feature generator. Pads the feature vectors accordingly
        returns: X_unigrams, X_bigrams, y, info (includes padding size, vocab sizes),
        uni_word_to_idx, bi_word_to_idx
        '''
        
        uni_feature_vectors, bi_feature_vectors, uni_word_to_idx, bi_word_to_idx = self.FeatureGenerator() 
        
        labels = self.BIESToNumerical()
        
        padded_labels = pad_sequences(labels, truncating='pre', padding='post', maxlen = self.PaddingSize)
        y =  K.utils.to_categorical(padded_labels, num_classes=4)
        
        X_unigrams = pad_sequences(uni_feature_vectors, truncating='pre',
                                   padding='post', maxlen = self.PaddingSize)
        X_bigrams = pad_sequences(bi_feature_vectors, truncating='pre',
                                  padding='post', maxlen = self.PaddingSize)
        
        info = {"uni_VocabSize": len(uni_word_to_idx)+1,
                "bi_VocabSize": len(bi_word_to_idx)+1}
        return X_unigrams, X_bigrams, y, info, uni_word_to_idx, bi_word_to_idx
    
    def DateGenTest(self):
        '''Data generator given the feature generator. 
        No padding done. To be used only for test datasets.
        '''
        
        X_unigrams, X_bigrams, uni_word_to_idx, bi_word_to_idx = self.FeatureGenerator() 
        
        return X_unigrams, X_bigrams
    
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
        '''Generates features based on unigrams and bigrams going line by line
        returns: unigram_feature_vectors, bigram_feature_vectors
        if training then returns also the word_to_idx for both unigrams and bigrams
        '''
        
        uni_feature_vectors, bi_feature_vectors = [], []
        
        if self.set_type == 'training':
            uni_word_to_idx, bi_word_to_idx = self.generateVocab()
        else:
            uni_word_to_idx, bi_word_to_idx = self.TrainingVocab

        with open(self.Input_File, 'r', encoding ='utf8') as f1:
            for line in f1:
                line = line.rstrip()
                
                unigrams = self.split_into_grams(line, 'uni_grams')
                bigrams = self.split_into_grams(line,'bi_grams')
                
                uni_feature_vectors.append([uni_word_to_idx.get(i, 1) for i in unigrams])
                bi_feature_vectors.append([bi_word_to_idx.get(i, 1) for i in bigrams])
                
        return uni_feature_vectors, bi_feature_vectors, uni_word_to_idx, bi_word_to_idx
    
    def generateVocab(self):
        '''
        Generates vocabulary based on file
        args: Inputfile, returns: word_to_index for unigrams and bigrams seperetly 
        '''
        with open(self.Input_File, 'r', encoding ='utf8') as f1:
            lines = f1.readlines()
            raw = ' '.join(' '.join(lines).split()) #one long string
        #creating unigrams and bigrams
        unigrams, bigrams = self.split_into_grams(raw, 'uni_grams'), self.split_into_grams(raw, 'bi_grams')
        del raw #erase from memory
        #geting seperate vocavularies
        unigrams_vocab, bigrams_vocab = set(unigrams), set(bigrams) 
        #initializing sepeate dictionaries
        uni_word_to_idx, bi_word_to_idx = dict(), dict()
        #Handling OOV
        uni_word_to_idx["<UNK>"], bi_word_to_idx["<UNK>"] = 1, 1
        #creating the rest of the word to index dict
        uni_word_to_idx.update({value:key for key,value in enumerate(unigrams_vocab, start = 2)})
        bi_word_to_idx.update({value:key for key,value in enumerate(bigrams_vocab, start = 2)})

        return uni_word_to_idx, bi_word_to_idx
        
    
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

def data_feed(subset='pku',padding=50):
    '''function that creates a dataset -- training, dev, and test
    args: subset: any subset, padding: padding size
    returns: (X_train, y_train), (X_dev, y_dev), (X_test, y_test), info_dev
    '''
    return_dict = dict()
    
    type_ = "training"
    print("*****{}*****".format(type_))
    Label_file, Input_file = ChooseDataset(type_, subset)
    A = CreateDataset(Label_file, Input_file, padding, type_, None)
    X_train_uni, X_train_bi, y_train, info_train, uni_word_to_idx, bi_word_to_idx = A.DateGen()
    return_dict.update({"train": {"X": [X_train_uni, X_train_bi],
                                 "y": y_train,
                                 "info": info_train}})
    print("X uni-bi shape: {}{}\ny shape: {}".format(X_train_uni.shape, X_train_bi.shape, y_train.shape))
    
    type_ = 'dev'
    Label_file, Input_file = ChooseDataset(type_, subset)
    A = CreateDataset(Label_file, Input_file, padding, type_, [uni_word_to_idx, bi_word_to_idx])
    X_dev_uni, X_dev_bi, y_dev, info_dev, _, _ = A.DateGen()
    return_dict.update({"dev": {"X": [X_dev_uni, X_dev_bi],
                               "y": y_dev}})
    print("*****{}*****".format(type_))
    print("X uni-bi shape: {}{}\ny shape: {}".format(X_dev_uni.shape, X_dev_bi.shape, y_dev.shape))
    
    return return_dict, uni_word_to_idx, bi_word_to_idx