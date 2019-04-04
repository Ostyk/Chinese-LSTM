import os
import sys

class Preprocess(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def deleteFiles(self, extension = '.txt'):
        '''
        Deletes all the .txt files in a given directory
        Only done to get rid of duplicate .txt files as I am using .utf8 files for training
        '''
        print("Deleting {} files from {}".format(extension, self.dataset_path))
        for path, subdir, files in os.walk(self.dataset_path):
            for name in files:
                if name.endswith(extension):
                    os.system('rm '+ os.path.join(self.dataset_path,name))
                    print('Succesfully removed: {}'.format(name))
            print("Done\n")

    def HanziConvert(self, input_file):
        '''transoforms a file form Traditional to Simplified Chinese'''

        output_file = input_file.split(".utf8")[0]+"_simplified.utf8"
        bash_ = 'hanzi-convert -s ' + input_file + ' > ' + output_file

        os.system(bash_)
        print("Created {}".format(output_file))
        os.system("rm "+input_file)
        print("Deleted {}".format(input_file))


    def action(self, action_type):
        '''performs severtal types of actions on a given directory of files
        args:
        -- self
        -- action_type: translate, InputFile, LabelFile
        '''
        count, fail = 0, 0
        len_ = len(os.listdir(self.dataset_path))
        print("starting ***{}*** for path: {}\n".format(action_type, self.dataset_path))
        for path, subdir, files in os.walk(self.dataset_path):
            for name in files:
                x = os.path.join(self.dataset_path, name)
                print("File: {}".format(x))
                if action_type == 'Translate':

                    self.HanziConvert(x)
                elif list(os.path.splitext(x))[0].split('/')[-1].endswith('simplified'): #checks for duplicates, so only done from original files
                    self.FileEncoder(x, action_type)
                else:
                    fail+=1
                count+=1
                print(">>>{}/{} complete {}\n".format(count, len_, '='*40))
        print("done\t --failed for {} files\n".format(fail))

    def FileEncoder(self, file_path, type_output='LabelFile'):
        '''gives a file in utf converts to a txt in with all lines in the BIES format'''

        type_of_file = file_path.split("/")[-1].split('_')[0] #to see if its an 'AS' file
        output_name = list(os.path.splitext(file_path))
        output_name.insert(1, '_' + type_output) #inserting file type to output file name
        output_name = ''.join(output_name)

        with open(file_path, 'r', encoding ='utf8') as f1, open(output_name, "w") as f2:
            for line in f1:
                if type_output=='LabelFile':
                    out = self.BIESencoder(line, type_of_file)
                    f2.writelines(out+"\n")
                elif type_output=='InputFile':
                    out = self.InputEncoder(line, type_of_file)
                    f2.writelines(out)

    @staticmethod
    def InputEncoder(line, type_of_file):
        '''gets rid of spaces in a line. This will be fed to the TF MODEL'''
        return line.replace("\u3000", "") if type_of_file == 'as' else line.replace(" ", "")


    @staticmethod
    def BIESencoder(line, type_of_file):
        '''encodes a single line to the BIES Format'''

        new_line = ''
        #uses the +u3000 seperator if the file is 'AS'
        words = line.rstrip().split("\u3000") if type_of_file == 'as' else line.rstrip().split(" ")
        #print("N of segments:",len(words))
        def bies_format(word):
            '''changes word to BIES format'''
            return 'B'+'I'*int(len(word)-2)+'E' if len(word)>1 else 'S'

        for word in words:
            #if verbose: print("this is a word: {} of length {}".format(word, len(word)))
            new_line+=bies_format(word)

        return new_line



if __name__ == '__main__':

    datasets = {"training":'../icwb2-data/training',
                "dev":'../icwb2-data/gold',
                "testing":'../icwb2-data/testing'}
    dataset = sys.argv[1]
    print(dataset, type(dataset))
    while not dataset in datasets.keys():
        print("Enterted dataset: "dataset)
        dataset = str(sys.argv[1])
    print("Success")
    print(datasets[dataset])

    P = Preprocess(datasets[dataset])
    P.deleteFiles('.txt')
    P.deleteFiles('.b5')
    P.action('Translate') #Converts from Tradition to Simplified Chinese
    P.action('LabelFile') #creates LabelFiles
    P.action('InputFile') #creates LabelFiles
