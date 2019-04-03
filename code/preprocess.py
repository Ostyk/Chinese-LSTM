import os

def walk_dir(path):
    count=0
    len_ = len(os.listdir(path))
    print("start for path: {}\n".format(path))
    for path, subdir, files in os.walk(path):
        for name in files: 
            x = os.path.join(path,name)
            TraditionalToSimplifiedChinese(x)
            count+=1
            print(">>>{}/{} complete============\n".format(count, len_))
    print("end\n")
    return None

def TraditionalToSimplifiedChinese(input_file):
    '''transforms files'''
    
    name_, format_ = input_file.split(".")
    file_name_input = name_.split('/')[-1]
    
    name_ += '_simplified.' + format_
    file_name_output = name_.split('/')[-1]
    
    bash_ = 'hanzi-convert -s ' + input_file + ' > ' + name_
    os.system(bash_)
    print("Created {}".format(file_name_output))
    #os.system('rm '+ input_file)
    #print("Deleted {}".format(file_name_input))
    return None

datasets_ = {"training":'./icwb2-data/training',
             "dev":'./icwb2-data/gold',
             "testing":'./icwb2-data/testing'}

def encoder(line, type_of_file, verbose = False):
    '''encodes a single line'''
    
    new_line = ''
        
    #uses the +u3000 seperator if the file is 'AS'
    words = line.rstrip().split("\u3000") if type_of_file == 'as' else line.rstrip().split(" ")
    print("N of segments:",len(words))
    
    def bies_format(word):
        '''changes word to BIES format'''
        return 'B'+'I'*int(len(word)-2)+'E' if len(word)>1 else 'S'
    
    for word in words:
        if verbose: print("this is a word: {} of length {}".format(word, len(word)))
        new_line+=bies_format(word)
        
    return new_line

def FileToBIES(file_path):
    '''gives a file in utf converts to a txt in with all lines in the BIES format'''

    name_ = file_path.split(".")[-2] + '_labels'
    type_of_file = name_.split("/")[-1].split('_')[0] #to see if its an 'AS' file
    
    with open(file_path, 'r', encoding ='utf8') as f1, open(name_+".txt", "w") as f2:
        c=0
        for line in f1:
            c+=1
            print("input line:\t",line.rstrip())
            out = encoder(line, type_of_file)
            print("output line:\t{}".format(out))
            f2.writelines(out+"\n")
            if c==2:
                break
    return None
if __main__ == '__name__':
    
    #Traditional Chinese to Simplified Chinese
    walk_dir(datasets['training'])
    walk_dir(datasets['dev'])
    
    #Creates input file with no spaces
    
    #TO DO:
    
    #Creates label file based on original (simplified)
    FileToBIES(file_path = 'icwb2-data/training/pku_training.utf8')
    
    
    