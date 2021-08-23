# one hot encode dataset
# this was written by Dr. St. Clair
import copy
import pandas as pd
import numpy as np
from csv import writer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# label encoding converts each value to an ordinal number
from sklearn.preprocessing import OneHotEncoder
# one hot encoding converts to set of binary values

pd.set_option("display.max_rows", None, "display.max_columns", None)

# will create column names to match one-hot-encoded columns
def convertcatcolnames(cat_cols, coded_cols):
    fd = []
    prefixcode = coded_cols[0].partition('_')[0]
    catprefix = cat_cols[0]
    i = 0
    for j in range(len(coded_cols)):
        nextprefixcode = coded_cols[j].partition('_')[0]
        suffix = coded_cols[j].partition('_')[2]
        if (nextprefixcode != prefixcode):
            prefixcode = nextprefixcode
            i+=1
            catprefix = cat_cols[i]
        fd.append(catprefix + '_' + suffix)
        
    return list(fd)

def writetofile(filename, cols, data):
    with open(filename, 'w', newline = '') as file:
        data_writer = writer(file)
        data_writer.writerow(cols)
        for line in data:
            data_writer.writerow(line)

def main():
    # If class attribute should be label encoded, 
    # need to uncomment code to label encode class
    # otherwise treated as cat or num
    
    infilename = input("enter input file name: ")
    df = pd.read_csv(infilename)
    
    # uncomment for label encoder
    #numcols = len(df.columns)
    
    # separate into X and y only if class needs label encoder
    # uncomment for label encoder
    #X = df.drop(df.columns[numcols-1], axis=1)
    #y = df.drop(df.columns[0:numcols-1], axis=1)
    X = df
    
    # get data types
    cat_cols = X.select_dtypes(include='object').keys()
    num_cols = X.select_dtypes(include='number').keys()

    # uncomment for label encoder
    #class_col = y.columns
    
    # one-hot encode categorical attributes
    encoder = OneHotEncoder(drop='first')
    onehot_data = encoder.fit_transform(X[cat_cols])
    
    # normalize numeric attributes
    num_data = X[num_cols]
    # scales and converts to numpy array
    num_data = MinMaxScaler().fit_transform(num_data)
    
    # label encode class attribute
    # uncomment for label encoder
    #labelencoder = LabelEncoder()
    #class_data = labelencoder.fit_transform(y)
    
    # notice extra () in parms to hstack
    final_data = np.hstack((onehot_data.toarray(), num_data))
    
    # uncomment for label encoder
    #class_data = class_data[:,np.newaxis]
    #final_data = np.hstack((final_data, class_data))    
    
    # get column names to match encoded data
    coded_cols = encoder.get_feature_names()
    coded_cols = convertcatcolnames(cat_cols, coded_cols)
    final_col_names = list(copy.deepcopy(coded_cols))
    final_col_names.extend(num_cols)
    # uncomment for label encoder
    #final_col_names.extend(class_col)
    
    outfilename = infilename.partition('.')[0] + 'out' + '.' + infilename.partition('.')[2]
    writetofile(outfilename, final_col_names, final_data)

main()

