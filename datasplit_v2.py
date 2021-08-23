# Splits a dataset based on a sensitive attribute provided by the user,
# where the tuples corresponding to each value of the attribute comprise a
# dataset

import numpy as np
from csv import reader
from csv import writer

# Processes the csv file
# filename is full name of file, assumed to be a csv file
# returns the header as a list of attribute names and the dataset as a list 
# of lists of values
def loadfile(filename):
    dataset = []
    header = []
    # dataset: list of lists representing csv data file
    with open(filename, newline='') as file:
        data_reader = reader(file)
        # keep header line separate
        header = next(data_reader)
        # process remaining lines
        for line in data_reader:
            dataset.append(line)
    return header, dataset


# Recodes the values of the sensitive attribute per user specification. This
# is useful if the values are numeric so that the split file names can be
# more intuitive
# attrunique is an array of unique values of the sensitive attribute
def recode(attrunique):
    newvals = []
    for val in attrunique:
        newval = input('Instead of ' + val + ', use ')
        newvals.append(newval)
    return newvals
        


# Splits a dataset based on a sensitive attribute provided in attributes
# filename is the name of the csv file without the '.csv' extension
# header is a list of attribute names
# dataset is a list of lists of values
# attributes is a list of sensitive attributes containing only 1 value
# classattr is the name of the attribute indicating the class value
def split_data (filename, header, dataset, attributes, classattr):
    attrarray = []
    classidx = header.index(classattr)
    # get column index corresponding to the sensitive attribute
    index = header.index(attributes[0])
    # add all values of that attribute to attrarray
    for line in dataset:
        attrarray.append(line[index])
    # get the unique values of that attribute
    attrunique = np.unique(attrarray)
    print('Unique Values', attrunique)
    choice = input('Would you like the above values to be used to distinguish between the split datasets? - yes or no\n ')
    filevals = attrunique
    if choice == 'no':
        filevals = recode(attrunique) 
    
    # for each unique value of the attribute, create a csv file of just those
    # tuples with the unique value
    # the file name will be of the form filename_AttributeValue.csv
    for i in range(len(attrunique)):
      with open(filename + '_' + filevals[i] + '.csv', 'w', newline = '') as file:
          data_writer = writer(file)
          data_writer.writerow(header[:classidx+1])
          for line in dataset:
              if line[index] == attrunique[i]:
                  data_writer.writerow(line[:classidx+1])
    print('Dataset separated into', len(attrunique), 'datasets', sep = ' ')
              

# Splits a dataset based on a sensitive attribute provided in attributes
# filename is the name of the csv file without the '.csv' extension
# header is a list of attribute names
# dataset is a list of lists of values
# attributes is a list of sensitive attributes containing only 2 values
# classattr is the name of the attribute indicating the class value
def split_data_2 (filename, header, dataset, attributes, classattr):
    attrdict = {}
    classidx = header.index(classattr)
    # get column indices corresponding to the sensitive attributes
    index1 = header.index(attributes[0])
    index2 = header.index(attributes[1])
    attrdict[attributes[0]] = []
    attrdict[attributes[1]] = []
    # add all values of the attributes to attrdict
    for line in dataset:
        attrdict[attributes[0]].append(line[index1])
        attrdict[attributes[1]].append(line[index2])
    # get the unique values of each attribute
    attrunique1 = np.unique(attrdict[attributes[0]])
    attrunique2 = np.unique(attrdict[attributes[1]])
    print('Unique Values', attrunique1)
    choice = input('Would you like the above values to be used to distinguish between the split datasets? - yes or no ')
    filevals1 = attrunique1
    if choice == 'no':
        filevals1 = recode(attrunique1)   
    print('Unique Values', attrunique2)
    choice = input('Would you like the above values to be used to distinguish between the split datasets? - yes or no ')
    filevals2 = attrunique2
    if choice == 'no':
        filevals2 = recode(attrunique2)   
    # for each unique combination of values of the attributes, create a csv 
    # file of just those tuples with the unique combination of values
    # the file name will be of the form 
    # filename_Attribute1Value_Attribute2Value.csv
    for i in range(len(attrunique1)):
        for j in range(len(attrunique2)):
          with open(filename + '_' + filevals1[i] + '_' + filevals2[j] + '.csv', 'w', newline = '') as file:
              data_writer = writer(file)
              data_writer.writerow(header[:classidx+1])
              for line in dataset:
                  if (line[index1] == attrunique1[i]) and (line[index2] == attrunique2[j]):
                      data_writer.writerow(line[:classidx+1])
    print('Dataset separated into', len(attrunique1)*len(attrunique2), 'datasets', sep = ' ')


def main():
    # filename should be germanwithheaderoutwithgender.csv
    filename = input("Enter input file name: ")
    header, dataset = loadfile(filename)
    attributes = []
    classattr = ''
    validattributes = False
    validclassattr = False
    # Get sensitive attributes from user. Loops until valid
    while not validattributes:
        validattributes = True
        print('Which attribute(s) would you like to split the data on?', \
              '(List one attribute or two comma delimited attributes)')
        attributes = input('Options: ' + ', '.join(header[:len(header)]) + 
                           '\n').split(',')
        
        if len(attributes) > 2:
            validattributes = False
            print('Too many attributes selected. Please try again.\n')
        else:
            for attr in attributes:
                if attr not in header:
                    validattributes = False
                    print('Invalid attributes selected. Please try again.\n')
    
    while not validclassattr:
        classattr = input('What is the class attribute? ')
        if not header.index(classattr) < 0:
            validclassattr = True
        else:
            print('Invalid attribute. Please try again.\n')
    # Split the file based on 1 sensitive attribute
    if len(attributes) == 1:        
        split_data(filename[:filename.index('.csv')], header, dataset, attributes, classattr)
    # Split the file based on 2 sensitive attributes
    elif len(attributes) == 2:        
        split_data_2(filename[:filename.index('.csv')], header, dataset, attributes)
    
    
main()
        

    

