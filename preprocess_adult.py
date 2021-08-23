# Preprocesses the adult dataset in preparation for one-hot encoding and
# scaling

import pandas as pd
import numpy as np
from csv import writer

#pd.set_option("display.max_rows", None, "display.max_columns", None)


# Replaces all values of an attribute with specified values.
# df is a dataframe, attr is an attribute of the dataframe
def recode_attr(df, attr):
    replace_dict = {}
    values = np.unique(df[[attr]].to_numpy())
    for val in values:
        if not val == '?':
            replace_val = input('What would you like to replace ' + val + ' with? ')
            replace_dict[val] = replace_val
    new_df = df.replace({attr: replace_dict})
    return new_df

# Replaces all minority values of an attribute with a specified value.
# df is a dataframe, attr is an attribute of the dataframe, and replace_val
# is a string specifiying the value to replace all minority values with
def recode_attr_minority(df, attr, replace_val):
    replace_dict = {}
    values, counts = np.unique(df[[attr]].to_numpy(), return_counts = True)
    values = list(values)
    counts = list(counts)
    majority_val = values[counts.index(max(counts))]
    values.remove(majority_val)
    for key in values:
        if not key == '?':
            replace_dict[key] = replace_val
    new_df = df.replace({attr: replace_dict})
    return new_df


# writes data out to a csv file with name given by filename
# filename is the name of the file written to, cols is a list of the column
# headers, and data is the main body of the data
def writetofile(filename, cols, data):
    with open(filename, 'w', newline = '') as file:
        data_writer = writer(file)
        data_writer.writerow(cols)
        for line in data:
            if not '?' in line:
                data_writer.writerow(line)



def main():
    train_df = pd.read_csv('adulttrain.csv')
    test_df = pd.read_csv('adulttest.csv')
    # combine the datasets into one
    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    print('dataframe:', df.shape)
    
    columns = list(df.columns.values)
    
    # counts how many tuples have missing values, i.e., '?'
    missingcount = 0
    for line in df.to_numpy():
        if '?' in line:
            missingcount += 1
    print('num missing', missingcount)
    
    # if workclass is 'Never-worked', changes occupation from '?' to 'None'        
    df.loc[df['workclass'] == 'Never-worked', ['occupation']] = 'None'
    

    # recode native countries other than US as Non-United-States
    df = recode_attr_minority(df, 'native-country', 'Non-United-States')
    
    
    # recode class values as 0s and 1s        
    # 0 - <= 50K (majority)
    # 1 - > 50K (minority)
    df = recode_attr(df, 'class')
    
    # write data out to file adultprep.csv
    writetofile('adultprep.csv', columns, df.to_numpy())
    


main()