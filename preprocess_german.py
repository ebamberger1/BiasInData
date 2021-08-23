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
            replace_val = input('What would you like to replace ' + str(val) + ' with? ')
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
    df = pd.read_csv('germanwithheader.csv')
    
    print('dataframe:', df.shape)
    
    columns = list(df.columns.values)
    pstatidx = columns.index('personalstatus')
    genders = []
    
    for idx in df.index:
        if df['personalstatus'][idx] == 'A92':
            genders.append('F')
        else:
            genders.append('M')
    
    df.insert(pstatidx + 1, 'gender', genders)
    columns.insert(pstatidx + 1, 'gender')
    
    
    # recode personal status values       
    # MD - A91, A92, A94
    # S - A93, A95(none)
    df = recode_attr(df, 'personalstatus')
    
    
    # recode class values
    # 1 - Good (was 1)
    # 0 - Bad (was 2)
    df = recode_attr(df, 'creditclass')
    
    
    # write data out to file germanprep.csv
    writetofile('germanprep.csv', columns, df.to_numpy())
    


main()