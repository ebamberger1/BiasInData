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
    #df = pd.read_csv('compas-scores-two-years-v2.csv')
    #https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    
    #print('dataframe:', df.shape)
    
    
    df = pd.read_csv('compas-scores-two-years-v3.csv')
    #https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    
    print('dataframe:', df.shape)
    
    # This section considers the below listed attributes on black and white
    # individuals, removing bad data (negative jail times, those who were not
    # observed for the entire 2-year time span, etc.)
    #columns = list(df.columns.values)
    
    # Attributes Considered:
    # years_since_screening (# of years between COMPAS screening date and date
    # of data collection (Apr 1, 2016)), sex, age_at_screening (the age of the
    # defendent at time of COMPAS screening), race, juv_fel_count (# of felonies committed as juvenile),
    # juv_misd_count (# of misdemeanors committed as juvenile), juv_other_count
    # (# of other offenses committed as juvenile), priors_count (# of prior
    # offenses committed), days_in_jail (# of days spent in jail, calculated 
    # in Excel from c_jail_in and c_jail_out), c_charge_degree (whether a 
    # person is charged with a felony or misdemeanor), two_year_recid (outcome, likelihood of a new 
    # arrest within two years)
    
    columns = ['years_since_screening', 'sex', 'age_at_screening', 'race', 
               'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
               'priors_count', 'days_in_jail', 'c_charge_degree', 
               'two_year_recid']
    
    # dataframe should only include the attributes of interest
    df = df[columns]
    print(df.shape)
    
    
    # Select only those rows associated with Caucasian or African-American
    # individuals
    # End up with 6150, lose 1064 instances
    df = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')]
    print(df.shape)
    
    
    # PROBLEMS WITH DATA
    
    # For certain defendents, their jail_in time is later than their jail_out
    # time (this problem has not been documented to my knowledge).
    # Select only those rows in which days_in_jail >= 0
    # End up with 5968, lose additional 182 instances from last step
    df = df[df['days_in_jail'] >= 0]
    print(df.shape)
    
    
    # Age seems to be based off of date of data collection (Apr 1, 2016). Two
    # columns were created in Excel: (1) age_at_screening which represents the
    # age of the defendent at the time of their COMPAS screening, and 
    # (2) age_at_offense which represents the age of the defendent at the time
    # of their arrest or the time of their offense. Note that some defendents
    # have neither a date of arrest or date of offense, so (2) sometimes has
    # missing data.
    # Handled in Excel and in columns statement (age_at_screening chosen for
    # now to avoid missing data issue)
    
    
    
    # Data from Broward County was collected for all of 2013 and 2014, and
    # the cut-off date was set as April 1, 2016 (the date they starting looking
    # at the data). They included all defendents who either did not recidivate
    # in 2 years or who recidivated. The problem is that this means that
    # non-recidivists were included from Jan 1, 2013 through Apr 1, 2014, but
    # recidivists were included from Jan 1, 2013 through Dec 31, 2014, 
    # inflating the number of recidivists.
    # End up with 5111, lose additional 857 instances from last step
    df = df[df['years_since_screening'] >= 2]
    # We will not want to use years_since_screening as a predictor, so remove
    # from dataset
    columns.remove('years_since_screening')
    df = df[columns]
    print(df.shape)    
    
       
    
    # write data out to file compasprep.csv
    writetofile('compasprep.csv', columns, df.to_numpy())
    


main()