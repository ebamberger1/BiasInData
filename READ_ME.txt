Project Name: Identifying and Eliminating Bias in Data

Contributors: Ethan Bamberger and Dr. Caroline St. Clair

Purpose: The purpose of this project was to study various datasets in an effort to determine to what extent, 
if any, certain methodologies may reduce bias in data. We particularly focused on how to reduce the bias of a
classifier in terms of how it treats individuals with respect to a specified sensitive attribute (e.g., race,
gender).

Method: In this project, we looked at improving fairness using the German Credit Dataset, Adult Dataset, and
ProPublica's COMPAS Dataset. Both the German Credit and Adult datasets are readily accessible via the UCI Machine
Learning Repository ([2] and [3], respectively). ProPublica's COMPAS Dataset is accessible via [4].

For each dataset, we began by preprocessing the data. We scaled each numeric attribute to be between 0 and 1,
and we applied one-hot encoding to each categorical attribute, dropping the first column for each attribute. 
For the German Credit Dataset, we considered all 20 attributes. We separated out Attribute 9 (Personal Status and Sex)
so that we could study gender separately from personal status. Hence, the personal status attribute was comprised of
single individuals (values of A93 and A95) and non-single individuals (values of A91, A92, or A94), and the gender
attribute was comprised of males (values of A91, A93, and A94) and females (values of A92 and A95). It is important to
note that there were no instances of single females (A95). After one-hot encoding, we ended up with a total of 48
features, including the class attribute, for which 0 indicated that the applicant had a poor credit score and 1
indicated that the applicant had a good credit score. The sensitive attribute considered in our analysis was gender.

For the Adult Dataset, we considered all 14 attributes. Since each native country other than the United States was
represented by a very small portion of the sample, we recoded the native country attribute by two values such that 
individuals were grouped by whether their country of origin was the United States. Additionally, we considered the
presence of missing data in this dataset. We noticed that for those individuals with a work class of 'Never Worked,' 
their occupation was noted as missing data. Rather than considering this as missing data, we changed all cases of this
such that these individuals' occupations were marked as 'None.' We then removed all other individuals with missing data
from the dataset. After one-hot encoding, we ended up with a total of 60 features, including the class attribute, for 
which 0 indicated that the individual had an income of at most $50,000 and 1 indicated that the individual had an income
of more than $50,000. The sensitive attribute considered in our analysis was gender.

For ProPublica's COMPAS Dataset, we considered 9 attributes, which were as follows: sex, age_at_screening (the age of the
defendant at the time of the COMPAS screening), race, juv_fel_count (the number of felonies committed as a juvenile),
juv_misd_count (the number of misdemeanors committed as a juvenile), juv_other_count (the number of other offenses 
committed as a juvenile), priors_count (the number of prior offenses committed), days_in_jail (the number of days the
defendant spent in jail, calculated in Excel from c_jail_in and c_jail_out), and c_charge_degree (whether the defendant 
was charged with a felony or misdemeanor). The outcome variable was two_year_recid (whether the defendant committed another 
crime within two years). We considered both the sensitive attributes of race and gender separately in our analysis. We only 
included individuals labeled in the data as Caucasian or African American. There were several data entry errors that resulted 
in bad data that consequently had to be excluded from the dataset, some of which were noted by [1]. One error was that some 
individuals' c_jail_in times were later than their c_jail_out times, resulting in the impossible situation of a negative 
quantity of days spent in jail. Another issue was found in that the age attribute was calculated based off of the date of data 
collection, April 1, 2016, not based on the time of the offense. Two columns were created in Excel in an effort to attain a 
better metric for age: 
(1) age_at_screening, which represents the age of the defendant at the time of their COMPAS screening, and 
(2) age_at_offense, which represents the age of the defendant at the time of their arrest or the time of their offense. 
Note that some defendants have neither a date of arrest nor a date of offense, so (2) sometimes has missing data. As such, (1)
was chosen as the best metric for age since it avoided missing data. Lastly, another error in data entry was found in that
the number of recidivists was inflated. ProPublica acquired data from Broward County, Florida, for all of 2013 and 2014, and the 
cut-off date was set as April 1, 2016, the date they starting looking at the data. They included all defendants who either 
did not recidivate in 2 years or who recidivated. The problem is that this means that those non-recidivists who committed their 
original offenses any time during January 1, 2013 through April 1, 2014 were included, but recidivists who committed their 
original offenses any time during January 1, 2013 through December 31, 2014 were included, a much larger time frame. 
By collecting recidivists for a much larger time frame than non-recidivists, the number of recidivists was inflated,
necessitating the exclusion of recidivists who committed their initial offenses after April 1, 2014.

We trained a binary logistic regression classifier on the entire dataset to create the original model, using 10-fold 
cross-validation. We used the class predictions derived from each fold of the model to create a confusion matrix for 
each value of the chosen sensitive attribute (e.g., male and female for gender; White and Black for race). For each 
statistical measure we looked at (all of which are outlined by [5]), these matrices were used to calculate the desired 
probability for each value of the sensitive attribute. The statistical measures of fairness we focused on were the most 
versatile fairness definitions to easily compare across several datasets. The classifier’s accuracy and Brier score 
were also calculated for the entire dataset and for each value of the sensitive attribute.

The coefficients of the original model, however, were influenced heavily by the value of the sensitive attribute that held 
the majority. In an effort to improve fairness, we created two manipulated models (split and merged), using the same technique 
of binary logistic regression with 10-fold cross-validation. The split model was intended to improve fairness by removing any 
influence of the sensitive attribute on the model’s coefficients, whereas the merged model was intended to improve fairness 
by equating the influence of the two values of the sensitive attribute on the coefficients. To obtain our split model, we split 
the dataset into two datasets based on the values of the sensitive attribute and trained a classifier on each split dataset. 
For our merged model, we used the entire dataset and trained two sub-models for each fold of the model, one sub-model for each 
value of the sensitive attribute. Each coefficient for a fold was calculated via the unweighted average of the coefficients of 
both sub-models. For both manipulated models (split and merged), we calculated the same statistical measures of fairness, 
accuracies, and Brier scores.


References
[1] Barenstein, M. (2019). ProPublica’s COMPAS data revisited. arXiv. Retrieved June 24, 2021, 
	    from arXiv:1906.04711v3
[2] Hofmann, H. (1994). Statlog (German credit data) data set [Data set]. UCI Machine Learning 
		Repository. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
[3] Kohavi, R., & Becker, B. (1996). Adult data set [Data set]. UCI Machine Learning Repository. 
		https://archive.ics.uci.edu/ml/datasets/Adult
[4] Larson, J., Roswell, M., & Atlidakis, V. (2017). Compas-scores-two-years [Data set]. 
		ProPublica. https://github.com/propublica/compas-analysis
[5] Verma, S., & Rubin, J. (2018, May 29). Fairness definitions explained. FairWare’18: 
		IEEE/ACM International Workshop on Software Fairness, Gothenburg, Sweden. 
		https://doi.org/10.1145/3194770.3194776


Files:
datasplit_v2.py : splits a dataset based on the values of a specified sensitive attribute
logisticregressionCV2_fair_condense_adult_v2.py : performs the logistic regression analysis on the Adult dataset with respect
							to gender
logisticregressionCV2_fair_condense_compas.py : performs the logistic regression analysis on ProPublica's COMPAS dataset with
							respect to race
logisticregressionCV2_fair_condense_compas_gen.py : performs the logistic regression analysis on ProPublica's COMPAS dataset
							with respect to gender
logisticregressionCV2_fair_condense_v2.py : performs the logistic regression analysis on the German Credit dataset with
							respect to gender
onehotencode.py : performs one-hot encoding on a dataset
preprocess_adult.py : handles preprocessing of Adult dataset in preparation for one-hot encoding
preprocess_compas.py : handles preprocessing of ProPublica's COMPAS dataset in preparation for one-hot encoding
preprocess_german.py : handles preprocessing of the German Credit dataset in preparation for one-hot encoding

 