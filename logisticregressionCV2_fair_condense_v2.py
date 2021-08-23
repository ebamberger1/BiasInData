# Cross validation implementation of logistic regression
# Includes fairness measurements
# To be used with German Credit dataset
# Handles automatic and manual versions of cross validation implementation of
# logistic regression on total and split datasets, where the datasets are
# split by gender
# Does not use gender in training of split models
# Also handles merging split models into one

import copy
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
pd.set_option("display.max_rows", None, "display.max_columns", None)

def cross_validate_method(X1, y1):
        
    # set model to logistic regression
    model_kfold = LogisticRegression(max_iter=1000)
    
    # set up 10-fold cross-validation 
    kfold = model_selection.KFold(n_splits=10)
    
    # perform 10-fold cross-validation for logistic regression
    results_kfold = model_selection.cross_validate(model_kfold, X1, y1, cv=kfold, return_estimator=True)
    
    numcoefs = len(X1[0])
    coefsums = [0] * numcoefs
    
    # prints coef of each fold
    for model in results_kfold['estimator']:
        # sum coefficients by column
        for i in range(numcoefs):
            # coef is a list of list with one element
            coefsums[i]+=model.coef_[0][i]
            i+=1
    scores = results_kfold['test_score']
    return scores, coefsums


def manual_cross_validation(X1, y1, col_names):
    scores = []
    predictedvalues = []
    predictedprobs = []
    numcoefs = len(X1[0])
    coefsums = [0] * numcoefs
    intercepts = []
        
    # set up 10-fold cross-validation 
    kfold = model_selection.KFold(n_splits=10)
    
    # force logistic regression on each kfold separately to capture scores
    for train_idx, test_idx in kfold.split(X1):
        # train_ix, test_ix are subscript values of each fold
        # for example, train [0-899] test [900-999]
            
        # get data
        train_X, test_X = X1[train_idx], X1[test_idx]
        train_y, test_y = y1[train_idx], y1[test_idx]
            
        # perform cross validation for each fold
        # fit model
        model_kfold = LogisticRegression(max_iter=1000)
        model_kfold.fit(train_X, train_y)
    	
        # evaluate model - get predicted and probs
        yhat = model_kfold.predict(test_X)
        probs = model_kfold.predict_proba(test_X)
        temp = (list(yhat))
            
        # add predicted values to list
        predictedvalues.extend(temp)
            
        # add max of predicted probabilities to list
        for i in range(len(probs)):
            predictedprobs.append(probs[i][1])
        
        # calculate accuracy of this fold
        acc = metrics.accuracy_score(test_y, yhat)
           
        # sum coef to get averages
        for i in range(numcoefs):
            coefsums[i]+=model_kfold.coef_[0][i]
    
     	# store score
        scores.append(acc)
        
        # store this fold's intercept
        intercepts.append(model_kfold.intercept_)
    
    # get average intercept
    intercept = sum(intercepts)/len(intercepts)
    
    return scores, coefsums, predictedvalues, predictedprobs, intercept


def get_manual_probs(test_X, coefs, intercept):
    # Matrix multiplication between the test set and the transpose of the
    # coefficients; add intercept to each row of the result to get the 
    # linear combination for each row
    lincombos = np.dot(test_X, np.transpose(coefs)) + intercept
    # calculate e^x / (1+e^x) for each row
    manualprobs = np.exp(lincombos)/(1 + np.exp(lincombos))
    return list(manualprobs)


def manual_cross_validation_merge(X1, y1, col_names, new_cols, genidx):
    scores = []
    predictedvalues = []
    predictedprobs = []
    numcoefs = len(new_cols[0:-1]) #sensitive attribute(s) will be removed
    coefsums = [0] * numcoefs
    intercepts = []
    
    # set up 10-fold cross-validation 
    kfold = model_selection.KFold(n_splits=10)
       
        
    # force logistic regression on each kfold separately to capture scores
    for train_idx, test_idx in kfold.split(X1):
        # train_ix, test_ix are subscript values of each fold
        # for example, train [0-899] test [900-999]
            
        # get data
        train_X, test_X = X1[train_idx], X1[test_idx]
        train_y, test_y = y1[train_idx], y1[test_idx]
        
        train_df = pd.DataFrame(train_X, columns = col_names[0:-1])
        test_df = pd.DataFrame(test_X, columns = col_names[0:-1])
        
        # tuple is male if gender attribue is 1 and female if it is 0
        m_train_X = train_df[train_df[col_names[genidx]] == 1]
        f_train_X = train_df[train_df[col_names[genidx]] == 0]
        
        train_y_list = list(train_y)
        m_train_y = [train_y_list[idx] for idx in m_train_X.index.values]
        f_train_y = [train_y_list[idx] for idx in f_train_X.index.values]
        
        # remove gender column from training and testing sets
        m_train_X = m_train_X[new_cols[0:-1]]
        f_train_X = f_train_X[new_cols[0:-1]]
        test_df = test_df[new_cols[0:-1]].to_numpy()
        
        # perform cross validation for each fold
        # fit model
        m_model_kfold = LogisticRegression(max_iter = 1000)
        m_model_kfold.fit(m_train_X, m_train_y)
        
        f_model_kfold = LogisticRegression(max_iter = 1000)
        f_model_kfold.fit(f_train_X, f_train_y)
        
        # average the intercept
        intercept = (m_model_kfold.intercept_ + f_model_kfold.intercept_)/2
        m_coefs = m_model_kfold.coef_[0]
        f_coefs = f_model_kfold.coef_[0]

        coefs = [] # this will contain merged coefs
        
        # average the coefficients
        for i in range(len(m_coefs)):
            coefs.append((m_coefs[i]+f_coefs[i])/2)
            
        # use the averaged coefficients and averaged intercept to manually
        # calculate prediction probabilities on each row of the testing set
        manualpredictprobs = get_manual_probs(test_df, coefs, intercept)
        predictedprobs.extend(manualpredictprobs)
        
        # if the prediction probability is >= 0.5, its predicted value is 1,
        # and if it is < 0.5, its predicted value is 0
        yhat = []
        for prob in manualpredictprobs:
            if prob < 0.5:
                yhat.append(0)
            else:
                yhat.append(1)
        
        predictedvalues.extend(yhat)
        
        # caluclate accuracy of this fold
        acc = metrics.accuracy_score(test_y, yhat)
                
        # sum coef to get averages
        for i in range(len(coefs)):
            coefsums[i]+=coefs[i]
    
     	# store score
        scores.append(acc)
        
        # store this fold's intercept
        intercepts.append(intercept)
        
    # get average intercept
    intercept = sum(intercepts)/len(intercepts)
            
    return scores, coefsums, predictedvalues, predictedprobs, intercept
    


def get_model_coefficients(X1, col_names, coefsums, scores):
    # get coef averages
    coefavgs = []
    print('Coefficient Averages')
    for i in range(len(coefsums)):
        coefavgs.append(coefsums[i] / 10.0)
        print(col_names[i], coefavgs[i])
        
    # prints accuracy of each fold
    print('Accuracy for folds', scores)
    print('Average Accuracy', sum(scores)/10)
    return coefavgs


def has_group_fairness(m_predicted, f_predicted):
    # 3.1.1 Group fairness
    prob_m = m_predicted.count(1)/len(m_predicted)
    prob_f = f_predicted.count(1)/len(f_predicted)
    # determine whether the probabilities are the same between both genders
    print('3.1.1:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05
    
    
def has_predictive_parity(m_tp, m_fp, f_tp, f_fp):
    # 3.2.1 Predictive parity
    # Calculate PPVs
    prob_m = m_tp / (m_tp + m_fp)
    prob_f = f_tp / (f_tp + f_fp)
    # determine whether the probabilities are the same between both genders
    print('3.2.1:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05

    
def has_false_positive_error_rate_balance(m_fp, m_tn, f_fp, f_tn):
    # 3.2.2 False positive error rate balance
    # Calculate FPRs
    prob_m = m_fp / (m_fp + m_tn)
    prob_f = f_fp / (f_fp + f_tn)
    # determine whether the probabilities are the same between both genders
    print('3.2.2:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05
    
    
def has_false_negative_error_rate_balance(m_fn, m_tp, f_fn, f_tp):
    # 3.2.3 False negative error rate balance
    # Calculate FNRs
    prob_m = m_fn / (m_tp + m_fn)
    prob_f = f_fn / (f_tp + f_fn)
    # determine whether the probabilities are the same between both genders
    print('3.2.3:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05
    
    
def has_equalized_odds(m_tn, m_fp, m_fn, m_tp, f_tn, f_fp, f_fn, f_tp):
    # 3.2.4 Equalized odds
    # Calculate TPRs
    prob_m1 = m_tp / (m_tp + m_fn)
    prob_f1 = f_tp / (f_tp + f_fn)
    # Calculate FPRs
    prob_m2 = m_fp / (m_fp + m_tn)
    prob_f2 = f_fp / (f_fp + f_tn)
    # determine whether the probabilities are the same between both genders
    print('3.2.4:', round(prob_m1, 2), round(prob_f1, 2), round(prob_m2, 2), round(prob_f2, 2), (abs(prob_m1 - prob_f1) < 0.05) and (abs(prob_m2 - prob_f2) < 0.05))
    return (abs(prob_m1 - prob_f1) < 0.05) and (abs(prob_m2 - prob_f2) < 0.05)
    
    
def has_conditional_use_accuracy_equality(m_tn, m_fp, m_fn, m_tp, f_tn, f_fp, f_fn, f_tp):    
    # 3.2.5 Conditional use accuracy equality
    # Calculate PPVs
    prob_m1 = m_tp / (m_tp + m_fp)
    prob_f1 = f_tp / (f_tp + f_fp)
    # Calculate NPVs
    prob_m2 = m_tn / (m_tn + m_fn)
    prob_f2 = f_tn / (f_tn + f_fn)
    # determine whether the probabilities are the same between both genders
    print('3.2.5:', round(prob_m1, 2), round(prob_f1, 2), round(prob_m2, 2), round(prob_f2, 2), (abs(prob_m1 - prob_f1) < 0.05) and (abs(prob_m2 - prob_f2) < 0.05))
    return (abs(prob_m1 - prob_f1) < 0.05) and (abs(prob_m2 - prob_f2) < 0.05)
    
    
def has_overall_accuracy_equality(m_tp, m_tn, f_tp, f_tn, m_predicted_len, f_predicted_len):
    # 3.2.6 Overall accuracy equality
    prob_m = (m_tp + m_tn) / m_predicted_len
    prob_f = (f_tp + f_tn) / f_predicted_len
    # determine whether the probabilities are the same between both genders
    print('3.2.6:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05
    
    
def has_treatment_equality(m_fn, m_fp, f_fn, f_fp):    
    # 3.2.7 Treatment equality
    prob_m = m_fn / m_fp
    prob_f = f_fn / f_fp
    # determine whether the probabilities are the same between both genders
    print('3.2.7:', round(prob_m, 2), round(prob_f, 2), abs(prob_m - prob_f) < 0.05)
    return abs(prob_m - prob_f) < 0.05
    
    
def has_test_fairness(m_predicted_score_info, f_predicted_score_info):    
    # 3.3.1 Test-fairness
    m_probs = {}
    f_probs = {}
    # for each predicted probability score (0 - 1.0), find the probability that
    # the actual class value is 1
    for key in m_predicted_score_info.keys():
        if not m_predicted_score_info[key][1] == 0:
            m_probs[key] = m_predicted_score_info[key][0] / m_predicted_score_info[key][1]
        else:
            m_probs[key] = 0
        if not f_predicted_score_info[key][1] == 0:
            f_probs[key] = f_predicted_score_info[key][0] / f_predicted_score_info[key][1]
        else:
            f_probs[key] = 0
            
    
    # for each predicted probability score (0 - 1.0), find how many probabilities
    # are the same between the genders
    num_same = 0
    for key in m_probs.keys():
        if(abs(m_probs[key] - f_probs[key]) < 0.05):
            num_same += 1
    
    if num_same == 0:
        print('3.3.1', 'Same-Different Ratio:', round(num_same/11, 2), False)
        return False
    elif num_same == 11:
        print('3.3.1', 'Same-Different Ratio:', round(num_same/11, 2), True)
        return True
    print('3.3.1', 'Same-Different Ratio:', round(num_same/11, 2), 'Partially True')
    return 'Partially True'


def has_well_calibration(m_predicted_score_info, f_predicted_score_info):
    # 3.3.2 Well-calibration
    m_probs = {}
    f_probs = {}
    # for each predicted probability score (0 - 1.0), find the probability that
    # the actual class value is 1
    for key in m_predicted_score_info.keys():
        if not m_predicted_score_info[key][1] == 0:
            m_probs[key] = m_predicted_score_info[key][0] / m_predicted_score_info[key][1]
        else:
            m_probs[key] = 0
        if not f_predicted_score_info[key][1] == 0:
            f_probs[key] = f_predicted_score_info[key][0] / f_predicted_score_info[key][1]
        else:
            f_probs[key] = 0
    
    
    # for each predicted probability score (0 - 1.0), find how many probabilities
    # are the same between the genders, and moreover, are also equivalent
    # to the predicted probability score
    num_same = 0
    for key in m_probs.keys():
        if(abs(m_probs[key] - f_probs[key]) < 0.05) and (abs(m_probs[key] - key) < 0.05) and (abs(f_probs[key] - key) < 0.05):
            num_same += 1
    if num_same == 0:
        print('3.3.2', 'Same-Different Ratio:', round(num_same/11, 2), False)
        return False
    elif num_same == 11:
        print('3.3.2', 'Same-Different Ratio:', round(num_same/11, 2), True)
        return True
    print('3.3.2', 'Same-Different Ratio:', round(num_same/11, 2), 'Partially True')
    return 'Partially True'


def has_balance_for_positive_class(m_y, f_y, m_predicted_score_info, f_predicted_score_info):
    # 3.3.3 Balance for positive class
    # E(X) = P(X_1) * X_1 + P(X_2) * X_2 + ...
    m_weighted_probs = []
    f_weighted_probs = []
    # find how many males and females have actual class value of 0 and how many
    # have actual class value of 1
    m_unique, m_counts = np.unique(m_y, return_counts = True)
    f_unique, f_counts = np.unique(f_y, return_counts = True)
    # for each predicted probability score (0 - 1.0), find the probability that
    # the actual class value is 1, and then multiply by the key value
    for key in m_predicted_score_info.keys():
        m_weighted_probs.append(key * m_predicted_score_info[key][0]/m_counts[1])
        f_weighted_probs.append(key * f_predicted_score_info[key][0]/f_counts[1])
    # determine whether the expected values are the same between both genders
    print('3.3.3:', round(sum(m_weighted_probs), 2), round(sum(f_weighted_probs), 2), abs(sum(m_weighted_probs) - sum(f_weighted_probs)) < 0.05)
    return abs(sum(m_weighted_probs) - sum(f_weighted_probs)) < 0.05
    

def has_balance_for_negative_class(m_y, f_y, m_predicted_score_info, f_predicted_score_info):
    # 3.3.4 Balance for negative class
    m_weighted_probs=[]
    f_weighted_probs=[]
    # find how many males and females have actual class value of 0 and how many
    # have actual class value of 1
    m_unique, m_counts = np.unique(m_y, return_counts = True)
    f_unique, f_counts = np.unique(f_y, return_counts = True)
    # for each predicted probability score (0 - 1.0), find the probability that
    # the actual class value is 0, and then multiply by the key value
    for key in m_predicted_score_info.keys():
        m_weighted_probs.append(key * (m_predicted_score_info[key][1]-m_predicted_score_info[key][0])/m_counts[0])
        f_weighted_probs.append(key * (f_predicted_score_info[key][1]-f_predicted_score_info[key][0])/f_counts[0])
    # determine whether the expected values are the same between both genders
    print('3.3.4:', round(sum(m_weighted_probs), 2), round(sum(f_weighted_probs), 2), abs(sum(m_weighted_probs) - sum(f_weighted_probs)) < 0.05) 
    return abs(sum(m_weighted_probs) - sum(f_weighted_probs)) < 0.05


def brier_score(y_true, y_prob):
    # Brier Score
    brier = metrics.brier_score_loss(y_true, y_prob)
    print('Brier', brier)
    return brier


def calculate_fairness_scores(m_y1, m_predictedvalues, m_predictedprobs, f_y1, f_predictedvalues, f_predictedprobs):
    # male and female confusion matrices
    # tn = true negative, fp = false positive,
    # fn = false negative, and tp = true positive
    m_tn, m_fp, m_fn, m_tp = metrics.confusion_matrix(m_y1, m_predictedvalues).ravel()
    f_tn, f_fp, f_fn, f_tp = metrics.confusion_matrix(f_y1, f_predictedvalues).ravel()
    
    print('Total Accuracy', (m_tn + m_tp + f_tn + f_tp)/(len(m_predictedvalues) + len(f_predictedvalues)))
    print('Male Accuracy', (m_tn + m_tp)/len(m_predictedvalues))
    print('Female Accuracy', (f_tn + f_tp)/len(f_predictedvalues))

    
    # Print Confusion Matrix
    print('True Negative', m_tn + f_tn, 'M:', m_tn, 'F:', f_tn)
    print('False Positive', m_fp + f_fp, 'M:', m_fp, 'F:', f_fp)
    print('False Negative', m_fn + f_fn, 'M:', m_fn, 'F:', f_fn)
    print('True Positive', m_tp + f_tp, 'M:', m_tp, 'F:', f_tp)
    
    #Fairness Definitions
    fairness_dict = {}
    
    # SECTION 3.1 Fairness Definitions
    # 3.1.1 Group Fairness
    fairness_dict['3.1.1'] = has_group_fairness(m_predictedvalues, f_predictedvalues)
    
    
    # SECTION 3.2 Fairness Definitions
    # 3.2.1 Predictive Parity
    fairness_dict['3.2.1'] = has_predictive_parity(m_tp, m_fp, f_tp, f_fp)
    
    # 3.2.2 False Positive Error Rate Balance
    fairness_dict['3.2.2'] = has_false_positive_error_rate_balance(m_fp, m_tn, f_fp, f_tn)
    
    # 3.2.3 False Negative Error Rate Balance
    fairness_dict['3.2.3'] = has_false_negative_error_rate_balance(m_fn, m_tp, f_fn, f_tp)
    
    # 3.2.4 Equalized Odds
    fairness_dict['3.2.4'] = has_equalized_odds(m_tn, m_fp, m_fn, m_tp, f_tn, f_fp, f_fn, f_tp)
    
    # 3.2.5 Conditional Use Accuracy Equality
    fairness_dict['3.2.5'] = has_conditional_use_accuracy_equality(m_tn, m_fp, m_fn, m_tp, f_tn, f_fp, f_fn, f_tp)
    
    # 3.2.6 Overall Accuracy Equality
    fairness_dict['3.2.6'] = has_overall_accuracy_equality(m_tp, m_tn, f_tp, f_tn, len(m_predictedvalues), len(f_predictedvalues))
    
    # 3.2.7 Treatment Equality
    fairness_dict['3.2.7'] = has_treatment_equality(m_fn, m_fp, f_fn, f_fp)
    
    
    # SECTION 3.3 Fairness Definitions
    # Each key-value pair is of the form
    # {predicted probability: [number of corresponding tuples where Y=1, total
    # corresponding tuples]}
    m_predicted_score_info = {0:[0, 0], 0.1:[0, 0], 0.2:[0, 0], 0.3:[0, 0], 
                              0.4:[0, 0], 0.5:[0, 0], 0.6:[0, 0], 0.7:[0, 0], 
                              0.8:[0, 0], 0.9:[0, 0], 1.0:[0, 0]}
    f_predicted_score_info = {0:[0, 0], 0.1:[0, 0], 0.2:[0, 0], 0.3:[0, 0], 
                              0.4:[0, 0], 0.5:[0, 0], 0.6:[0, 0], 0.7:[0, 0], 
                              0.8:[0, 0], 0.9:[0, 0], 1.0:[0, 0]}
    
    # bin the predicted probabilities into the groups 0, 0.1, 0.2, 0.3, 0.4, 0.5
    # 0.6, 0.7, 0.8, 0.9, and 1.0
    for i in range(len(m_predictedprobs)):
        key = round(m_predictedprobs[i], 1)
        # if the actual class value is 1
        if m_y1[i] == 1:
            m_predicted_score_info[key][0] += 1
            
        m_predicted_score_info[key][1] += 1
               
    for i in range(len(f_predictedprobs)):
        key = round(f_predictedprobs[i], 1)
        # if the actual class value is 1
        if f_y1[i] == 1:
            f_predicted_score_info[key][0] += 1
            
        f_predicted_score_info[key][1] += 1
        
    # 3.3.1 Test Fairness
    fairness_dict['3.3.1'] = has_test_fairness(m_predicted_score_info, f_predicted_score_info)
    
    # 3.3.2 Well-Calibration
    fairness_dict['3.3.2'] = has_well_calibration(m_predicted_score_info, f_predicted_score_info)
    
    # 3.3.3 Balance for Positive Class
    fairness_dict['3.3.3'] = has_balance_for_positive_class(m_y1, f_y1, m_predicted_score_info, f_predicted_score_info)        
    
    # 3.3.4 Balance for Negative Class
    fairness_dict['3.3.4'] = has_balance_for_negative_class(m_y1, f_y1, m_predicted_score_info, f_predicted_score_info)
    

    # Brier Score
    # Male Brier Score
    fairness_dict['Brier_M'] = brier_score(m_y1, m_predictedprobs)
    
    # Female Brier Score
    fairness_dict['Brier_F'] = brier_score(f_y1, f_predictedprobs)
    
    # Total Brier Score
    y1 = copy.deepcopy(list(m_y1))
    y1.extend(list(f_y1))
    predictedprobs = copy.deepcopy(m_predictedprobs)
    predictedprobs.extend(f_predictedprobs)
    
    fairness_dict['Brier_Total'] = brier_score(y1, predictedprobs)
    
    return fairness_dict

    
def main():
    # determine method to use
    print(' (1) Automatic Cross Validate on Whole Dataset\n',
          '(2) Manual Cross Validate on Whole Dataset\n',
          '(3) Automatic Cross Validate on Split Datasets\n',
          '(4) Manual Cross Validate on Split Datasets\n',
          '(5) Merge Split Cross-Validated Models')
    
    userinput = int(input('What would you like to do? '))
    if userinput < 3:  # manual log reg on each fold
        # define the location of the dataset
        # file should be germanwithheaderout.csv
        file = "germanwithheaderout.csv"
        #"https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
        
        # load the dataset with first row as column names
        # called "flipped..." because of 'creditclass' values being flipped
        flipped_dataframe = read_csv(file, header=0)
                
        # dataset has 0 as good and 1 as bad. To change this to 1- good and 0- bad,
        dataframe = flipped_dataframe.replace({'creditclass': {0: 1, 1: 0}})
        
        col_names = list(dataframe.columns.values)
        
        # make a dataframe that does not contain the class values
        # note that one-hot encoding and min-max scaling have already
        # been implemented on categorical and numeric attributes, respectively
        X1 = dataframe[col_names[0:-1]].to_numpy()
        
        # get the class values
        y1 = dataframe['creditclass']
        
        if userinput == 1: # using cross_validate method - does automatically on each fold
            scores, coefsums = cross_validate_method(X1, y1)
            get_model_coefficients(X1, col_names, coefsums, scores)
            
        else:
            scores, coefsums, predictedvalues, predictedprobs, intercept = manual_cross_validation(X1, y1, col_names)
        
            get_model_coefficients(X1, col_names, coefsums, scores)
            
            # A tuple is married/divorced female if personalstatus_A92 is 1
            f_rows = dataframe[dataframe.personalstatus_A92 == 1]
            
            # list of actual classification values for married/divorced female tuples
            f_y1 = list(f_rows['creditclass'])
            # list of predicted classification values for married/divorced female tuples
            f_predictedvalues = []
            # list of predicted classification probabilities for married/divorced female 
            # tuples
            f_predictedprobs = []
            for i in f_rows.index:
                f_predictedvalues.append(predictedvalues[i])
                f_predictedprobs.append(predictedprobs[i])
             
            
            # A tuple is married/divorced male if it is both not female and not
            # a single male (personalstatus_A92 = 0 = personalstatus_A93). Note
            # that it is not possible for both personalstatus_A92 and 
            # personalstatus_A93 to be 1
            #m_rows = dataframe[dataframe.personalstatus_A92 == dataframe.personalstatus_A93]
            
            # If instead you want to consider all males,
            m_rows = dataframe[dataframe.personalstatus_A92 == 0]
            
            # list of actual classification values for married/divorced male tuples
            m_y1 = list(m_rows['creditclass'])
            # list of predicted classification values for married/divorced male tuples
            m_predictedvalues = []
            # list of predicted classification probabilities for married/divorced male 
            # tuples
            m_predictedprobs = []
            for i in m_rows.index:
                m_predictedvalues.append(predictedvalues[i])
                m_predictedprobs.append(predictedprobs[i])
            
            fairness_dict = calculate_fairness_scores(m_y1, m_predictedvalues, m_predictedprobs, f_y1, f_predictedvalues, f_predictedprobs)
            
            print(fairness_dict)
        
    
    elif userinput < 5:
        # define the location of the dataset
        # should be germanprepout_Male.csv and germanprepout_Female.csv
        m_file = "germanprepout_Male.csv"
        f_file = "germanprepout_Female.csv"
        
        # load the datasets with first row as column names
        m_dataframe = read_csv(m_file, header=0)
        f_dataframe = read_csv(f_file, header=0)
        
        col_names = list(m_dataframe.columns.values) # both genders have same attributes
            
        # if personalstatus is separate from gender
        col_names.remove('gender_M')
        
        
        # make a dataframe for each gender that does not contain the class values
        # also takes gender out of dataframes
        # note that one-hot encoding and min-max scaling have already
        # been implemented on categorical and numeric attributes, respectively
        m_X1 = m_dataframe[col_names[0:-1]].to_numpy()
        f_X1 = f_dataframe[col_names[0:-1]].to_numpy()
        
        # get the class values
        m_y1 = m_dataframe['creditclass']
        f_y1 = f_dataframe['creditclass']
        
        if userinput == 3: # using cross_validate method - does automatically on each fold
            m_scores, m_coefsums = cross_validate_method(m_X1, m_y1)
            get_model_coefficients(m_X1, col_names, m_coefsums, m_scores)
            f_scores, f_coefsums = cross_validate_method(f_X1, f_y1)
            get_model_coefficients(f_X1, col_names, f_coefsums, f_scores)
        
        else: # userinput is 4
            m_scores, m_coefsums, m_predictedvalues, m_predictedprobs, m_int = manual_cross_validation(m_X1, m_y1, col_names)
            f_scores, f_coefsums, f_predictedvalues, f_predictedprobs, f_int = manual_cross_validation(f_X1, f_y1, col_names)
    
            print('Coefficients For Males:')
            get_model_coefficients(m_X1, col_names, m_coefsums, m_scores)
            print('Coefficients For Females:')
            get_model_coefficients(f_X1, col_names, f_coefsums, f_scores)
            
            fairness_dict = calculate_fairness_scores(m_y1, m_predictedvalues, m_predictedprobs, f_y1, f_predictedvalues, f_predictedprobs)
            
            print(fairness_dict)
            
    else: # userinput == 5
        # define the location of the dataset
        # file should be germanprepout.csv
        file = "germanprepout.csv"
       
        # load the dataset with first row as column names
        dataframe = read_csv(file, header=0)
        
        col_names = list(dataframe.columns.values)
        
        # make a dataframe that does not contain the class values
        # note that one-hot encoding and min-max scaling have already
        # been implemented on categorical and numeric attributes, respectively
        X1 = dataframe[col_names[0:-1]].to_numpy()
        
        # get the class values
        y1 = dataframe['creditclass']
        
        new_cols = copy.deepcopy(col_names)
        new_cols.remove('gender_M')
        
        scores, coefsums, predictedvalues, predictedprobs, intercept = manual_cross_validation_merge(X1, y1, col_names, new_cols, col_names.index('gender_M')) #col_names.index('personalstatus_A92')) 
        
        # get coef averages
        get_model_coefficients(X1, new_cols, coefsums, scores)         
        
        
        # A tuple is female if gender_M is 0
        f_rows = dataframe[dataframe.gender_M == 0]
        
        # list of actual classification values for female tuples
        f_y1 = list(f_rows['creditclass'])
        # list of predicted classification values for female tuples
        f_predictedvalues = []
        # list of predicted classification probabilities for female tuples
        f_predictedprobs = []
        for i in f_rows.index:
            f_predictedvalues.append(predictedvalues[i])
            f_predictedprobs.append(predictedprobs[i])
         
        # A tuple is male if gender_M is 1
        m_rows = dataframe[dataframe.gender_M == 1]
        
        # list of actual classification values for male tuples
        m_y1 = list(m_rows['creditclass'])
        # list of predicted classification values for male tuples
        m_predictedvalues = []
        # list of predicted classification probabilities for male tuples
        m_predictedprobs = []
        for i in m_rows.index:
            m_predictedvalues.append(predictedvalues[i])
            m_predictedprobs.append(predictedprobs[i])
        
        fairness_dict = calculate_fairness_scores(m_y1, m_predictedvalues, m_predictedprobs, f_y1, f_predictedvalues, f_predictedprobs)
        
        print(fairness_dict)
       
    
main()