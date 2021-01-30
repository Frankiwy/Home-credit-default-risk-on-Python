import time
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide = 'ignore')
from sklearn.impute import SimpleImputer
from usefull_functions import directory_function, counting_columns_with_NaN, imputation_function, describe_function, \
    fix_DAYS_EMPLOYED_feature, economical_features, from_days_to_years, pearson_correaltion_coefficient, heat_map_function, \
    qq_plot, skewness_kurtosis_plots, Skewness_and_Kurtosis_num, apply_transformation_definitely, imputing_category, \
    statistics_for_numerical_and_categorical, merge_df, pearson_for_others, customer_aggregation, detect_multicollinearity, \
    PCA_plot, pca_function
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC





now = datetime.now() # timer for date
starttime = time.time() # timer

####################################################
# (1) DIRECTORY FUNCTION  and REDUCE_SIZE FUNCTION #
####################################################

for dir in ['html_summary_pages', 'images']:
    directory_function(dir) # create two folders that will be used to store .hlmt and .png files

################################################################
# (2) DATA EXPLORATION & DATA CLEANING  ON TRAIN AND TEST SETS #
################################################################
train = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/application_train.csv", sep=',')
test = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/application_test.csv", sep=',')
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))

missing_info_train = counting_columns_with_NaN(train, printing=True) # check how many NaN in Train
missing_info_test = counting_columns_with_NaN(test, printing=True) # checl how many NaN in Test

list_datasets=['TRAIN', 'TEST']
train_test_list = list()
for e, dataframe in enumerate([train, test]):
    print()
    train_test_list.append(imputation_function(dataframe, printing=False))
    print('For the {}:'.format(list_datasets[e]))
    counting_columns_with_NaN(dataframe, printing=True)
train = train_test_list[0]
test = train_test_list[1]

del dataframe, missing_info_test, missing_info_train, train_test_list

descriptive_dataframe = describe_function(train) # it returns a dataframe with all the statistics for numerical features
descriptive_dataframe.to_html('html_summary_pages/descriptive_train.html') # export as html in order to see it better
del descriptive_dataframe
'''
As it is possible seeing from the .html file above, there are some anomalous data inside the data set.
Then, in  this section what it's gonna do it is to try and fix them. In particular:
- DAYS_BIRTH, 
- 'DAYS_REGISTRATION',
- 'DAYS_ID_PUBLISH'
are expressed in negative days so, by using the "from_days_to_years" funtion they will be converted in positive years.
Moreover, the 'DAYS_EMPLOYED' feature has some anomalous values because the maximum value is about 1000 years so,
by using the function "..." a new boolean column (where 0 indicates no Anomalous and the other way around) will be created,
but also the DAYS_EMPLOYED column will be modified by assigning to the anomalous observations the median value of the
column.
'''
############################
# (2.1) Fix Anomalous Data #
############################

train_categorical = train.select_dtypes(include='object')
test_categorical = test.select_dtypes(include='object')
train_numerical = train.select_dtypes(include=['float64', 'int64'])
test_numerical = test.select_dtypes(include=['float64', 'int64'])

train_numerical = fix_DAYS_EMPLOYED_feature(train_numerical) # update and fix 'DAYS_EMPLOYED' feature
test_numerical = fix_DAYS_EMPLOYED_feature(test_numerical) # update and fix 'DAYS_EMPLOYED' feature
'''
Create Economical Features:
    - CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    - ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    - CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    - DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
    - INCOME_PER_PERSON: income per person
'''

# create economical features and convert some values from days to years
print('\n','Creating economical features and adjusting [DAYS_BIRTH, DAYS_REGISTRATION, DAYS_ID_PUBLISH] for both TRAIN and TEST sets...')

train_numerical = economical_features(train_numerical)
test_numerical = economical_features(test_numerical)

train_numerical = from_days_to_years(train_numerical, ['DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'])
test_numerical = from_days_to_years(test_numerical, ['DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'])

###########################################################
# (2.2) Correlation coefficient Among Target and Features #
###########################################################
'''
In order to find out which are the most relevant features the person correlation coefficient has been computed
and only the first top 10 either positive or negative correlations have been taken in cosideration.
The other one will be discarded and not considered while building up the model. 
'''
negative, positive, pearson_scores = pearson_correaltion_coefficient(train_numerical, top=20) # call the function and compute the pearson coefficient
list_negative = list(negative.index)
list_positive = list(positive.index)
train_subset = train_numerical[list_negative + list_positive + ['TARGET']]
test_subset = test_numerical[list_negative + list_positive]

heat_map_function(train_subset, name_image='correlation_all_numerics') #save heat-map into folder

#del train_numerical, test_numerical
################################
# (2.3) feature transformation #
################################

# compute skewness and kurtosis for all the selected numerical features, by applying four different transformations
# in order to see if one of them it is able to churn out better results in terms of distribution.
Skewness_and_Kurtosis_num(train_numerical, list_negative + list_positive)\
    .to_html('html_summary_pages/skewness_&_kurtosis.html') # the df will be stored into a .html file in order to better see it

transform_features = [
    'DAYS_ID_PUBLISH_IN_YEARS', 'DAYS_REGISTRATION_IN_YEARS', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'FLOORSMAX_AVG','FLOORSMAX_MEDI', 'FLOORSMAX_MODE', 'ELEVATORS_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_EMPLOYED_PERCENT', 'TOTALAREA_MODE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'CNT_CHILDREN', 'ANNUITY_INCOME_PERCENT', 'CNT_FAM_MEMBERS']#, 'OBS_60_CNT_SOCIAL_CIRCLE'],'OBS_30_CNT_SOCIAL_CIRCLE']
print()
for e,f in enumerate(transform_features):
    print('Computing skew and kurt for selected features to see improvements ')
    qq_plot(train_subset, f, f)
    skewness_kurtosis_plots(train_subset, f, f)
    print('Still {} features to be processed'.format(len(transform_features)- (e+1)), end="\r")
    if len(transform_features)- (e+1) == 0:
        print('All features processed!')

del transform_features, train_numerical, test_numerical

# Apply transformation on TRAIN and TEST
print('\n', 'Train transformation...')
sqrt_transform = ['DAYS_ID_PUBLISH_IN_YEARS', 'DAYS_REGISTRATION_IN_YEARS', 'FLOORSMAX_AVG', 'FLOORSMAX_MEDI', 'FLOORSMAX_MODE', 'ELEVATORS_AVG',
                  'AMT_REQ_CREDIT_BUREAU_YEAR']#, 'OBS_60_CNT_SOCIAL_CIRCLE']# , 'OBS_30_CNT_SOCIAL_CIRCLE']
cbrt_transform = ['REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED', 'DEF_30_CNT_SOCIAL_CIRCLE', 'DAYS_EMPLOYED_PERCENT', 'TOTALAREA_MODE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                      'CNT_CHILDREN', 'ANNUITY_INCOME_PERCENT', 'CNT_FAM_MEMBERS']
ln_transform = ['AMT_GOODS_PRICE', 'AMT_CREDIT']

train_transformed = apply_transformation_definitely(train_subset,sqrt_transformation=sqrt_transform, cbrt_transformation=cbrt_transform,
                                                    ln_transformation= ln_transform, square_transformation=[])
print('\n', 'Test transformation...')
test_transformed = apply_transformation_definitely(test_subset, sqrt_transformation=sqrt_transform, cbrt_transformation=cbrt_transform,
                                                   ln_transformation=ln_transform, square_transformation=[])
del sqrt_transform, cbrt_transform, ln_transform#, train_subset, test_subset

#Plot the qqplot for the TRAIN
print()
for e, col in enumerate(list(train_transformed.columns)):
    qq_plot(train_transformed, col, col)
    print('Still {} qqplots need to be generated'.format(len(train_transformed.columns)-(e+1)), end="\r")
    if len(train_transformed.columns)-(e+1) == 0:
        print('Done, all qqplots have been generated!')



############################################
# (2.4) Imputation on Categorical features #
############################################

train_category, missing_train = imputing_category(train_categorical) # do imputation
test_category, missing_test = imputing_category(test_categorical) # do imputation


train_for_models = pd.concat([train_transformed, train_category], axis=1)
train_for_models['SK_ID_CURR'] = train['SK_ID_CURR']


test_for_models = pd.concat([test_transformed, test_category], axis=1)
test_for_models['SK_ID_CURR'] = test['SK_ID_CURR']

del train_transformed, test_transformed, train_category, test_category, train_categorical, test_categorical

train_for_models.to_csv('train_for_models.csv', index=False) # store file into .csv in order to don't generate it every time
test_for_models.to_csv('test_for_models.csv', index=False) # store file into .csv in order to don't generate it every time

#####################################
# [3] INCREASING NUMBER OF FEATURES #
#####################################
'''
In the previous stages it is retorned a dataframe with 182 features that can be used to build a model for predictions.
Howere, these features are not enough because they actually are not able to explain, in an accurate way, the variance of our dependent variable.
For that reason, it seems reasonable to try to increase the number of important features that can be used. In this section all the other
.csv files will be imported and from them, different statistics will be computed in order to produce useful features. 
'''


####################
# (3.1) bureau.csv #
####################

print('bureau.csv file')

bureau = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/bureau.csv", sep=',')

bureau_for_corr_pearson = merge_df(
    statistics_for_numerical_and_categorical(df=bureau.drop(columns = ['SK_ID_BUREAU']), aggregation_variable='SK_ID_CURR', new_var_name='bureau', category='numerical'),
    statistics_for_numerical_and_categorical(df=bureau, aggregation_variable='SK_ID_CURR', new_var_name='bureau', category='categorical'), merging_on='SK_ID_CURR')


bureau_final, bureau_final_columns, bureau_pearson_results = pearson_for_others(train, bureau_for_corr_pearson, threshold=0.02, imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(bureau_for_corr_pearson, list(bureau_final.columns))\
    .to_html('html_summary_pages/bureau_final_skewness_&_kurtosis.html')


# feauters that benefit of a transformation:

sqrt_transform = ['bureau_DAYS_ENDDATE_FACT_count', 'bureau_CREDIT_TYPE_Credit card_mean'] #featurs get benefit from sqrt transformation
cbrt_transform = ['bureau_DAYS_CREDIT_ENDDATE_max', 'bureau_DAYS_CREDIT_ENDDATE_min', 'bureau_CREDIT_TYPE_Microloan_mean',
                  'bureau_DAYS_CREDIT_UPDATE_sum', 'bureau_DAYS_CREDIT_UPDATE_min', 'bureau_DAYS_CREDIT_sum', 'bureau_DAYS_CREDIT_ENDDATE_mean',
                  'bureau_DAYS_CREDIT_ENDDATE_sum', 'bureau_DAYS_ENDDATE_FACT_sum', 'bureau_DAYS_ENDDATE_FACT_min', 'bureau_DAYS_CREDIT_max', 'bureau_DAYS_CREDIT_UPDATE_mean'] # features get benefit from cbrt transformation

bureau_final = apply_transformation_definitely(bureau_final, sqrt_transformation=sqrt_transform, cbrt_transformation= cbrt_transform,
                                                          ln_transformation=[], square_transformation=[]) # apply transofrmation and return final bureau

bureau_final.to_csv('bureau_final.csv', index=False) # store file into .csv in order to don't generate it every time

del sqrt_transform, cbrt_transform, bureau_for_corr_pearson, bureau_final_columns, bureau_pearson_results

############################
# (3.2) bureau_balance.csv #
############################
print('bureau_balance.csv file')

bureau_balance = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/bureau_balance.csv", sep=',')

df_bureau_balance_loan = statistics_for_numerical_and_categorical(df=bureau_balance, aggregation_variable='SK_ID_BUREAU', new_var_name='bureau_balance', category='numerical').merge(
    statistics_for_numerical_and_categorical(df=bureau_balance, aggregation_variable='SK_ID_BUREAU', new_var_name='bureau_balance', category='categorical'),
    right_index= True, left_on = 'SK_ID_BUREAU', how='outer') # a the bureau_balance dataframe grouped by loan


# insert the SK_ID_CURR necessary for merging with train and test
bureau_balance_loan  = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(df_bureau_balance_loan, on = 'SK_ID_BUREAU', how='left')
bureau_balance_loan_for_corr_pearson = statistics_for_numerical_and_categorical(bureau_balance_loan.drop(
    columns=['SK_ID_BUREAU']), aggregation_variable= 'SK_ID_CURR', new_var_name='customer', category='numerical') # groups the loans by client

bureau_balance_final, bureau_balance_final_columns, bureau_balance_pearson_results = pearson_for_others(train, bureau_balance_loan_for_corr_pearson, threshold=0.02,
                                                                                                        imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(bureau_balance_final, list(bureau_balance_final.columns))\
    .to_html('html_summary_pages/bureau_balance_FINAL_skewness_&_kurtosis.html')

# feauters take advantages of a transformation:
sqrt_transform = ['customer_bureau_balance_STATUS_X_sum_max', 'customer_bureau_balance_STATUS_C_sum_mean',
                  'customer_bureau_balance_STATUS_0_sum_max', 'customer_bureau_balance_STATUS_0_sum_mean']
cbrt_transform = [ 'customer_bureau_balance_MONTHS_BALANCE_count_mean', 'customer_bureau_balance_MONTHS_BALANCE_count_max', 'customer_bureau_balance_MONTHS_BALANCE_count_sum',
                   'customer_bureau_balance_STATUS_C_sum_sum', 'customer_bureau_balance_STATUS_X_sum_mean', 'customer_bureau_balance_STATUS_X_sum_sum',
                   'customer_bureau_balance_STATUS_C_mean_sum', 'customer_bureau_balance_STATUS_0_sum_sum', 'customer_bureau_balance_MONTHS_BALANCE_mean_max',
                   'customer_bureau_balance_MONTHS_BALANCE_mean_sum', 'customer_bureau_balance_MONTHS_BALANCE_min_max', 'customer_bureau_balance_MONTHS_BALANCE_min_sum',
                   'customer_bureau_balance_MONTHS_BALANCE_sum_sum', 'customer_bureau_balance_MONTHS_BALANCE_sum_mean', 'customer_bureau_balance_MONTHS_BALANCE_sum_min']
ln_transform = ['customer_bureau_balance_MONTHS_BALANCE_count_min']

bureau_balance_final = apply_transformation_definitely(bureau_balance_final, sqrt_transformation=sqrt_transform, cbrt_transformation=cbrt_transform,
                                                       ln_transformation=ln_transform, square_transformation=[]) #transform features and return the final bureau balance


bureau_balance_final.to_csv('bureau_balance_final.csv', index=False) # store file into .csv in order to don't generate it every time

del bureau_balance, df_bureau_balance_loan, bureau_balance_loan, bureau_balance_loan_for_corr_pearson, bureau_balance_final_columns, bureau_balance_pearson_results, bureau



##################################
# (3.3) previous_application.csv #
##################################
print('previous_application.csv file')
previous_app = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/previous_application.csv", sep=',')


previous_app = merge_df(statistics_for_numerical_and_categorical(previous_app, aggregation_variable='SK_ID_CURR', new_var_name='previous_app', category='numerical'),
                        statistics_for_numerical_and_categorical(previous_app, aggregation_variable='SK_ID_CURR', new_var_name='previous_app', category='categorical'),
                        merging_on='SK_ID_CURR') # compute numerical and categorical statistics for each previous customer loan applications


previous_app_final, previous_app_final_columns, previous_app_pearson_results = pearson_for_others(train, previous_app,threshold=0.02, imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(previous_app_final, list(previous_app_final.columns))\
    .to_html('html_summary_pages/previous_app_final_skewness_&_kurtosis.html') # check skew and kurtosis for selected features

# features benefit of transformation:
sqrt_transform = ['previous_app_CODE_REJECT_REASON_XAP_mean', 'previous_app_DAYS_FIRST_DRAWING_sum', 'previous_app_RATE_DOWN_PAYMENT_sum',
                  'previous_app_NAME_YIELD_GROUP_low_normal_mean', 'previous_app_RATE_DOWN_PAYMENT_max', 'previous_app_DAYS_FIRST_DRAWING_count',
                  'previous_app_NAME_CONTRACT_TYPE_Revolving loans_mean', 'previous_app_NAME_YIELD_GROUP_high_mean', 'previous_app_NAME_CONTRACT_STATUS_Refused_mean']
cbrt_transform = ['previous_app_AMT_ANNUITY_mean', 'previous_app_PRODUCT_COMBINATION_Cash X-Sell: low_mean',
                  'previous_app_PRODUCT_COMBINATION_POS industry with interest_mean', 'previous_app_AMT_ANNUITY_min',
                  'previous_app_CHANNEL_TYPE_AP+ (Cash loan)_mean', 'previous_app_PRODUCT_COMBINATION_Card Street_mean',
                  'previous_app_CODE_REJECT_REASON_LIMIT_mean', 'previous_app_DAYS_DECISION_mean', 'previous_app_CODE_REJECT_REASON_HC_mean',
                  'previous_app_CODE_REJECT_REASON_SCOFR_mean', 'previous_app_NAME_PRODUCT_TYPE_walk-in_mean']
square_tranform = ['previous_app_DAYS_FIRST_DRAWING_mean']
previous_app_final = apply_transformation_definitely(previous_app_final,
                                                       sqrt_transformation=sqrt_transform,
                                                       cbrt_transformation=cbrt_transform,
                                                       ln_transformation=[],
                                                       square_transformation=square_tranform) #transform features and return the final bureau balance

previous_app_final.to_csv('previous_app_final.csv', index=False) # store file into .csv in order to don't generate it every time
del sqrt_transform, cbrt_transform, square_tranform, previous_app, previous_app_final_columns, previous_app_pearson_results

##############################
# (3.4) POS_CASH_balance.csv #
##############################
print('pos_cash_balance.csv file')

cash = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/POS_CASH_balance.csv", sep=',')
cash_by_customer = customer_aggregation(cash, aggregation_variables = ['SK_ID_PREV', 'SK_ID_CURR'], names = ['cash', 'customer'])

print('cash by client imported')
cash_by_customer_final, cash_by_customer_final_columns, cash_by_customer_pearson_results = pearson_for_others(train, cash_by_customer, threshold=0.02, imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(cash_by_customer_final, list(cash_by_customer_final.columns))\
    .to_html('html_summary_pages/cash_by_customer_FINAL_skewness_&_kurtosis.html')

# features benefit of transformation:
sqrt_transform = ['customer_cash_CNT_INSTALMENT_FUTURE_max_min']
cbrt_transform = ['customer_cash_NAME_CONTRACT_STATUS_Active_mean_sum', 'customer_cash_CNT_INSTALMENT_FUTURE_min_count', 'customer_cash_MONTHS_BALANCE_max_mean', 'customer_cash_MONTHS_BALANCE_max_sum',
                  'customer_cash_MONTHS_BALANCE_sum_min', 'customer_cash_MONTHS_BALANCE_min_sum', 'customer_cash_CNT_INSTALMENT_FUTURE_mean_min','customer_cash_MONTHS_BALANCE_mean_sum']
ln_transform = ['customer_cash_NAME_CONTRACT_STATUS_XNA_mean_count', 'customer_cash_CNT_INSTALMENT_mean_min', 'customer_cash_CNT_INSTALMENT_max_min']

cash_by_customer_final = apply_transformation_definitely(cash_by_customer_final, sqrt_transformation=sqrt_transform, cbrt_transformation=cbrt_transform,
                                                       ln_transformation=ln_transform, square_transformation=[]) #transform features and return the final bureau balance

cash_by_customer_final.to_csv('cash_by_customer_final.csv', index=False) # store file into .csv in order to don't generate it every time
del sqrt_transform, cbrt_transform, ln_transform, cash_by_customer, cash_by_customer_final_columns, cash_by_customer_pearson_results


#################################
# (3.5) credit_card_balance.csv #
#################################
print('credit_card_balance.csv file')

credit = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/credit_card_balance.csv", sep=',')
credit_by_customer = customer_aggregation(credit, aggregation_variables = ['SK_ID_PREV', 'SK_ID_CURR'], names = ['credit', 'client'])
credit_by_customer_final, credit_by_customer_final_columns, credit_by_customer_pearson_results = pearson_for_others(train, credit_by_customer, threshold=0.03, imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(credit_by_customer_final, list(credit_by_customer_final.columns))\
    .to_html('html_summary_pages/credit_by_customer_FINAL_skewness_&_kurtosis.html') # check skew and kurtosis for selected features

credit_by_customer_final.to_csv('credit_by_customer_final_03.csv', index=False) # store file into .csv in order to don't generate it every time

del credit, credit_by_customer_final_columns, credit_by_customer_pearson_results

###################################
# (3.6) installments_payments.csv #
###################################
print('installments_payment.csv file')

payments = pd.read_csv("C:/Users/Francesco/Desktop/home-credit-default-risk/installments_payments.csv", sep=',')
payments_by_customer = customer_aggregation(payments, aggregation_variables = ['SK_ID_PREV', 'SK_ID_CURR'], names = ['payments', 'client'])
payments_by_customer_final, payments_by_customer_final_columns, payments_by_customer_pearson_results = pearson_for_others(train, payments_by_customer, threshold=0.02, imputation_method='median') #return most relevant features

# apply transformation and store the df in a html file in order to check it out and find if some features can benefit
# from a trasformation
Skewness_and_Kurtosis_num(payments_by_customer_final, list(payments_by_customer_final.columns))\
    .to_html('html_summary_pages/payments_by_client_FINAL_skewness_&_kurtosis.html') # check skew and kurtosis for selected features

# features benefit of transformation:
no_transform = ['SK_ID_CURR', 'client_payments_DAYS_ENTRY_PAYMENT_mean_mean', 'client_payments_DAYS_ENTRY_PAYMENT_min_mean', 'client_payments_DAYS_INSTALMENT_min_mean' ]
ln_transform = ['client_payments_NUM_INSTALMENT_NUMBER_max_min']
cbrt_transform = [col for col in payments_by_customer_final.columns if col not in no_transform+ln_transform]

payments_by_customer_final = apply_transformation_definitely(payments_by_customer_final, sqrt_transformation=[],
                                                           cbrt_transformation=cbrt_transform, ln_transformation=ln_transform,
                                                           square_transformation=[]) #transform features and return the final bureau balance

payments_by_customer_final.to_csv('payments_by_customer_final.csv', index=False) # store file into .csv in order to don't generate it every time

del no_transform, ln_transform, cbrt_transform, payments_by_customer_final_columns, payments_by_customer_pearson_results

####################################################
# [4] MERGING NEW FEATURES DFs WITH TRAIN AND TEST #
####################################################

imputer = SimpleImputer(missing_values=np.nan, strategy='median')  # define imputer object to fill NaN values
print('Shapes before merging')
print('train_for_models shape: {}'.format(train_for_models.shape), 'test_for_models shape: {}'.format(test_for_models.shape))
dataset_list = list()
for dataset in [train_for_models, test_for_models]:
    for information in [bureau_final, bureau_balance_final, previous_app_final, cash_by_customer_final, credit_by_customer_final, payments_by_customer_final]:
        dataset = dataset.merge(information, on='SK_ID_CURR', how='left')
    columns_to_use = dataset.columns
    dataset_list.append(pd.DataFrame(imputer.fit_transform(dataset), columns=columns_to_use))
train_for_models = dataset_list[0]
test_for_models = dataset_list[1]
del bureau_final, bureau_balance_final, previous_app_final, cash_by_customer_final, credit_by_customer_final, dataset_list
print('shapes after merging')
print('train_for_models shape: {}'.format(train_for_models.shape), 'test_for_models shape: {}'.format(test_for_models.shape))


###############################
# [5] CHECK MULTICOLLINEARITY #
###############################

'''
Multicollinearity is a problem in statistics and machine learning, because it describes the situation in which 2 independent
features are strongly correlated with each other meaning that, when they are both present, do not add much more information.
This means that, if both present, they just increase the complexity of the model, for that reason it is recommended to avoid it.
For that reason, the "detect_multicollinearity" function is involved to check in the final dataframe if multicollinearity occurs among all
the features. If it happens, the feature with lowest Pearson Correlation Coefficent computed with the "TARGET" will be discarded.
'''

multico_features = detect_multicollinearity(train_for_models, threshold=0.8) # features that have to be dropped

# It has been decided of not deleteing features do to the fact that there is worsening on the predictions
train_for_models_without_features = train_for_models.drop(columns=multico_features)

#stores features into .txt file
with open('multico_features.txt', 'w') as file:
    for elm in multico_features:
        file.write('%s\n' % elm)

##################################
# [6] ALLIGN TRAIN AND TEST SETS #
##################################

target_labels = train_for_models['TARGET'] # store the target for future
train_for_models = train_for_models.drop(columns='TARGET')
#train_for_models, train_for_models_final_columns, train_for_models_pearson_results = pearson_for_other_files(train, train_for_models2, threshold=0.03) #return most relevant features
train_for_models.drop(columns='SK_ID_CURR', inplace=True)
#test_for_models.drop(columns='SK_ID_CURR', inplace=True)

print('train and test shape BEFORE allignment: train={}, test={}'.format(train_for_models.shape, test_for_models.shape))
print('Aligning the datasets...')
train_for_models, test_for_models = train_for_models.align(test_for_models, join='inner', axis=1) # Align the training and testing data by keeping only shared columns
print('train and test shape AFTER alignment: train={}, test={}'.format(train_for_models.shape, test_for_models.shape))

train_for_models['TARGET'] = target_labels # Add the target back in

#############################################################
# [7] STANDARDIZE DATA, PCA and SPLIT IN TRAIN AND VALIDATION SETs #
#############################################################

df_train = train_for_models.copy().sample(n=1000, random_state=100)
y = df_train['TARGET']
df_train.drop(['TARGET'], axis=1, inplace = True)

scaler = StandardScaler() # Initialize instance of StandardScaler
df_train = scaler.fit_transform(df_train) # Fit and transform the data (excluded target feature)
df_train = pd.DataFrame(data = df_train, columns = [f for f in train_for_models.columns if f not in ['TARGET']])

print('Computing PCA and storing the image inside images folder')
PCA_plot(df_train, 175, .0, .9)
df_train_PCA = pca_function(df_train, 175)


# Split into train and test set.
X = df_train.copy()
#y = train_for_models['TARGET']

print('Slit in train (70%) and validation (30%) sets')
# split the df in train(70%) and test(30%) set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#############################
# [8] HYPERPARAMETER TUNING #
#############################

classifiers = {
    'knn': KNeighborsClassifier(),
    #'svm': SVC()#,
    'rf': RandomForestClassifier(),
    'ada': AdaBoostClassifier(),
    'gbc': GradientBoostingClassifier()

}
print('starts tuning the hyperparameters for: {} models'.format(str([k for k in classifiers.keys()])[1:-1]))
paramiters = {

    'knn': {'n_neighbors': [int(x) for x in np.linspace(1, 200, num=5)],
           'weights': ['uniform', 'distance']},

#    'svm': [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
#             'C': [10, 100, 1000]},
#            #{'kernel': ['linear'], 'C': [10, 100, 1000]}
#           ],
    'rf': {
        'n_estimators': [int(x) for x in np.linspace(100, 200, num=2)],
        'max_depth': [int(x) for x in np.linspace(10, 100, 2)],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True]
    },
   'ada': {
        'n_estimators': [n for n in range(100, 500, 100)],
        'learning_rate': [0.01, .1]
   },
    'gbc': {
        'loss': ['deviance'],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [n for n in range(100, 500, 100)],
        'subsample':[1.0],
        'min_samples_split':[10],
        'max_depth': [int(x) for x in np.linspace(10, 100, 2)],
        'max_features': ['auto', 'sqrt', 'log2'],

    }
}

GS_scores = {}  # dictionary where are stored all the scores
GS_best_scores = {}  # dictionary where are stored only the best scores per each model
GS_best_paramiters = {}  # dictionary where are stored only the best models
GS_best_estimators = {}  # dictionary where are are stored best estimators (so we don't need to write them)

for k in classifiers.keys():
    # we run up to all the models are tuned
    grid = GridSearchCV(
        classifiers[k],
        paramiters[k],
        cv=KFold(n_splits=10, random_state=25, shuffle=True), scoring='roc_auc', n_jobs=-1)

    grid.fit(X_train, y_train)
    # fitted_models['rf'] = grid
    GS_scores[k] = grid.cv_results_
    GS_best_scores[k] = grid.best_score_
    GS_best_paramiters[k] = grid.best_params_
    GS_best_estimators[k] = grid.best_estimator_

print('Models have been trained!')

######################
# [9] DO PREDICTIONS #
######################
print('Start doing predictions...')
df_test = test_for_models.copy()
df_test = df_test.fillna(0)
scaler = StandardScaler() # Initialize instance of StandardScaler
df_test = scaler.fit_transform(df_test) # Fit and transform the data
df_test = pd.DataFrame(data = df_test, columns = [f for f in test_for_models.columns ])


predictions = GS_best_estimators['ada'].predict_proba(df_test)
df_predictions = pd.DataFrame()
df_predictions['SK_ID_CURR'] = test['SK_ID_CURR']
df_predictions['TARGET'] = predictions[:,1]

df_predictions
df_predictions.to_csv('submission_ada.csv',index=False)
print('"submission.csv file has been created!"')

###### PRINT ELAPSED TIME AND DATE ######
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
end_time = time.time()
duration = (end_time - starttime)/60
intero = int(duration)
decimal = int((duration - intero)*60)

print('\n', 'Process completed in {}:{} minutes '.format(intero, decimal), sep='\n')
print('Today: ', date_time)





