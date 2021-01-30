import os
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


def directory_function(path):
    '''
    It is a function used to create a directory
    :param path: the directory path
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        print('{} folder created!'.format(path))

def counting_columns_with_NaN(df, printing):
    '''
    Fuction to count NaN values in a dataframe
    :param df: the dataframe that has to be checked out
    :return: a serie where the index are the features with NaN and the column says how many NaN are per each of them
    '''
    missing_info = df.isnull().sum().rename('Missed Values')
    missing_info = missing_info[missing_info.iloc[:] != 0].sort_values(ascending=False)
    if printing:
        print('There are {}  out of {} columns with missing values'.format(
            str(missing_info.shape[0]),
            str(df.shape[1])))
    return missing_info

def imputation_function(df, printing):
    '''
    :param df: the dataframe that has to be imputed (it will do only on quantitative variables)
    :return: the dataframe
    '''
    #### IMPUTATION ON NUMERICAL FEATURES ####
    print('Do imputation on numerical features by using column mean...')

    missing_info = counting_columns_with_NaN(df, printing)
    cols_list = [rows[0] for index, rows in missing_info.reset_index().iterrows()
                 if (df[rows[0]].dtype == np.float or df[rows[0]].dtype == np.int64)]  # get columns names if int or float

    for e,elm in enumerate(cols_list):
        df[elm].fillna(df[elm].mean(), inplace=True) #impute every column with their mean value
    return df

def describe_function(df):
    '''
    :param df: a dataframe to be checked
    :return: a dataframe with statistics information
    '''
    cols = [col for col in df.columns if (df[col].dtype == np.float or df[col].dtype == np.int64)] # get only columns that are not objects
    df_list = list()
    for col in cols: # itereate over the columns
        dict_info = {'column':col}
        for index, row, in df[col].describe().reset_index().iterrows():
            dict_info[row[0]] = round(row[1],1) # get statistic name and associated value
        df_list.append(pd.DataFrame(dict_info, index=[0])) # convert into df and append to the list
    descriptive_df = pd.concat(df_list) # concatenate the df
    return descriptive_df

def from_days_to_years(df, old_columns):
    '''
    :param df: the dataframe to be modified
    :param old_columns: list of the columns that require to be converted from days to years
    :return: the modified datarame
    '''

    for col in old_columns:
        df[col+'_IN_YEARS'] = df[col] / -365
    df.drop(labels=col, axis=1, inplace=True)
    return df

def fix_DAYS_EMPLOYED_feature(df):
    '''
    :param df: The dataframe to be modified
    :return: the dataframe
    '''
    df['DAYS_EMPLOYED_ANOMALOUS'] = np.where(df['DAYS_EMPLOYED'] == 365243, 1, 0) # create the boolean column
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True) # substitute the anomalous observations with NaN
    df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median(), inplace=True) # do imputation on the previous created NaN values
    return df

def economical_features(df):
    df['CREDIT_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    return df

def pearson_correaltion_coefficient(df, top=10):
    '''
    :param df: it takes in input the dataframe that containes the features
    :return: two dataframes one with the top 10 most positive, and the on the other one the top 10 most negative
    '''
    print('\n', 'Computing Pearson Correlation coefficient...', sep='\n')
    pearson_results = df.corr(method='pearson')['TARGET'].sort_values()
    top_negative = pearson_results.head(top)
    top_positive = pearson_results.tail(top+1)[:-1].sort_values(ascending=False)
    # sort in descending order and exclude the TARGET, this is done because it also returns the correlation among
    # the target-target, which is of course one, and we don't want to consider it into the top 10,
    # for that reason it has been picked the top 11 and discarded the first one.
    print('Done !')
    return top_negative, top_positive, pearson_results

def heat_map_function(df, title='Correlation Map', fig_size=(10,8), size_num=10, labelsize=6, name_image='correlation'):

    directory_function('images/correlation_map')
    plt.ioff()  # deactivate interactive mode (it will show the plot iff you requested: plt.show())
    corr = round(df.corr(), 3)

    # plot figure
    fig, ax = plt.subplots(figsize = fig_size)
    plt.title(title, y=1.5, fontsize=18, color='orange', fontweight="bold")
    sns.heatmap(corr, cmap= 'YlGnBu', annot=True, annot_kws={"size": size_num}, linewidth=3, ax =ax);
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    plt.savefig('images/correlation_map/{}.png'.format(name_image));
    plt.close(fig)

def qq_plot(df, feature_name, name_image):
    '''
    Plot a density plot and qq-plot for a selected feature
    :param df: The dataframe containing the data
    :param feature_name: the feature for which is necessary compute the qqplot
    :param name_image: the name used to save the figure
    :return: nothing, it saves the image into the images folder
    '''

    directory_function('images/qq_plot') # create directory where store images
    df = df[feature_name]
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)  # distribution plot
    ax2 = fig.add_subplot(122)  # QQ Plot
    # distribution plot
    sns.distplot(df,
                 kde_kws={"color": "g", "lw": 3, "label": "Kernal Density Estimation"},
                 hist_kws={"histtype": "step", "linewidth": 3,
                           "alpha": 1, "color": "b"}, norm_hist=False, ax=ax1);  # distribution plot
    stats.probplot(df, plot=sns.mpl.pyplot);  # QQ Plot

    ax1.set_title(str(feature_name) + ' Distribution', color='k', fontsize=25)
    ax1.set(xlabel=feature_name)
    ax1.xaxis.labelpad = 20
    ax1.xaxis.label.set_size(20)

    ax2.set_title('QQ Plot ' + str(feature_name), color='k', fontsize=25)
    ax2.xaxis.labelpad = 20
    ax2.xaxis.label.set_size(20)
    ax2.yaxis.label.set_size(20)
    ax2.get_lines()[0].set_marker('+')  # QQ plot markers
    ax2.get_lines()[0].set_markersize(12.0)  # QQ plot markers size

    plt.savefig('images/qq_plot/qqplot_{}.png'.format(name_image));
    plt.close(fig)


def transformation_function(df, feature_name):
    '''
    'transformation_function' is a function that execute different transformation on a selected feature.
    :param df: the used dataframe
    :param feature_name: the feature that is required to transform
    :return: - T_results = which is a dict containing skweness and kurtosis for each transformation
             - T_df_results = is a dict contaning per each transformation
    '''
    T_results = defaultdict(list)
    T_df_results = defaultdict()

    # ln(x)
    Ln_transformation = np.log(df[feature_name])  # function computes the value of ln(x)
    T_results[str(feature_name) + '_Ln'].extend(
        [round(Ln_transformation.skew(), 4), round(Ln_transformation.kurt(), 4)])  # compute skewness and kurtosis
    T_df_results[str(feature_name) + '_Ln'] = Ln_transformation

    # sqrt(x)
    root_transformation = np.sqrt(df[str(feature_name)])  # returns the sqrt(x)
    T_results[str(feature_name) + '_root'].extend(
        [round(root_transformation.skew(), 4), round(root_transformation.kurt(), 4)])  # compute skewness and kurtosis
    T_df_results[str(feature_name) + '_root'] = root_transformation

    # cbrt(x)
    croot_transformation = np.cbrt(df[str(feature_name)])  # returns the cbrt(x)
    T_results[str(feature_name) + '_croot'].extend(
        [round(croot_transformation.skew(), 4), round(croot_transformation.kurt(), 4)])  # compute skewness and kurtosis
    T_df_results[str(feature_name) + '_croot'] = croot_transformation

    # square(x)
    square_transformation = np.square(df[str(feature_name)])  # returns the sqrt(x)
    T_results[str(feature_name) + '_square'].extend(
        [round(square_transformation.skew(), 4), round(square_transformation.kurt(), 4)])  # compute skewness and kurtosis
    T_df_results[str(feature_name) + '_square'] = square_transformation

    return T_results, T_df_results


def skewness_kurtosis_plots(df, feature_name, name_image):
    '''
    Function that plot distribution of a particoular feature after 'transformation_function'
    :param df: the dataframe to use
    :param feature_name: the feature that it is requested to plot
    :return:
    '''

    directory_function('images/skew_and_kurt')
    #plt.ioff()  # deactivate interactive mode (it will show the plot iff you requested: plt.show())
    T_results, T_df_results = transformation_function(df, feature_name)

    colors = ['darkorange', 'dodgerblue', 'seagreen', 'mediumvioletred', 'darkslategray', 'maroon']
    key_list = [k for k in T_results.keys()]  # list with all the transformation

    fig = plt.figure(figsize=(20, 15))

    if feature_name == 'TARGET':
        fig.suptitle('Target Transformation Distribution', fontsize=20)
    else:
        fig.suptitle(feature_name + ' Transformation Distribution', fontsize=20)

    for n, k in enumerate(key_list):
        try:  # because if there are 0 it is not possible to compute the log
            ax = fig.add_subplot(2, 2, (n + 1))
            sns.distplot(T_df_results[k], color=colors[n], kde_kws={"label": "Kernal Density Estimation"})
            ax.set_title(k+ ', Skewness: ' + str(T_results[k][0]) + ' Kurtosis: ' + str(T_results[k][1]),
                         color='Red', fontsize=15)
            ax.set(xlabel=k.split("_")[1] + str(feature_name))
            ax.xaxis.labelpad = 8
            ax.xaxis.label.set_size(12)
        except:
            continue

    plt.savefig('images/skew_and_kurt/skew_kurt_plot_{}.png'.format(name_image));
    plt.close(fig)

# 'Skewness_and_Kurtosis_num' is a function that compute Skewness and Kurtosis on the selected features. (In this case Numerical)
def Skewness_and_Kurtosis_num (df, numerical_features):
    '''
    :param df: the dataframe containing the data
    :param numerical_features: a list of all the features for which is required compute the skewness and kurtosis
    :return: a dataframe containing skewness and kurtosis for each feature
    '''

    numerical_S_and_K = defaultdict() # dictionary where--> 'Feature Name': [Skewness, Kurtosis]
    for f in numerical_features:
        numerical_S_and_K[f]= [round(df[f].skew(), 3), round(df[f].kurt(), 3),
                               round(np.log(df[f]).skew(),3), round(np.log(df[f]).kurt(),3),
                               round(np.sqrt(df[f]).skew(), 3), round(np.sqrt(df[f]).kurt(), 3),
                               round(np.square(df[f]).skew(), 3), round(np.square(df[f]).kurt(), 3),
                               round(np.cbrt(df[f]).skew(), 3), round(np.cbrt(df[f]).kurt(), 3)
                               ] # compute Skewness and Kurtosis

    df_s_and_k = pd.DataFrame.from_dict(numerical_S_and_K,
                                        orient='index',
                                        columns = ['Skewness', 'Kurtosis','S_ln', 'K_ln', 'S_sqrt', 'K_sqrt',
                                                   'S_square', 'K_square', 'S_cbrt', 'K_cbrt']) # convert dictionary into df
    return (df_s_and_k)


def apply_transformation_definitely(df, sqrt_transformation, cbrt_transformation, ln_transformation, square_transformation):

    print('\n', 'Start applying transformation on selected features...', sep='\n')
    # SQRT transformation
    if len(sqrt_transformation) != 0:
        for sqrt in sqrt_transformation:
            df['{}_sqrt'.format(sqrt)] = np.sqrt(df[sqrt]) # add transformed column
            df.drop(columns=[sqrt], inplace=True)  # drop column
    # SQUARE transformation
    if len(square_transformation) != 0:
        for square in square_transformation:
            df['{}_square'.format(square)] = np.sqrt(df[square]) # add transformed column
            df.drop(columns=[square], inplace=True)  # drop column
    # CBRT transformation
    if len(cbrt_transformation) != 0:
        for cbrt in cbrt_transformation:
            df['{}_cbrt'.format(cbrt)] = np.cbrt(df[cbrt]) # add transformed column
            df.drop(columns=[cbrt], inplace=True)  # drop column
    # LN transformation
    if len(ln_transformation) != 0:
        for ln in ln_transformation:
            df['{}_ln'.format(ln)] = np.log(df[ln]) # add transformed column
            df.drop(columns=[ln], inplace=True)  # drop column
    print('Done, all selected feature transformed and stored inside "df_transformed_features" dataframe!', '\n\n')
    return df


def label_and_dummie_encoding(df):
# mettici un if* per le altre variabili per le quali vuoi usere one hot encoding una volta fatto
    encoder = LabelEncoder() # encoder object
    for col in df.columns: # iterate over columns
        if df[col].dtype == 'object': # selected only categorical columns
            if len(list(df[col].unique())) <=2: # if there less then 3 unique entries uses label encoding
                encoder.fit_transform(df[col])
    df = pd.get_dummies(df) # uses the get_dummies function for one-hot-encoding
    return df



def imputing_category(df):

    missing_info = counting_columns_with_NaN(df, printing=False)
    missing_columns = list(missing_info.index)

    for col in missing_columns:
        df[col] = df[col].astype('category') # from object to category (it is necessary for encoding)
        df[col] = df[col].cat.codes # encode into numbers
    df.replace(-1, np.nan, inplace=True) # replace -1 with nan because we want to find out them using knn

    print('\n', 'Label and dummies encoding', sep='\n')
    df = label_and_dummie_encoding(df)
    print('Start imputing features using most frequent approach...')
    #imputer = KNNImputer(n_neighbors=2, weights="uniform")
    #imputer.fit_transform(df[missing_columns[0]])
    dummie_df = df[missing_columns]
    simple_imp = SimpleImputer(strategy="most_frequent") # define imputer object
    dummie_df = pd.DataFrame(data=simple_imp.fit_transform(dummie_df), columns=missing_columns) # give it to the df

    dummie_df = pd.get_dummies(dummie_df[missing_columns].astype('category'))  # use the get_dummies function for one-hot-encoding
    df.drop(missing_columns, axis=1, inplace=True)
    df_full = pd.concat([df, dummie_df], axis = 1)

    return df_full, missing_columns


def statistics_for_numerical_and_categorical(df, aggregation_variable, new_var_name, category):

    group_ids = df[aggregation_variable]  # store grouping variable into serie, it is necessary because sometimes we group by objects

    if category == 'numerical':
        print('Compute numerical statistics for {} file'.format(new_var_name))
        for col in df:
            if col != aggregation_variable and 'SK_ID' in col:
                df = df.drop(columns=col)  # it removes the cols that are not required for grouping

        new_df = df.select_dtypes('number')  # discard object dtype columns
        new_df[aggregation_variable] = group_ids  # add again the grouping variable
        new_df = new_df.groupby(aggregation_variable).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()  # compute stats
        # from multiindex (2 levels) to single index
        accepted_columns = ['{}_{}_{}'.format(new_var_name, var, stat) for var in new_df.columns.levels[0] if var != aggregation_variable for stat in new_df.columns.levels[1][:-1]]
        accepted_columns.insert(0, aggregation_variable)


    elif category == 'categorical':
        print('Compute categorical statistics for {} file'.format(new_var_name))
        new_df = pd.get_dummies(df.select_dtypes('object'))  # pick on il categorical variable and do one-hot-encoding
        new_df[aggregation_variable] = group_ids  # insert person id that will be used to groupby
        new_df = new_df.groupby(aggregation_variable).agg(['sum', 'mean'])  # compute sum and mean per each person
        # from multiindex (2 levels) to single index
        accepted_columns = ['{}_{}_{}'.format(new_var_name, var, stat)
                           for var in new_df.columns.levels[0]  # iterate over the first level
                           for stat in ['sum', 'mean']]  # iterate over the second level


    new_df.columns = accepted_columns
    _, idx = np.unique(new_df, axis = 1, return_index = True)
    new_df = new_df.iloc[:, idx]
    return new_df


def pearson_for_others(main_df, df, threshold=0.03, imputation_method='median'):
    '''
    :param df: the dataframe to be processed
    :param threshold: the correlation threshold
    :return: the df with highest correlation columns, and the df with all the corr score (only for checking)
    '''
    print('Pre-processing for Pearson Correlation coefficient...')
    id_and_target = main_df[['SK_ID_CURR', 'TARGET']]
    df_combined = df.merge(id_and_target, on='SK_ID_CURR', how='left')

    print('imputation by using {}'.format(imputation_method))
    imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_method)  # define imputer object to fill NaN values
    columns_to_use = df_combined.columns # columns name
    df_combined = pd.DataFrame(imputer.fit_transform(df_combined), columns=columns_to_use) # do imputation and return a df

    print('Computing Pearson Correlation coefficient...')
    pearson_results = df_combined.corr(method='pearson')['TARGET'].sort_values() # compute corr coefficient

    columns_accepted = ['SK_ID_CURR']
    for index, v in pearson_results.iteritems():  # iterate over index and column
        if index != 'TARGET':  # if not target
            if abs(v) >= threshold:  # if the corr is greater than the threshold, take it otherwise discard
                columns_accepted.append(index)

    print('Computing dataframe...')
    df = df[columns_accepted]
    print('Done for correlation coefficient!')
    del id_and_target, df_combined, index, v
    return df, columns_accepted, pearson_results

def merge_df(df1, df2, merging_on):
    df_combined = df1.merge(df2, on=merging_on, how='left')
    return df_combined


def customer_aggregation(df, aggregation_variables, names):
    '''
    :param df: the input dataframe that need to aggregate per each client
    :param aggregation_variables: the variables used to do aggregations
    :param names: the initial name to assign to every created statistical column
    :return:
    '''

    # Compute statistics and aggregate the numerical columns
    df_agg = statistics_for_numerical_and_categorical(df, aggregation_variable=aggregation_variables[0], new_var_name=names[0], category='numerical')
    # If there are categorical variables
    if any(df.dtypes == 'object'):
        # compute statisctics for categorical columns
        counts = statistics_for_numerical_and_categorical(df, aggregation_variable=aggregation_variables[0], new_var_name=names[0], category='categorical')
        # combine the precompute statistics for numerical with the statistics of the categorical features
        df_by_loan = counts.merge(df_agg, on=aggregation_variables[0], how='outer')

        # combine the df_by_loan with the input df in order to get the clients id
        df_by_loan = df_by_loan.merge(df[[aggregation_variables[0], aggregation_variables[1]]], on=aggregation_variables[0], how='left')
        # drop the loan id
        df_by_loan.drop(columns=[aggregation_variables[0]], inplace=True)
        # Aggregate numeric stats by column
        df_by_client = statistics_for_numerical_and_categorical(df_by_loan, aggregation_variable=aggregation_variables[1], new_var_name=names[1], category='numerical')
        del df_agg, counts, df_by_loan
    # No categorical variables
    else:
        # get the clients id from the input df
        df_by_loan = df_agg.merge(df[[aggregation_variables[0], aggregation_variables[1]]], on=aggregation_variables[0], how='left')
        # Remove the loan id
        df_by_loan.drop(columns=[aggregation_variables[0]], inplace=True)
        # compute statistics for every client
        df_by_client = statistics_for_numerical_and_categorical(df_by_loan, aggregation_variable=aggregation_variables[1], new_var_name=names[1], category='numerical')
        del df_agg,  df_by_loan

    return df_by_client


def detect_multicollinearity(df, threshold=0.8):
    print('Computing correlation among all ({}) features in the given dataframe...'.format(len(df.columns)))
    corr = round(df.corr(), 3) # compute correlation
    index_col = list(corr.index) # store index
    columns = list(corr.columns) # store columns
    corr_array = np.array(corr) # convert into array

    print('Start checking for multicolinearity')
    columns_to_drop = list()
    target_array = corr_array[index_col.index('TARGET')] # array where the are the correlations among the target and the features
    for n,row in enumerate(corr_array): #iterate over rows
        if index_col[n] != 'TARGET': # if it is the target, just skip
            for e,elm in enumerate(row): #iterate over every element of the array
                if elm >= threshold and elm != 1: # if the corr is above or equal to a threshold then
                    if abs(target_array[n]) >= abs(target_array[e]): # select the feature with the lowest correlation with the target to be dropped
                        columns_to_drop.append(columns[e])
                    else:
                        columns_to_drop.append(index_col[n])
    columns_to_drop = set(columns_to_drop)
    print('There are {} columns that have to be dropped'.format(len(columns_to_drop)))
    return columns_to_drop


def PCA_plot(df, x, ymin, ymax):

    #Fit the PCA algorithm with our Data
    pca = PCA().fit(df.iloc[:,3:])

    fig, ax = plt.subplots(figsize = (15,12))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='green')
    plt.title('PCA analysis', fontsize = 20)
    plt.xlabel('Number of Components', fontsize = 18)
    plt.ylabel('Variance (%)', fontsize = 18)

    ax.axvline(x=x, ymin=ymin, ymax=ymax,color='red')
    #ax.hlines(y=0.975, xmin=0.88, xmax=29, color='red')

    plt.savefig('images/PCA.png');
    plt.close(fig)


def pca_function(Data_Frame, components):
    # movements = {'left':1,'right':2, 'up':3, 'down':4, 'square':5, 'triangle':6, 'circleCw':7, 'circleCcw':8}

    pca = PCA(n_components=components)  # we decide the number of components
    pca_data = pca.fit_transform(Data_Frame)
    columns = ['PC_{}'.format(n) for n in range(components)]
    new_df = pd.DataFrame(pca_data, columns=columns)
    return new_df

