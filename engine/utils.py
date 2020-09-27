# General imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from engine.vars import *
# if separate df with all features is existent, set path
# otherwise merge raw df with Features module

TARGET = '취급액'

#########################
## For Data Preprocessing
#########################


def load_df(path):
    """
    :objective: load data
    :return: pandas dataframe
    """
    try:
        df = pd.read_pickle(path)
        return df.reset_index()
    except:
        print("check file directory")


def drop_useless(df):
    """
    :objective: drop useless features for model.
    :return: pandas dataframe
    """
    #useless features
    xcol = ['방송일시', '노출(분)', '마더코드', '상품명', 'exposed', 'ymd', 'volume',
            'years', 'days', 'hours', 'week_num', 'holidays', 'red', 'min_range', 'brand',
            'small_c_code', 'middle_c_code', 'big_c_code', 'sales_power']
    col = [x for x in df.columns if x in xcol]
    df = df.drop(columns=col)
    df = df.copy()
    return df


def check_na(df):
    """
    :objective: show na
    :return: columns with na / na counts
    """
    print(df.isnull().sum())


# 새로운 변수에 na 있으면 일단 imputation 따로 해야함. 아니면 여기 list에 변수 이름 추가해도 됨.
def na_to_zeroes(df):
    """
    :objective: Change all na's to zero.(just for original lag!)
    :return: pandas dataframe
    """
    xcol = [x for x in df.columns if
            x in lag_col1 + lag_col2 + ['mid_click_r', 'age30_middle', 'age40_middle', 'age50_middle',
                                        'age60above_middle', 'pc_middle', 'mobile_middle']]
    for col in xcol:
        df[col] = df[col].fillna(0)

    return df


## Encoding
# One-hot-Encoding
def run_onehot(df):
    """
    :objective: Perform ohe for categorical columns
    :return: pandas dataframe
    """
    cats_col = ['min_start','japp','parttime', 'primetime','exposed_t','상품군','weekdays', 'small_c','middle_c','big_c','pay','men']
    num = df.drop(columns = cats_col)
    X1 = df[cats_col]
    # Onehotencoder
    ohe = OneHotEncoder()
    ohe.fit(X1)
    onehotlabels = ohe.transform(X1).toarray()
    cat_labels = ohe.get_feature_names(['min_start','japp','parttime', 'primetime','exposed','상품군','weekdays', 'small_c','middle_c','big_c', 'pay','men'])
    cat = pd.DataFrame(onehotlabels, columns=cat_labels)
    df_ohe = num.join(cat)

    return df_ohe

# Label Encoding


def get_label_features(df):
    """
    :objective: Show features that need labelencoding
    :return: list
    """
    lab_col = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return lab_col


def get_label_mapping(le):
    """
    :objective: Return a dict mapping labels to their integer values / Run right after 'run_label_separately' / le = a fitted SKlearn LabelEncoder
    :return: integer
    """
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

#integerMapping = get_label_mapping(le)
#integerMapping['Monday'] #하나만 보고 싶을 때
#이미 한번 인코딩을 한 변수에 또 labelencoding을 할 경우 mapping 못불러옴
#get_label_features 아웃풋 보고 한번씩 separate하게 인코딩 하고, 인코딩 한번 할 때마다 get_label_mapping(le)으로 확인하는거 추천
#get_label_features 확인 -> run_label_separately(df, colname) -> get_label_mapping(le) 저장 -> run_label_separately(df, colname) -> get_label_mapping(le) 저장


# mapping 안궁금하고 그냥 한꺼번에 다 돌리기
def run_label_all(df):
    """
    :objective: Perform labelencoding for all categorical/object columns
    :return: pandas dataframe
    """
    lab_col = df.select_dtypes(include=['object','category']).columns.tolist()
    for col in lab_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)

    return df


# remove outliers
def remove_outliers(df_train):
    """
    :objective: Remove outliers. Before dividing into X/y
    :return: pandas dataframe
    """
    numeric_colnum = df_train.columns.get_indexer(['판매단가', '취급액']).tolist()
    feature_set = df_train.iloc[:, numeric_colnum]
    # identify outliers in the training dataset
    iso = IsolationForest(n_estimators=50, max_samples=50, contamination=float(0.05), max_features=1.0)
    iso.fit(feature_set)
    pred = iso.predict(feature_set)
    feature_set['anomaly'] = pred
    outliers = feature_set.loc[feature_set['anomaly'] == -1]
    outlier_index = list(outliers.index)
    df_train = df_train.loc[~df_train.index.isin(outlier_index)].reset_index()

    return df_train


def run_preprocess(df):
    """
    :objective: Run Feature deletion, NA imputation, label encoding
    :return: pandas dataframe
    """
    df = drop_useless(df)
    df = na_to_zeroes(df)
    # df = remove_outliers(df)
    df = run_label_all(df)
    df1 = df.copy()
    return df1


#####################
## For Model Training
#####################

# Seeder
def seed_everything(seed=127):
    random.seed(seed)
    np.random.seed(seed)

# metrics
# negative mape (For Bayesian Optimization)

def neg_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    result = (-1) * mape
    return result

# MAPE
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    final = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return final

# RMSE
def get_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# MAE
def get_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# CV splits
def cv_split(df, month, printprop=False):
    split = int(df[df['months'] == month].index.values.max())
    prop = str(split / df.shape[0])
    if printprop:
        print(f'Proportion of train set is {prop}')
        return split
    else:
        return split

# Divide into train/test
def divide_train_val(df_pp, month, drop):
    split = cv_split(df=df_pp, month=month)
    train_x = df_pp.iloc[:split, :].drop(columns=['index',
                                                  'show_id', TARGET] + drop)  ## 'index' check!!
    train_y = df_pp.iloc[:split, :][TARGET]
    val_x = df_pp.iloc[split:, :].drop(columns=['index',
                                                'show_id', TARGET] + drop)
    val_y = df_pp.iloc[split:, :][TARGET]
    return train_x, train_y, val_x, val_y


# Divide into high-rank / Low-rank (BY: mean_sales_origin)
def divide_top(df, num_train, num_val):
    top_df = df.sort_values('mean_sales_origin', ascending=False)

    top_tr_lag_x = top_df.iloc[:num_train, :].drop(['index', 'show_id', TARGET], axis=1)
    top_tr_lag_y = top_df.iloc[:num_train, :][TARGET]
    top_v_lag_x = top_df.iloc[num_train:(num_train + num_val), :].drop(['index', 'show_id', TARGET], axis=1)
    top_v_lag_y = top_df.iloc[num_train:(num_train + num_val), :][TARGET]

    return top_df, top_tr_lag_x, top_tr_lag_y, top_v_lag_x, top_v_lag_y
