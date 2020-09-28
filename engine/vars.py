import os
#####################
###### TARGET #######
#####################
TARGET = '취급액'

#####################
#### DIRECTORIES ####
#####################
LOCAL_DIR = "../"
MODELS_DIR = LOCAL_DIR + "data/saved_models/"
FEATURED_DATA_DIR = LOCAL_DIR + 'data/20/'
SUBMISSION_DIR = LOCAL_DIR + "submission/"
OPT_DATA_DIR = LOCAL_DIR + "data/13/"
RAW_DATA_DIR = LOCAL_DIR + "data/00/"
PLOT_DIR = LOCAL_DIR + "plot/"

#####################
#### GLOBAL VARS ####
#####################
lag_col1 = ['lag_scode_price', 'lag_scode_count', 'lag_mcode_price', 'lag_mcode_count', 'lag_bigcat_price',
            'lag_bigcat_count', 'lag_bigcat_price_day', 'lag_bigcat_count_day', 'lag_small_c_price',
            'lag_small_c_count', 'lag_all_price_show', 'lag_all_price_day']

lag_col2 = ['ts_pred', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_21',
            'rolling_mean_28', 'mean_sales_origin']

lag_wd = ['lag_sales_wd_1', 'lag_sales_wd_2', 'lag_sales_wd_3','lag_sales_wd_4', 'lag_sales_wd_5']
lag_wk = ['lag_sales_wk_1', 'lag_sales_wk_2']
full_lag_col = ['lag_sales_1', 'lag_sales_2', 'lag_sales_5', 'lag_sales_7']

cat_col = ['상품군', 'weekdays', 'show_id', 'small_c', 'middle_c', 'big_c',
           'pay', 'months', 'hours_inweek', 'weekends', 'japp', 'parttime',
           'min_start', 'primetime', 'prime_smallc',
           'freq', 'bpower', 'steady', 'men', 'pay', 'luxury',
           'spring', 'summer', 'fall', 'winter', 'rain']

encoded_cols = ['상품코드', '상품군', 'weekdays', 'parttime', 'show_id','small_c', 'middle_c',
                'big_c', 'original_c', 'pay', 'exposed_t']

base_cols = ['방송일시', '노출(분)', '마더코드', '상품코드', '상품명', '상품군', '판매단가', '취급액']