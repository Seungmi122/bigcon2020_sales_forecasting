# directories

LOCAL_DIR = ".."
MODELS_DIR = LOCAL_DIR + "/data/saved_models/"
FEATURED_DATA_DIR = LOCAL_DIR + '/data/fin_data/'
SUBMISSION_DIR = LOCAL_DIR + "/submission/"

# set some Global Vars

data_list = ['df_wk_lag', 'df_wk_no_lag', 'df_wd_lag', 'df_wd_no_lag', 'df_all_lag']

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