# General imports
import warnings
warnings.filterwarnings("ignore")

# data
import pickle

# visualize
import matplotlib.pyplot as plt

# model
from lightgbm import LGBMRegressor

from engine.features import Features
from engine.train import *

###############################################################################
################################# Load Data ###################################
##############################################################################
## Import 2 types of dataset
## Descriptions:
#   hung1 : hungarian input for no hierarchical model
#   hung2 : hungarian input for hierarchical model


t1 = Features(types="hungarian_h1", not_divided=True)
hung1 = t1.run_hungarian()
hung1_cols = hung1.columns.to_list()  # check!
hung1_times = hung1.iloc[:125]['방송일시']  # for output

t2 = Features(types="hungarian", not_divided=True)
hung2 = t2.run_hungarian()
hung2_cols = hung2.columns.to_list()
hung2_times = hung2.iloc[:660]['방송일시']  # for output

####################################################################
########################### New train ##############################
####################################################################
"""
Adjust full preprocess train dataset for modeling
as we cannot define some time lag features for hungarian input.
Drop those columns and train 
"""
df_full_lag = pd.read_pickle(FEATURED_DATA_DIR + "/train_fin_light_ver.pkl")
df_full_lag = df_full_lag[hung1_cols+['취급액']]
df_full_test = pd.read_pickle(FEATURED_DATA_DIR + "/test_fin_light_ver.pkl")
df_full_test = df_full_test[hung1_cols+['취급액']]

# combined data for label encoding
tmp_combined = pd.concat([df_full_lag, df_full_test, hung1, hung2])
# Preprocessed datasets
tmp_combined = run_preprocess(tmp_combined)
df_full_lag_PP = tmp_combined.iloc[:df_full_lag.shape[0]].reset_index()
df_full_test_PP = tmp_combined.iloc[df_full_lag.shape[0]:(df_full_lag.shape[0]+df_full_test.shape[0])].reset_index()
hung1_PP = tmp_combined.iloc[(df_full_lag.shape[0]+df_full_test.shape[0]):(df_full_lag.shape[0]+
                                                                           df_full_test.shape[0]+hung1.shape[0])]
hung2_PP = tmp_combined.iloc[-hung2.shape[0]:]

# for opt model
hung_list = hung2[['상품코드', '상품명']].drop_duplicates()  # item list for hung2
hung_list['row_num'] = hung2_PP.상품코드.unique()

train_x, train_y, val_x, val_y = divide_train_val(df_full_lag_PP, 8, drop=['original_c', '상품코드', 'show_id'])
top_opt, top_tr_opt_x, top_tr_opt_y, top_v_opt_x, top_v_opt_y = divide_top(df_full_lag_PP, 5500, 2500)
top_tr_opt_x.drop(columns=['상품코드', 'original_c'], inplace=True)
top_v_opt_x.drop(columns=['상품코드', 'original_c'], inplace=True)

####################################################################
########################### Light GBM ##############################
####################################################################

def run_hung_lgbm():
    params_all_opt = params_all_wd.copy()
    params_all_opt['categorical_feature'] = [4, 5, 6, 8]
    params_top_opt = params_top_wd.copy()
    params_top_opt['categorical_feature'] = [4, 5, 6, 8]

    # base model
    run_lgbm(params_all_opt, train_x, train_y, val_x, val_y, 'all_opt')
    # top model
    run_lgbm(params_top_opt, top_tr_opt_x, top_tr_opt_y, top_v_opt_x, top_v_opt_y, 'top_opt')


if __name__ == "__main__":
    run_hung_lgbm()
