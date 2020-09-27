# General imports
import warnings
warnings.filterwarnings("ignore")

# data
import pickle

# visualize
import matplotlib.pyplot as plt

# model
from lightgbm import LGBMRegressor

from engine.utils import *
from engine.features import Features
from engine.vars import *

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

# combined data for label encoding
tmp_combined = pd.concat([df_full_lag, hung1, hung2])
# Preprocessed datasets
tmp_combined = run_preprocess(tmp_combined)
df_full_lag_PP = tmp_combined.iloc[:df_full_lag.shape[0]].reset_index()
hung1_PP = tmp_combined.iloc[df_full_lag.shape[0]:(df_full_lag.shape[0]+hung1.shape[0])]
hung2_PP = tmp_combined.iloc[-hung2.shape[0]:]

# for opt model
hung_list = hung2[['상품코드', '상품명']].drop_duplicates() # item list for hung2
hung_list['row_num'] = hung2_PP.상품코드.unique()

train_x, train_y, val_x, val_y = divide_train_val(df_full_lag_PP, 8, drop=['original_c', '상품코드', 'show_id'])
####################################################################
########################### Light GBM ##############################
####################################################################
params = {
            'feature_fraction': 1,
            'learning_rate': 0.001,
            'min_data_in_leaf': 135,
            'n_estimators': 3527,
            'num_iterations': 2940,
            'subsample': 1,
            'boosting_type': 'dart',
            'objective': 'regression',
            'metric': 'mape',
            'categorical_feature': [2, 3, 4, 6]   # weekdays, small_c, middle_c, big_c
}

gbm = LGBMRegressor(**params)


def run_hung_lgbm(train_x, train_y, val_x, val_y):
    print("run lgbm for hungarian inputs!")
    seed_everything(seed=127)
    estimator = gbm.fit(train_x, train_y,
                        eval_set=[(val_x, val_y)],
                        verbose=100,
                        eval_metric='mape',
                        early_stopping_rounds=100)
    lgbm_preds = gbm.predict(val_x, num_iteration=estimator.best_iteration_)
    lgbm_preds[lgbm_preds < 0] = 0

    # Plot LGBM: Predicted vs. True values
    plt.figure(figsize=(20, 5), dpi=80)
    x = range(0, len(lgbm_preds))
    plt.plot(x, val_y, label='true')
    plt.plot(x, lgbm_preds, label='predicted')
    plt.legend()
    plt.title('LGBM -' + 'opt')
    plt.show()

    # show scores
    print(f'MAPE of best iter is {get_mape(val_y,lgbm_preds)}')
    print(f'RMSE of best iter is {get_rmse(val_y,lgbm_preds)}')

    # save model
    data_type = 'all'
    model_name = MODELS_DIR+'lgbm_opt_mape_lr001_'+data_type+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))


if __name__ == "__main__":
    run_hung_lgbm(train_x, train_y, val_x, val_y)
