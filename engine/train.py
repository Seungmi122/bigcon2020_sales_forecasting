# General imports
import warnings

warnings.filterwarnings("ignore")

# data
import pickle

# model
from lightgbm import LGBMRegressor

# visualize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from engine.utils import *
from engine.vars import *

###############################################################################
################################# Load Data ###################################
###############################################################################
# Import 2 types of dataset
# Descriptions:
#   - df_wd_lag : weekday / + lags
#   - df_wk_lag: weekend / + lags

df_wd_lag = load_df(FEATURED_DATA_DIR + 'train_fin_wd_lag.pkl')
df_wk_lag = load_df(FEATURED_DATA_DIR + 'train_fin_wk_lag.pkl')

df_wd_test = load_df(FEATURED_DATA_DIR + 'test_fin_wd_lag.pkl')
df_wk_test = load_df(FEATURED_DATA_DIR + 'test_fin_wk_lag.pkl')

# combined data for label encoding
tmp_combined = pd.concat([df_wd_lag, df_wk_lag, df_wd_test, df_wk_test]).drop(columns=['index'])
###############################################################################
############################### Preprocess  ##################################
###############################################################################

# Preprocessed datasets
tmp_combined = run_preprocess(tmp_combined)
df_wd_lag_PP = tmp_combined.loc[:, tmp_combined.columns.isin(df_wd_lag.columns)].iloc[:df_wd_lag.shape[0]].reset_index()
df_wk_lag_PP = tmp_combined.loc[:, tmp_combined.columns.isin(df_wk_lag.columns)]\
                .iloc[df_wd_lag.shape[0]:(df_wd_lag.shape[0]+df_wk_lag.shape[0])].reset_index()
df_wd_test_PP = tmp_combined.loc[:, tmp_combined.columns.isin(df_wd_test.columns)]\
                .iloc[(df_wd_lag.shape[0]+df_wk_lag.shape[0]):(df_wd_lag.shape[0]+df_wk_lag.shape[0]+df_wd_test.shape[0])]
df_wk_test_PP = tmp_combined.loc[:, tmp_combined.columns.isin(df_wk_test.columns)].iloc[-df_wk_test.shape[0]:]
df_test_PP = tmp_combined.loc[:, tmp_combined.columns.isin(df_wd_test.columns)]\
                .iloc[(df_wd_lag.shape[0]+df_wk_lag.shape[0]):]

# write pickle for test data
df_wd_test_PP.to_pickle(FEATURED_DATA_DIR + 'test_fin_wd_PP.pkl')
df_wk_test_PP.to_pickle(FEATURED_DATA_DIR + 'test_fin_wk_PP.pkl')
df_test_PP.to_pickle(FEATURED_DATA_DIR + 'test_fin_light_ver_PP.pkl')
# Divide data
# WD
train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y = divide_train_val(df_wd_lag_PP, 8, drop=[])
top_wd_lag, top_tr_wd_lag_x, top_tr_wd_lag_y, top_v_wd_lag_x, top_v_wd_lag_y = divide_top(df_wd_lag_PP, 4004, 2013)
# WK
train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y = divide_train_val(df_wk_lag_PP, 8, drop=[])
top_wk_lag, top_tr_wk_lag_x, top_tr_wk_lag_y, top_v_wk_lag_x, top_v_wk_lag_y = divide_top(df_wk_lag_PP, 2206, 999)

####################################################################
########################### Light GBM ##############################
####################################################################


def run_lgbm(params, train_x, train_y, val_x, val_y, df_type='wd_all'):
    """
    :objective: run lgbm model
    :param params: dictionary
    :param train_x: pd.DataFrame
    :param train_y: pd.DataFrame
    :param val_x: pd.DataFrame
    :param val_y: pd.DataFrame
    :param df_type: str - 'wd_all', 'wk_all', 'wd_top', 'wk_top'
    :return: LGBMRegressor, np.array
    """

    seed_everything(seed=127)

    model_lg = LGBMRegressor(**params)
    model_lg.fit(train_x, train_y)
    lgbm_preds = model_lg.predict(val_x)

    # Plot LGBM: Predicted vs. True values
    plt.figure(figsize= (40,5))
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["axes.grid"] = True
    x = range(0, len(lgbm_preds))
    plt.plot(x, val_y, label='true', marker='', color='grey', linewidth=2, alpha=0.8)
    plt.plot(x, lgbm_preds, label='predicted', marker='', color='tomato', linewidth=2)
    pop_b = mpatches.Patch(color='tomato', label='Predicted')
    pop_c = mpatches.Patch(color='grey', label='True')
    plt.legend(handles=[pop_b, pop_c], fontsize=27, loc=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel('Sales', fontsize=25)
    plt.show()

    # Get Scores
    print(f'MAPE of best iter is {get_mape(val_y, lgbm_preds)}')
    print(f'MAE of best iter is {get_mae(val_y, lgbm_preds)}')

    model_name = MODELS_DIR + 'lgbm_finalmodel_' + df_type + '.bin'
    pickle.dump(model_lg, open(model_name, 'wb'))

    return model_lg, lgbm_preds

##################################################################
################# Step 1. For ALL observations ###################
##################################################################


# parameters for wd/wk model
params_all_wd = {'feature_fraction': 1,
                 'learning_rate': 0.001,
                 'min_data_in_leaf': 135,
                 'n_estimators': 3527,
                 'num_iterations': 2940,
                 'subsample': 1,
                 'boosting_type': 'dart',
                 'objective': 'regression',
                 'metric': 'mape',
                 'categorical_feature': [3, 9, 10, 11]  ## weekdays, small_c, middle_c, big_c
                 }

params_all_wk = {'feature_fraction': 1,
                 'learning_rate': 0.001,
                 'min_data_in_leaf': 134,
                 'n_estimators': 3474,
                 'num_iterations': 2928,
                 'subsample': 1,
                 'boosting_type': 'dart',
                 'objective': 'regression',
                 'metric': 'mape',
                 'categorical_feature': [3, 9, 10, 11]}  ## weekdays, small_c, middle_c, big_c


###########################################################################
################### Step 2. For High-rank observations ###################
###########################################################################

params_top_wd = {'feature_fraction': 1,
                 'learning_rate': 0.0025,
                 'min_data_in_leaf': 70,
                 'n_estimators': 5000,
                 'num_iterations': 4000,
                 'subsample': 1,
                 'boosting_type': 'dart',
                 'objective': 'regression',
                 'metric': 'mape',
                 'categorical_feature': [3, 9, 10, 11]  ## weekdays, small_c, middle_c, big_c
                 }

params_top_wk = {'feature_fraction': 1,
                 'learning_rate': 0.0025,
                 'min_data_in_leaf': 30,
                 'n_estimators': 5000,
                 'num_iterations': 3500,
                 'subsample': 1,
                 'boosting_type': 'dart',
                 'objective': 'regression',
                 'metric': 'mape',
                 'categorical_feature': [3, 9, 10, 11]  ## weekdays, small_c, middle_c, big_c
                 }

#####################################################################
############## Step 3. Mix results from step1 & step2 ###############
#####################################################################


def mixed_df(model_top, top_df, val_all_df_x, preds_all, num_top):
    """
    :objective: combine results from step 1&2
    :param model_top: LGBMRegressor - step 2 model
    :param top_df: pandas Dataframe - divided df with high mean_sales_origin(by divide_top_function)
    :param val_all_df_x: pandas Dataframe - validation x
    :param preds_all: predicted y from step 1
    :param num_top: int - index to be splitted
    :return: pandas DataFrame
    """
    top_idx = set(top_df.iloc[:num_top, :].index)
    val_idx = set(val_all_df_x.index)
    top_in_val = list(val_idx.intersection(top_idx))

    val_copy = val_all_df_x.copy()
    val_copy[TARGET] = preds_all

    for i in top_in_val:
        val_copy[TARGET].loc[val_copy.index == i] = model_top.predict(val_all_df_x.loc[val_all_df_x.index == i])

    return val_copy


def mix_results(true_y, pred_y):
    """
    :objective: draw plot of true and estimated value
    :param true_y: pandas Series
    :param pred_y: pandas Series
    :return: plot figure
    """
    # Plot TOP: Predicted vs. True values
    plt.figure(figsize=(40, 5))
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["axes.grid"] = True
    x = range(0, len(true_y))
    plt.plot(x, true_y, label='true', marker='', color='grey', linewidth=2, alpha=0.8)
    plt.plot(x, pred_y, label='predicted', marker='', color='tomato', linewidth=2)
    pop_b = mpatches.Patch(color='tomato', label='Predicted')
    pop_c = mpatches.Patch(color='grey', label='True')
    plt.legend(handles=[pop_b, pop_c], fontsize=27, loc=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Sales', fontsize=20)
    plt.show()

    print(f'MAPE of mixed model is {get_mape(true_y, pred_y)}')
    print(f'MAE of mixed model is {get_mae(true_y, pred_y)}')
    print(f'RMSE of mixed model is {get_rmse(true_y, pred_y)}')


def run_models():
    # base model
    model_wd_all, preds_wd_all = run_lgbm(params_all_wd, train_wd_lag_x, train_wd_lag_y,
                                          val_wd_lag_x, val_wd_lag_y, 'wd_all')
    model_wk_all, preds_wk_all = run_lgbm(params_all_wk, train_wk_lag_x, train_wk_lag_y,
                                          val_wk_lag_x, val_wk_lag_y, 'wk_all')
    # top model
    model_wd_top, preds_wd_top = run_lgbm(params_top_wd, top_tr_wd_lag_x, top_tr_wd_lag_y,
                                          top_v_wd_lag_x, top_v_wd_lag_y, 'wd_top')
    model_wk_top, preds_wk_top = run_lgbm(params_top_wk, top_tr_wk_lag_x, top_tr_wk_lag_y,
                                          top_v_wk_lag_x, top_v_wk_lag_y, 'wk_top')
    # mixed
    mixed_wd = mixed_df(model_wd_top, top_wd_lag, val_wd_lag_x, preds_wd_all, num_top=6017)
    mix_results(val_wd_lag_y, mixed_wd[TARGET])
    mixed_wk = mixed_df(model_wk_top, top_wk_lag, val_wk_lag_x, preds_wk_all, num_top=3205)
    mix_results(val_wk_lag_y, mixed_wk[TARGET])


#####################################################################
######################## Cross Validation ###########################
#####################################################################

def cross_validation(cv_months=[7, 8, 9]):

    # for step 1 model
    # wd
    for num in cv_months:
        month = num
        train_x_wd, train_y_wd, val_x_wd, val_y_wd = divide_train_val(df_wd_lag_PP, month, drop=[])
        print(f'WD - CV with month {month} is starting.')
        run_lgbm(params_all_wd, train_x_wd, train_y_wd, val_x_wd, val_y_wd, 'wd_all')

    # wk
    for num in cv_months:
        month = num
        train_x_wk, train_y_wk, val_x_wk, val_y_wk = divide_train_val(df_wk_lag_PP, month, drop=[])
        print(f'WK - CV with month {month} is starting.')
        run_lgbm(params_all_wk, train_x_wk, train_y_wk, val_x_wk, val_y_wk, 'wk_all')

    # for step 3 model
    # wd
    cv_wd = [[2952, 1052, 12], [4524, 2093, 40]]
    for num in cv_months:
        print(f'WD - CV for Mixed model - month {num} is starting.')
        for lst in cv_wd:
            print(f'WD - CV for Mixed model - top {lst[2]}% is starting.')
            train = lst[0]
            val = lst[1]
            train_x_wd_all, train_y_wd_all, val_x_wd_all, val_y_wd_all = divide_train_val(df_wd_lag_PP, num, drop=[])
            top_cv, train_x_wd_top, train_y_wd_top, val_x_wd_top, val_y_wd_top = divide_top(df_wd_lag_PP, train, val)
            model_all_cv, preds_all_cv = run_lgbm(params_all_wd, train_x_wd_all, train_y_wd_all, val_x_wd_all, val_y_wd_all,
                                                  'wd_all')
            model_top_cv, preds_top_cv = run_lgbm(params_top_wd, train_x_wd_top, train_y_wd_top, val_x_wd_top, val_y_wd_top,
                                                  'wd_top')
            mixed_wd = mixed_df(model_top_cv, top_cv, val_x_wd_all, preds_all_cv, num_top=(lst[0] + lst[1]))
            mix_results(val_y_wd_all, mixed_wd[TARGET])

    # wk
    cv_wk = [[1205, 504, 16], [2856, 1359, 40]]
    for num in cv_months:
        print(f'WK - CV for Mixed model - month {num} is starting.')
        for lst in cv_wk:
            print(f'WK - CV for Mixed model - top {lst[2]}% is starting.')
            train = lst[0]
            val = lst[1]
            train_x_wk_all, train_y_wk_all, val_x_wk_all, val_y_wk_all = divide_train_val(df_wk_lag_PP, num, drop=[])
            top_cv, train_x_wk_top, train_y_wk_top, val_x_wk_top, val_y_wk_top = divide_top(df_wk_lag_PP, train, val)
            _, preds_all_cv = run_lgbm(params_all_wk, train_x_wk_all, train_y_wk_all, val_x_wk_all, val_y_wk_all, 'wk_all')
            model_top_cv, _ = run_lgbm(params_top_wk, train_x_wk_top, train_y_wk_top, val_x_wk_top, val_y_wk_top, 'wk_top')
            mixed_wk = mixed_df(model_top_cv, top_cv, val_x_wk_all, preds_all_cv, num_top=(lst[0] + lst[1]))
            mix_results(val_y_wk_all, mixed_wk[TARGET])

# #####################################################################
# #################### Robust Cross Validation ########################
# #####################################################################
# """
# perform cross validation on 2019-dec data by 2019 Jan to Aug data
# to guarantee the time series robustness of our model
# """

def robust_cross_validation():

    # Data preparation, Jan to Aug
    train_x_wd_rb, train_y_wd_rb, val_x_wd_rb, val_y_wd_rb = divide_train_val(df_wd_lag_PP, 8, drop=[])
    top_wd_rb, top_tr_x_wd_rb, top_tr_y_wd_rb, top_v_x_wd_rb, top_v_y_wd_rb = divide_top(df_wd_lag_PP, 4004, 2013)
    train_x_wk_rb, train_y_wk_rb, val_x_wk_rb, val_y_wk_rb = divide_train_val(df_wk_lag_PP, 8, drop=[])
    top_wk_rb, top_tr_x_wk_rb, top_tr_y_wk_rb, top_v_x_wk_rb, top_v_y_wk_rb = divide_top(df_wk_lag_PP, 2206, 999)
    # target - 2019 Dec
    DEC = 12
    wd_dec_x = df_wd_lag_PP[df_wd_lag_PP.months == DEC].drop(['index', 'show_id', TARGET], axis=1)
    wd_dec_y = df_wd_lag_PP[df_wd_lag_PP.months == DEC][TARGET]
    wk_dec_x = df_wk_lag_PP[df_wk_lag_PP.months == DEC].drop(['index', 'show_id', TARGET], axis=1)
    wk_dec_y = df_wk_lag_PP[df_wk_lag_PP.months == DEC][TARGET]

    # wd
    model_all_rb, preds_all_rb = run_lgbm(params_all_wd, train_x_wd_rb, train_y_wd_rb,
                                          val_x_wd_rb, val_y_wd_rb, 'wd_all')
    model_top_rb, _ = run_lgbm(params_top_wd, top_tr_x_wd_rb, top_tr_y_wd_rb, top_v_x_wd_rb, top_v_y_wd_rb,'wd_top')
    preds_wd_dec = model_all_rb.predict(wd_dec_x)
    mixed_wd_rb = mixed_df(model_top_rb, top_wd_rb, wd_dec_x, preds_wd_dec, num_top=6017)
    mix_results(wd_dec_y, mixed_wd_rb[TARGET])

    # wk
    model_all_rb, _ = run_lgbm(params_all_wk, train_x_wk_rb, train_y_wk_rb, val_x_wk_rb, val_y_wk_rb, 'wk_all')
    model_top_rb, _ = run_lgbm(params_top_wk, top_tr_x_wk_rb, top_tr_y_wk_rb, top_v_x_wk_rb, top_v_y_wk_rb,'wk_top')
    preds_wk_dec = model_all_rb.predict(wk_dec_x)
    mixed_wk_rb = mixed_df(model_top_rb, top_wk_rb, wk_dec_x, preds_wk_dec, num_top=3205)
    mix_results(wk_dec_y, mixed_wk_rb[TARGET])


# #####################################################################
# ######################## Feature Importance #########################
# #####################################################################


def feature_importance(model_df):
    """

    :param model_df:
    :return:
    """
    feature_imp = pd.DataFrame(sorted(zip(model_df.feature_importances_, train_wd_lag_x.columns)),
                               columns=['Value', 'Feature'])
    plt.figure(figsize=(20, 10))

    # Full length
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:30])
    plt.tight_layout()
    plt.show()


#############################################################################
########################### Hyperparameter Tuning ###########################
#############################################################################


# robust cv : 1~8로 12월 예측하기

'''
class BO:
    def __init__(self, df, tr_x, tr_y, v_x, v_y):
        self.data = df

    def neg_mape(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        result = (-1)*mape
        return result

    def cv_score(self, learning_rate, num_iterations, n_estimators, min_data_in_leaf, feature_fraction, subsample):
        hp_d = {}
        hp_d['learning_rate'] = learning_rate
        hp_d['n_estimators'] = int(n_estimators)
        hp_d['num_iterations'] = int(num_iterations)
        hp_d['subsample'] = subsample
        #hp_d['num_leaves'] = int(num_leaves)
        hp_d['feature_fraction'] = feature_fraction
        hp_d['min_data_in_leaf'] = int(min_data_in_leaf)
        #hp_d['scale_pos_weight'] = scale_pos_weight

        data_cv = self.data

        tscv = TimeSeriesSplit(n_splits = 3)
        score = []
        for train_index, val_index in tscv.split(data_cv):
            cv_train, cv_val = data_cv.iloc[train_index], data_cv.iloc[val_index]
            cv_train_x = cv_train.drop(TARGET, axis=1)
            cv_train_y = cv_train.TARGET
            cv_val_x = cv_val.drop(TARGET, axis=1)
            cv_val_y = cv_val.TARGET

            model = lgb.LGBMRegressor(boosting_type= 'dart', objective= 'regression', metric= ['mape'], seed=42,
                                  subsample_freq=1, max_depth=-1,
                                  categorical_feature= [3,9,10,11],
                                      **hp_d)

            model.fit(cv_train_x, cv_train_y)
            pred_values = model.predict(cv_val_x)
            true_values = cv_val_y

            score.append(self.neg_mape(true_values, pred_values))

        result  = np.mean(score)
        return result

    def BO_execute(self):

        int_list = ['n_estimators', 'num_iterations', 'min_data_in_leaf'
                   #,'num_leaves'
                   ]

        bayes_optimizer = BayesianOptimization(f=self.cv_score, pbounds={'learning_rate':(0.0001,0.01),
                                                          'subsample':(0.5,1),
                                                          'n_estimators':(3000,6000),
                                                          'num_iterations':(2000,5000),
                                                          'feature_fraction':(0.6,1),
                                                          'min_data_in_leaf':(100,300),
                                                           #'scale_pos_weight': (1,2)
                                                                        },
                                      random_state=42, verbose=2)

        bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)

        ## print all iterations
        #for i,res in enumerate(bayes_optimizer.res):
        #    print('Iteration {}: \n\t{}'.format(i, res))
        ## print best iterations
        #print('Final result: ', bayes_optimizer.max)


        BO_best = bayes_optimizer.max['params']
        for i in int_list:
            BO_best[i] = int(np.round(BO_best[i]))
        BO_best['boosting_type'] = 'dart'
        BO_best['objective'] = 'regression'
        BO_best['metric'] = 'mape'
        BO_best['categorical_feature'] = [3,9,10,11]
        #BO_best['feature_fraction'] = 1
        #BO_best['subsample'] = 1

        train_x = self.tr_x
        train_y = self.tr_y
        val_x = self.v_x
        val_y = self.v_y
        data_train = lgb.Dataset(train_x, label=train_y)
        data_valid = lgb.Dataset(val_x, label=val_y)

        BO_result = lgb.train(BO_best, data_train,
                            valid_sets = [data_train, data_valid],
                            verbose_eval = 500,
                             )
        return BO_best, BO_result

# params_wd, model_wd = BO(df_wd_lag_PP, train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y).BO_execute()
# params_wk, model_wk = BO(df_wk_lag_PP, train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y).BO_execute()
# '''

if __name__ == "__main__":
    run_models()

