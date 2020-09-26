# General imports
import warnings
warnings.filterwarnings("ignore")

from engine.features import Features
from engine.train import *


######################
##### Load Data ######
######################
t = Features(types='test', counterfactual=True, not_divided=False)
counterfact = t.run_all().reset_index()
counter_fin_wd_lag = counterfact.loc[counterfact.weekends == 0]
counter_fin_wd_lag.drop(columns=['lag_sales_wk_1', 'lag_sales_wk_2'], inplace=True)
counter_fin_wk_lag = counterfact.loc[counterfact.weekends == 1]
counter_fin_wk_lag.drop(columns=['lag_sales_wd_1', 'lag_sales_wd_2',
                                 'lag_sales_wd_3', 'lag_sales_wd_4', 'lag_sales_wd_5'], inplace=True)
counter_fin_wd_lag_PP = run_preprocess(counter_fin_wd_lag).drop(columns=['index', 'show_id', TARGET])
counter_fin_wk_lag_PP = run_preprocess(counter_fin_wk_lag).drop(columns=['index', 'show_id', TARGET])

# Preprocessed datasets
# filter out columns existed in counterfactual data
df_wk_lag_PP = run_preprocess(df_wk_lag).loc[:, df_wk_lag_PP.columns.isin(counter_fin_wk_lag.columns)]
df_wd_lag_PP = run_preprocess(df_wd_lag).loc[:, df_wd_lag_PP.columns.isin(counter_fin_wd_lag.columns)]

# Divide data
# WD
train_wd_lag_x, train_wd_lag_y, val_wd_lag_x, val_wd_lag_y = divide_train_val(df_wd_lag_PP, 8, drop=[])
top_wd_lag, top_tr_wd_lag_x, top_tr_wd_lag_y, top_v_wd_lag_x, top_v_wd_lag_y = divide_top(df_wd_lag_PP, 4004, 2013)
# WK
train_wk_lag_x, train_wk_lag_y, val_wk_lag_x, val_wk_lag_y = divide_train_val(df_wk_lag_PP, 8, drop=[])
top_wk_lag, top_tr_wk_lag_x, top_tr_wk_lag_y, top_v_wk_lag_x, top_v_wk_lag_y = divide_top(df_wk_lag_PP, 2206, 999)


######################
######## Train #######
######################
# base model
model_wd_all, preds_wd_all = run_lgbm(params_all_wd, train_wd_lag_x, train_wd_lag_y,
                                      val_wd_lag_x, val_wd_lag_y, 'wd_all_counter')
model_wk_all, preds_wk_all = run_lgbm(params_all_wk, train_wk_lag_x, train_wk_lag_y,
                                      val_wk_lag_x, val_wk_lag_y, 'wk_all_counter')
# top model
model_wd_top, preds_wd_top = run_lgbm(params_top_wd, top_tr_wd_lag_x, top_tr_wd_lag_y,
                                      top_v_wd_lag_x, top_v_wd_lag_y, 'wd_top_counter')
model_wk_top, preds_wk_top = run_lgbm(params_top_wk, top_tr_wk_lag_x, top_tr_wk_lag_y,
                                      top_v_wk_lag_x, top_v_wk_lag_y, 'wk_top_counter')

######################
###### Predict #######
######################
# Load Models
model_path = MODELS_DIR + 'lgbm_finalmodel_wd_all_counter.bin'
model_wd_all = pickle.load(open(model_path, 'rb'))

model_path = MODELS_DIR + 'lgbm_finalmodel_wd_top_counter.bin'
model_wd_top = pickle.load(open(model_path, 'rb'))

model_path = MODELS_DIR + 'lgbm_finalmodel_wk_all_counter.bin'
model_wk_all = pickle.load(open(model_path, 'rb'))

model_path = MODELS_DIR + 'lgbm_finalmodel_wk_top_counter.bin'
model_wk_top = pickle.load(open(model_path, 'rb'))

counter_wd_sort = counter_fin_wd_lag_PP.sort_values('mean_sales_origin', ascending=False)
# Predict all observations
pred_counter_wd_all = model_wd_all.predict(counter_fin_wd_lag_PP)
# Mixed DF (Top: 727개)
counter_mixed_wd = mixed_df(model_wd_top, counter_wd_sort, counter_fin_wd_lag_PP, pred_counter_wd_all, num_top=727)
counter_fin_wd_lag[TARGET] = counter_mixed_wd[TARGET]

counter_wk_sort = counter_fin_wk_lag_PP.sort_values('mean_sales_origin', ascending=False)
# Predict all observations
pred_counter_wk_all = model_wk_all.predict(counter_fin_wk_lag_PP)
# Mixed DF (Top: 249개)
counter_mixed_wk = mixed_df(model_wk_top, counter_wk_sort, counter_fin_wk_lag_PP, pred_counter_wk_all, num_top=249)
counter_fin_wk_lag[TARGET] = counter_mixed_wk[TARGET]

counter_combined = pd.concat([counter_fin_wd_lag,counter_fin_wk_lag], axis = 0)
counter_combined.sort_values(['방송일시'], inplace=True)
counter_combined.to_pickle("../data/20/counterfact_predicted.pkl")





