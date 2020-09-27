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
# Preprocessed datasets
# filter out columns existed in counterfactual data
df_wd_lag_PP_counter = df_wd_lag_PP.loc[:, df_wd_lag_PP.columns.isin(counter_fin_wd_lag.columns)]
df_wk_lag_PP_counter = df_wk_lag_PP.loc[:, df_wk_lag_PP.columns.isin(counter_fin_wk_lag.columns)]

counter_fin_wd_lag_PP = counter_fin_wd_lag.copy().loc[:,counter_fin_wd_lag.columns.isin(df_wd_test_PP.columns)]
counter_fin_wd_lag_PP[encoded_cols] = df_wd_test_PP[encoded_cols]
counter_fin_wd_lag_PP.drop(columns=['show_id', TARGET], inplace=True)
counter_fin_wk_lag_PP = counter_fin_wk_lag.copy().loc[:,counter_fin_wk_lag.columns.isin(df_wk_test_PP.columns)]
counter_fin_wk_lag_PP[encoded_cols] = df_wk_test_PP[encoded_cols]
counter_fin_wk_lag_PP.drop(columns=['show_id', TARGET], inplace=True)

# Divide data
# WD
train_wd_cnt_x, train_wd_cnt_y, val_wd_cnt_x, val_wd_cnt_y = divide_train_val(df_wd_lag_PP_counter, 8, drop=[])
top_wd_cnt, top_tr_wd_cnt_x, top_tr_wd_cnt_y, top_v_wd_cnt_x, top_v_wd_cnt_y = divide_top(df_wd_lag_PP_counter, 4004, 2013)
# WK
train_wk_cnt_x, train_wk_cnt_y, val_wk_cnt_x, val_wk_cnt_y = divide_train_val(df_wk_lag_PP_counter, 8, drop=[])
top_wk_cnt, top_tr_wk_cnt_x, top_tr_wk_cnt_y, top_v_wk_cnt_x, top_v_wk_cnt_y = divide_top(df_wk_lag_PP_counter, 2206, 999)


######################
######## Train #######
######################
params_all_wd_cnt = params_all_wd.copy()
params_all_wd_cnt['categorical_feature'] = [4, 10, 11, 12]
params_all_wk_cnt = params_all_wk.copy()
params_all_wk_cnt['categorical_feature'] = [4, 10, 11, 12]
params_top_wd_cnt = params_top_wd.copy()
params_top_wd_cnt['categorical_feature'] = [4, 10, 11, 12]
params_top_wk_cnt = params_top_wk.copy()
params_top_wk_cnt['categorical_feature'] = [4, 10, 11, 12]
# # base model
# model_wd_all, _ = run_lgbm(params_all_wd_cnt, train_wd_cnt_x, train_wd_cnt_y,
#                            val_wd_cnt_x, val_wd_cnt_y, 'wd_all_counter')
# model_wk_all, _ = run_lgbm(params_all_wk_cnt, train_wk_cnt_x, train_wk_cnt_y,
#                            val_wk_cnt_x, val_wk_cnt_y, 'wk_all_counter')
# # top model
# model_wd_top, _ = run_lgbm(params_top_wd_cnt, top_tr_wd_cnt_x, top_tr_wd_cnt_y,
#                            top_v_wd_cnt_x, top_v_wd_cnt_y, 'wd_top_counter')
# model_wk_top, _ = run_lgbm(params_top_wk_cnt, top_tr_wk_cnt_x, top_tr_wk_cnt_y,
#                            top_v_wk_cnt_x, top_v_wk_cnt_y, 'wk_top_counter')

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

counter_combined = pd.concat([counter_fin_wd_lag,counter_fin_wk_lag], axis=0)
counter_combined.sort_values(['방송일시'], inplace=True)
counter_combined = counter_combined.loc[:, base_cols]
counter_combined.to_excel(FEATURED_DATA_DIR + "counterfact_output.xlsx")
print("finish to run counterfactual data")





