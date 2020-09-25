# General imports
import warnings
warnings.filterwarnings("ignore")

import datetime
import sys

from opt.inputs import *

# optimization
from munkres import Munkres


####################################################################
########################### Predict ################################
####################################################################

lgbm_model_path = MODELS_DIR + 'lgbm_opt_mape_lr001_all.bin'
estimator = pickle.load(open(lgbm_model_path, 'rb'))
lgbm_preds_opt1 = estimator.predict(hung1_PP.drop(columns=['small_c', 'show_id', '상품코드']))
lgbm_preds_opt2 = estimator.predict(hung2_PP.drop(columns=['small_c', 'show_id', '상품코드']))

####################################################################
########################### Hung1 ##################################
####################################################################

hung_mat = lgbm_preds_opt1.reshape((125, 125))  #rows: items, cols: time
matrix = hung_mat
cost_matrix = []
for row in matrix:
    cost_row = []
    for col in row:
        cost_row += [sys.maxsize - col]
    cost_matrix += [cost_row]

m = Munkres()
indexes = m.compute(cost_matrix)
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value

print(f'total profit={total}')

full_items_list = pd.read_excel("../data/20/tmp_hung_firstweek_items.xlsx")
full_items_list.drop(columns=['Unnamed: 0', 'small_c_code', 'middle_c_code','big_c_code'], inplace=True)
hung_out = full_items_list.copy()
hung_out['방송일시'] = np.nan
for i in indexes:
    product = i[0]
    time = i[1]
    hung_out['방송일시'][product] = hung1_times[time]
hung_out.drop(columns=['노출(분)'], inplace=True)
hung_out.sort_values(['방송일시'], ascending=True, inplace=True)
hung_out.set_index('방송일시', inplace=True)
hung_out.to_excel("../data/20/opt_light_output.xlsx")
print("hungarian with 1 hierarchy completed")

####################################################################
########################### Hung2 ##################################
####################################################################
hung2_PP['pred'] = lgbm_preds_opt2
hung2_PP['방송일시'] = hung2.방송일시
hung2_PP['ymd'] = [d.date() for d in hung2_PP["방송일시"]]

prod_list = pd.read_excel('../data/01/2020sales_opt_temp_list.xlsx')

h1_output = []
h2_output = []
cat = prod_list.상품군.unique()
h2_total = 0

for i in pd.date_range(start=datetime.datetime(2020, 6, 2, 1), end=datetime.datetime(2020, 7, 1, 1)):

    if i == datetime.datetime(2020, 6, 2, 1):
        jun_day = hung2_PP.loc[hung2_PP.방송일시 <= i, :]
    else:
        i2 = i - datetime.timedelta(days=1)
        jun_day = hung2_PP.loc[(hung2_PP.방송일시 > i2) & (hung2_PP.방송일시 <= i), :]

    ind = np.random.multinomial(n=22,
                                pvals=[0.025, 0.115, 0.009, 0.217, 0.12, 0.121, 0.093, 0.107, 0.069, 0.113, 0.009])
    rmv_cat = np.where(np.isin(ind, 0))[0].tolist()
    jun_day_selected = jun_day.loc[~jun_day.상품군.isin(rmv_cat), :]
    jun_day_grp = jun_day_selected.groupby(['방송일시', '상품군']).pred.mean().to_frame()
    jun_day_grp.reset_index(inplace=True)
    jun_day_pvt = jun_day_grp.pivot(index='방송일시', columns='상품군')
    jun_day_hung = jun_day_pvt[np.repeat(jun_day_pvt.columns.values, ind[~np.isin(ind, 0)].tolist())]

    matrix = np.array(jun_day_hung)
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()

    indexes = m.compute(cost_matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value

    jun_day_hung.reset_index(inplace=True)
    cl = np.repeat(jun_day_pvt.columns.values, ind[~np.isin(ind, 0)].tolist())
    jun_day_fin = jun_day_hung.iloc[:, 0].to_frame()
    jun_day_fin['상품군'] = 0
    for id in indexes:
        jun_day_fin.iloc[id[0], 1] = cat[cl[id[1]][1]]

    h1_output.append(jun_day_fin)

    cat_temp = np.where(~np.isin(ind, 0))[0].tolist()
    if i < datetime.datetime(2020, 6, 29, 1):
        jun_week = hung2_PP.loc[(hung2_PP.방송일시.dt.week > i.isocalendar()[1] - 1) &
                               (hung2_PP.방송일시.dt.week <= i.isocalendar()[1]), :]
    else:
        jun_week = hung2_PP.loc[(hung2_PP.방송일시.dt.week >= i.isocalendar()[1] - 1), :]

    for j in cat_temp:

        runprod = jun_week.loc[jun_week.상품군 == j, :].상품코드.unique().tolist()
        runtime = jun_day_fin.loc[jun_day_fin.상품군 == cat[j], '방송일시']

        run_schd = jun_day.loc[jun_day.상품코드.isin(runprod) & jun_day.방송일시.isin(runtime),].groupby(
            ['상품코드', '방송일시']).pred.mean().to_frame()
        run_schd.reset_index(inplace=True)
        day_cat_hung = run_schd.pivot(index='방송일시', columns='상품코드')

        matrix = np.array(day_cat_hung)
        cost_matrix = []
        for row in matrix:
            cost_row = []
            for col in row:
                cost_row += [sys.maxsize - col]
            cost_matrix += [cost_row]

        m = Munkres()
        indexes = m.compute(cost_matrix)
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
            h2_total += value

        code = []
        ct = []
        for id2 in indexes:
            code.append((run_schd.상품코드.unique().tolist()[id2[1]]))

        prod_fin = pd.DataFrame({'방송일시': run_schd.방송일시.unique(),
                                 '상품코드': code})

        h2_output.append(prod_fin)
    h2_output_fin = pd.concat(h2_output)

print(f'total profit={h2_total}')
h1_output = pd.concat(h1_output)
h2_output_fin = h2_output_fin.sort_values('방송일시')
h2_output_name = pd.merge(h2_output_fin, hung_list, left_on='상품코드', right_on='row_num', how='left')
h2_output_name = h2_output_name.drop(['상품코드_x'], axis=1)
h2_output_name = h2_output_name.rename(columns={'상품코드_y': '상품코드'})
h2_output_name = h2_output_name.astype({'상품코드': 'int64'})
h2_output_final = pd.merge(h2_output_name, prod_list, on=['상품코드', '상품명'], how='left').drop('row_num', 1)

h1_output['n'] = 1
h1_output.pivot_table(index='방송일시', columns='상품군', fill_value=0).to_excel('../data/20/h1_output.xlsx')
h2_output_final.to_excel('../data/20/h2_output.xlsx')

# tmp = pd.read_pickle("../data/20/test_v2.pkl")
# tmp2 = tmp.loc[tmp.months==6,].groupby(['days','상품군']).show_id.nunique().to_frame()
# tmp2.reset_index(inplace=True)
# tmp3 = tmp2.pivot_table(index='days', columns='상품군',aggfunc=sum, margins=True,
#                         dropna=True, fill_value=0)
# tmp4 = tmp3.div( tmp3.iloc[:,-1], axis=0)
# tmp4.mean(axis=0)

print("hungarian with 2 hierarchy completed")