# target
TARGET = '취급액'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import datetime
# counter = pd.read_pickle("../data/20/counterfact_predicted.pkl")[TARGET]
# predicted = pd.read_csv("../submission/submission.csv")[TARGET]
# #
# plt.figure(figsize=(40, 5))
# plt.rcParams["axes.grid.axis"] = "y"
# plt.rcParams["axes.grid"] = True
# x = range(0, len(counter))
# plt.plot(x, counter, label='No-covid19 scenario', marker='', color='darkorange', linewidth=2, alpha = 0.7)
# plt.plot(x, predicted, label='predicted(actual)', marker='', color='grey', linewidth=2, alpha=0.8)
# pop_b = mpatches.Patch(color='darkorange', label='No-covid19 scenario')
# pop_c = mpatches.Patch(color='grey', label='predicted(actual)')
# plt.legend(handles=[pop_b, pop_c], fontsize=27, loc=2)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Time', fontsize=20)
# plt.ylabel('Sales', fontsize=20)
# plt.show()
#
## PROPHET
# from fbprophet import Prophet
# #
# train = pd.read_pickle("../data/20/train_v2.pkl")[['방송일시', '취급액','ymd','hours',
#                                                    'show_id','hours_inweek','week_num']]
# train['ymdh'] = pd.to_datetime(train['ymd'], format = "%Y-%m-%d") + \
#                 pd.to_timedelta(train['hours'], unit="h")
# daily_sales = train.groupby(['ymdh']).sum()['취급액']
# daily_sales = pd.DataFrame(daily_sales)
# daily_sales.index = pd.to_datetime(daily_sales.index, format="%Y-%m-%d %H")
# daily_sales = daily_sales.asfreq('H')
# daily_sales = daily_sales.fillna(method='bfill').fillna(method='ffill')
# hourly_sales = daily_sales.reset_index().rename(columns={'ymdh': 'ds', '취급액': 'y'}).copy()
#
# hr_sales_model = Prophet(interval_width=0.95)
# hr_sales_model.add_country_holidays(country_name='KR')
# hr_sales_model.fit(hourly_sales)
# hr_forecast = hr_sales_model.make_future_dataframe(freq='H',periods=4400)
# hr_forecast = hr_sales_model.predict(hr_forecast)

# plt.figure(figsize=(40, 10))
# plt.rcParams["axes.grid.axis"] = "y"
# plt.rcParams["axes.grid"] = True
# plt.plot(hourly_sales['y'][hr_forecast.ds < datetime.datetime(2019,7,1)],
#          marker='', color='darkorange', linewidth=2, alpha = 0.8)
# plt.plot(hr_forecast.yhat.loc[hr_forecast.ds < datetime.datetime(2019,7,1)],
#          marker='', color='grey', linewidth=2, alpha = 0.8)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Time', fontsize=20)
# plt.ylabel('Sales', fontsize=20)
# plt.savefig("../prophet_.png")

# HOURLY PLOT
# hours_inweek = pd.DataFrame({'hours_inweek' : sorted(train['hours_inweek'].unique()),
#     'sales': train.groupby(['hours_inweek']).취급액.sum()}).reset_index(drop = True)
# plt.figure(figsize=(40, 10))
# plt.rcParams["axes.grid.axis"] = "y"
# plt.rcParams["axes.grid"] = True
# plt.plot(hours_inweek,
#          marker='', color='grey', linewidth=2, alpha = 0.7)
# xposition = hours_inweek.loc[hours_inweek.hours_inweek.isin(list(range(0,168,24)))].index
# for xc in xposition:
#     plt.axvline(x=xc, color = "tomato", linewidth = 3, linestyle='dashed')
#
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Time', fontsize=25)
# plt.ylabel('Sales', fontsize=25)
# plt.show()
# plt.savefig("../hourly_plot.png")
#
# plt.figure(figsize=(40, 10))
# plt.rcParams["axes.grid.axis"] = "y"
# plt.rcParams["axes.grid"] = True
# # plt.plot(hourly_sales['y'][hr_forecast.ds < datetime.datetime(2019,7,1)],
# #          marker='', color='darkorange', linewidth=2, alpha = 0.8)
# plt.plot(hr_forecast.yhat.loc[hr_forecast.ds < datetime.datetime(2019,1,8)],
#          marker='', color='tomato', linewidth=2, alpha = 0.8)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Time', fontsize=25)
# plt.ylabel('Sales', fontsize=25)
# # plt.savefig("../prophet_1week.png")
# plt.show()