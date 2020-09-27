import warnings

warnings.filterwarnings("ignore")

# stat
import pandas as pd
import numpy as np

# data
import datetime
import itertools
import json

from engine.vars import *


class Features:
    def __init__(self, types, counterfactual=False, not_divided=False):
        """
        :objective: load data and engineer features
        :param test: boolean  - whether to get test data. if false, we can work on raw train dataset(2019 sales)
        """
        self.type = types
        self.counterfactual = counterfactual
        self.not_divided = not_divided
        
        if self.type == 'hungarian':
            df = pd.read_excel(OPT_DATA_DIR + "hhierarchy.xlsx")
        elif self.type == "hungarian_h1":
            df = pd.read_pickle(OPT_DATA_DIR + "hdirect.pkl")
        elif self.type == 'test':
            df = pd.read_excel(RAW_DATA_DIR + "202006schedule.xlsx", skiprows=1)
        else:
            df = pd.read_excel(RAW_DATA_DIR + "2019sales.xlsx", skiprows=1)
            # adjust data types
            df.rename(columns={' 취급액 ': '취급액'}, inplace=True)
            df['volume'] = df['취급액'] / df['판매단가']

        # define data types
        df['exposed'] = df['노출(분)']
        df.exposed = df.exposed.astype(float)
        df.마더코드 = df.마더코드.astype(int).astype(str).str.zfill(6)
        df.상품코드 = df.상품코드.astype(int).astype(str).str.zfill(6)
        df.방송일시 = pd.to_datetime(df.방송일시, format="%Y/%m/%d %H:%M")
        if (self.type == "hungarian")|(self.type == "hungarian_h1"):
            df.sort_values(['상품코드', '방송일시'], ascending=[True, True], inplace=True)
        else:
            df.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)

        df['ymd'] = [d.date() for d in df["방송일시"]]
        self.train = df
        # define ts_schedule, one row for each timeslot
        # then 방송일시 works as index in ts_schedule
        self.ts_schedule = df.copy().groupby('방송일시').first()
        self.ts_schedule.reset_index(inplace=True)
        
    def get_df(self):
        return self.train

    ##################################
    ## onair time/order info variables
    ##################################

    def get_time(self):
        """
        :** objective: get year, month, day, hours
        """
        self.train['years'] = self.train.방송일시.dt.year
        self.train['months'] = self.train.방송일시.dt.month
        self.train['days'] = self.train.방송일시.dt.day
        self.train['hours'] = self.train.방송일시.dt.hour
        self.train['week_num'] = self.train.방송일시.dt.week

    def get_weekday(self):
        """
        :** objective: get weekday
        """
        self.train['weekdays'] = self.train.방송일시.dt.day_name()

    def get_hours_inweek(self):
        """
        :** objective: get hours by week (1~168)
        """
        hours_inweek = []
        for i in range(0, len(self.train)):
            hr = self.train['hours'].iloc[i]
            dy = self.train['weekdays'].iloc[i]
            if dy == 'Tuesday':
                hours_inweek.append(hr + 24)
            elif dy == 'Wednesday':
                hours_inweek.append(hr + 24 * 2)
            elif dy == 'Thursday':
                hours_inweek.append(hr + 24 * 3)
            elif dy == 'Friday':
                hours_inweek.append(hr + 24 * 4)
            elif dy == 'Saturday':
                hours_inweek.append(hr + 24 * 5)
            elif dy == 'Sunday':
                hours_inweek.append(hr + 24 * 6)
            else:
                hours_inweek.append(hr)
        self.train['hours_inweek'] = hours_inweek

    def get_holidays(self):
        """
        :** objective: create a dummy variable for holidays (weekends + red)
        """
        holidays = []
        if self.type != 'train':
            holiday_dates = ['2020-06-06']
        else:
            holiday_dates = ['2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06',
                             '2019-03-01', '2019-05-06', '2019-06-06', '2019-08-15',
                             '2019-09-12', '2019-09-13', '2019-10-03', '2019-10-09',
                             '2019-12-25']
        for i in range(0, len(self.train)):
            dt = str(self.train['ymd'].iloc[i])
            dy = self.train['weekdays'].iloc[i]
            if dt in holiday_dates or dy == 'Saturday' or dy == 'Sunday':
                holidays.append(1)
            else:
                holidays.append(0)
        self.train['holidays'] = holidays

    def get_red_days(self):
        """
        :** objective: create a dummy variable for just red
        """
        red = []
        if self.type != "train":
            holiday_dates = ['2020-06-06']
        else:
            holiday_dates = ['2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06',
                             '2019-03-01', '2019-05-06', '2019-06-06', '2019-08-15',
                             '2019-09-12', '2019-09-13', '2019-10-03', '2019-10-09',
                             '2019-12-25']
        for i in range(0, len(self.train)):
            dt = str(self.train['ymd'].iloc[i])
            if dt in holiday_dates:
                red.append(1)
            else:
                red.append(0)
        self.train['red'] = red

    def get_weekends(self):
        """
        :** objective: create a dummy variable for just weekends
        """
        self.train['weekends'] = 0
        self.train.loc[(self.train['red'] == 0) & (self.train['holidays'] == 1), 'weekends'] = 1

    def get_min_start(self):
        """
        :** objective: get startig time (min)
        """
        temp = self.train.방송일시.dt.minute
        min_start = []
        for i in range(0, len(self.train)):
            time = temp[i]
            if (time < 20) & (time >= 0):
                rtn = 0
            elif (time < 40) & (time >= 20):
                rtn = 20
            else:
                rtn = 40
            min_start.append(rtn)
        self.train['min_start'] = min_start

    def filter_jappingt(self):
        """
        :objective: round up 방송일시
        """
        japp = []
        for i in range(0, len(self.train)):
            time = self.train['방송일시'].iloc[i]
            if (time.minute < 30) & (time.hour == 0):
                rtn = time.hour
            elif time.minute >= 30:
                if time.hour == 23:
                    rtn = 0
                else:
                    rtn = time.hour + 1
            else:
                if time.hour == 0:
                    rtn = 23
                else:
                    rtn = time.hour
            japp.append(rtn)
        self.train['japp'] = japp

    def fill_exposed_na(self):
        """
        :objective: fill out NA values on 'exposed' with mean(exposed)
        :return:pd.Dataframe with adjusted 'exposed' column
        """
        self.train["exposed"].fillna(self.train.groupby('방송일시')['exposed'].transform('mean'), inplace=True)

    def round_exposed(self):
        """
        :objective: round exposed variable to  avoid imbalance
        """
        self.train['exposed_t'] = self.train.exposed
        for i in self.train.exposed.unique():
            if i < 25:
                rtn = 20
            else:
                rtn = 30
            self.train.exposed_t.loc[self.train.exposed == i] = rtn
        self.train.exposed_t = self.train.exposed_t.astype('category')

    def get_ymd(self):
        """
        :objective: add 'ymd' variable to dataset
        """
        t = 1
        while t < 9:
            for i in self.ts_schedule.ymd.unique():
                if (i == datetime.date(2019, 1, 1)) | (i == datetime.date(2020, 6, 1)): continue
                time_idx = self.ts_schedule[self.ts_schedule.ymd == i].index[0]
                first_show = self.ts_schedule.iloc[time_idx]
                last_show = self.ts_schedule.iloc[time_idx - 1]
                if (first_show['마더코드'] == last_show['마더코드']) & (
                        first_show['방송일시'] <= last_show['방송일시'] + datetime.timedelta(minutes=last_show['exposed'])):
                    self.ts_schedule.ymd.iloc[time_idx] = self.ts_schedule.ymd.iloc[time_idx - 1]
            t = t + 1

    def timeslot(self):
        """
        :objective: get timeslot of each show
        """
        if self.type in ["hungarian", "hungarian_h1"]:
            self.train['parttime'] = 1 # only 1 for hungarian input

        else:
            show_counts = [len(list(y)) for x, y in itertools.groupby(self.ts_schedule.상품코드)]  # count repeated 상품코드
            self.ts_schedule['parttime'] = ""  # define empty column
            j = 0
            for i in range(0, len(show_counts)):
                first_idx = j
                self.ts_schedule.parttime[first_idx] = 1
                j += show_counts[i]
                if show_counts[i] == 1:
                    pass
                self.ts_schedule.parttime[(first_idx + 1):j] = np.arange(2, show_counts[i] + 1)

            self.train['parttime'] = ""  # define empty column
            # add timeslot variable to train dataset
            for i in range(0, len(self.ts_schedule)):
                self.train.parttime[self.train.방송일시 == self.ts_schedule.방송일시[i]] = self.ts_schedule.parttime[i]

    def get_show_id(self):
        """
        :objective: get show id for each day
        :return: pandas dataframe
        """
        if self.type == "hungarian": 
            self.train['show_id'] = [d.strftime("%Y-%m-%d %H") for d in self.train['방송일시']]

        elif self.type == "hungarian_h1":
            tmp = self.train.copy().iloc[:125].reset_index() #total time sequence of each product first week of june
            tmp['show_counts'] = np.nan
            for i in tmp.ymd.unique():
                rtn = tmp[tmp.ymd == i]
                slot_count = 0  # number of shows for each day
                for j in range(0, len(rtn)):
                    if rtn['parttime'].iloc[j] == 1:
                        slot_count += 1
                        idx = tmp[tmp.ymd == i].index[j]
                        tmp.show_counts.iloc[idx] = str(i) + " " + str(slot_count)
            show_id_seq = tmp['show_counts'].fillna(method='ffill')
            self.train['show_id'] = show_id_seq.to_list() * 125
        else:
            self.ts_schedule['show_counts'] = ""
            for i in self.ts_schedule.ymd.unique():
                rtn = self.ts_schedule[self.ts_schedule.ymd == i]
                slot_count = 0  # number of shows for each day
                for j in range(0, len(rtn)):
                    if rtn['parttime'].iloc[j] == 1:
                        slot_count += 1
                        idx = self.ts_schedule[self.ts_schedule.ymd == i].index[j]
                        self.ts_schedule.show_counts.iloc[idx] = str(i) + " " + str(slot_count)

    def get_min_range(self):
        """
        :objective: get minutes aired for each show
        :return: pandas dataframe
        """
        if self.type in ["hungarian", "hungarian_h1"]:
            self.train['min_range'] = 60
        else:
            self.ts_schedule['min_range'] = ""
            for i in range(0, len(self.ts_schedule)):
                if self.ts_schedule.parttime.iloc[i] == 1:
                    min_dur = self.ts_schedule.exposed.iloc[i]
                    j = i + 1
                    if j == (len(self.ts_schedule)): break
                    while self.ts_schedule.parttime.iloc[j] != 1:
                        min_dur += self.ts_schedule.exposed.iloc[j]
                        j += 1
                        if j == (len(self.ts_schedule)): break
                self.ts_schedule.min_range.iloc[i:j] = min_dur

    def add_showid_minran_to_train(self):
        """
        :objective: add show_id and min_range column to train data
        :return: pandas dataframe
        """
        self.train['min_range'] = ""
        self.train['show_id'] = ""
        for i in self.ts_schedule[self.ts_schedule['show_counts'] != ""].index:
            show_id = self.ts_schedule.show_counts.iloc[i]
            time_slot = self.ts_schedule.방송일시.iloc[i]
            minrange = self.ts_schedule.min_range.iloc[i]
            idx = self.train[(self.train.방송일시 >= time_slot) & (
                self.train.방송일시 < time_slot + datetime.timedelta(minutes=minrange))].index
            self.train.show_id.iloc[idx] = show_id
            self.train.min_range.iloc[idx] = minrange

    ############################
    ## primetime
    ############################
    def get_primetime(self):
        """
        :**objective: get primetime for week and weekends respectively
        """
        self.train['primetime'] = 0
        prime_week = [9, 10, 11]
        prime_week2 = [16, 17, 18]
        prime_weekend = [7, 8, 9]
        prime_weekend2 = [13, 15, 16, 17]

        self.train.loc[(self.train['red'] == 0) & (self.train['holidays'] == 1) & (
            self.train['hours'].isin(prime_weekend)), 'primetime'] = 1
        self.train.loc[(self.train['red'] == 0) & (self.train['holidays'] == 1) & (
            self.train['hours'].isin(prime_weekend2)), 'primetime'] = 2

        self.train.loc[(self.train['holidays'] == 0) & (self.train['hours'].isin(prime_week)), 'primetime'] = 1
        self.train.loc[(self.train['holidays'] == 0) & (self.train['hours'].isin(prime_week2)), 'primetime'] = 2

    def check_smallc_primet(self):
        """
        :objective: return 1 if its hour is within its small c's primetime
        """
        if self.type == 'train':
            # create table with top two sales hourly+small_c
            hours_smallc = self.train.groupby(['hours', 'small_c']) \
                ['취급액'].sum().rename("tot_sales").groupby(level=0, group_keys=False)
            hours_smallc_list = hours_smallc.nlargest(2)
            try:
                # write pickle file to be used later (when creating test)
                pd.DataFrame(hours_smallc_list).to_pickle("../data/12/small_c_primet.pkl")
            except:
                print("fail to write pickle")
        else:
            # if test, load pickle data storing prime time info based on train data
            hours_smallc_list = pd.read_pickle("../data/12/small_c_primet.pkl")

        # define blank col
        self.train['prime_smallc'] = 0
        for hr, small_c_nm in hours_smallc_list.index:
            self.train.prime_smallc.loc[(self.train.hours == hr) & (self.train.small_c == small_c_nm)] = 1

    ############################
    ## sales/volume power variables
    ############################
    def get_sales_power(self):
        """
        :objective: get sales power of each product, sum(exposed time)/sum(sales volume)
                    will not be included in the final input
        """
        if self.type == 'train':
            # will not be calculated when creating test
            self.train['sales_power'] = 0
            # sum(exposed time)/sum(sold volume)
            bp = self.train.groupby('상품코드').exposed.sum() / self.train.groupby('상품코드').volume.sum()
            for i in bp.index:
                self.train.sales_power.loc[self.train.상품코드 == i] = bp.loc[i]

    def freq_items(self):
        """
        :objective: identify frequently sold items by dummy variable "freq"
        """
        if self.type == 'train':
            # define top ten frequently sold items list
            freq_list = self.train.groupby('마더코드').show_id.nunique().sort_values(ascending=False).index[1:100]
            try:
                # write pickle file to be used later (when creating test)
                pd.DataFrame(freq_list).to_pickle("../data/12/freq_list.pkl")
            except:
                print("fail to write pickle")
        else:
            freq_list = pd.read_pickle("../data/12/freq_list.pkl")
            freq_list = freq_list['마더코드']
            freq_list = freq_list.to_list()
        self.train['freq'] = 0
        self.train.freq.loc[self.train.마더코드.isin(freq_list)] = 1

    def check_steady_sellers(self):
        """
        :objective: check if it is included in top 40(by total sales)
        """
        if self.type == 'train':
            # define top ten steady items list
            steady_list = self.train.groupby('마더코드') \
                              .apply(lambda x: sum(x.취급액) / x.show_id.nunique()).sort_values(ascending=False).index[1:100]
            try:
                pd.DataFrame(steady_list).to_pickle("../data/12/steady_list.pkl")
            except:
                print("fail to write pickle")
        else:
            steady_list = pd.read_pickle("../data/12/steady_list.pkl")
            steady_list = steady_list['마더코드']
            steady_list = steady_list.to_list()
        self.train['steady'] = 0
        self.train.steady.loc[self.train.마더코드.isin(steady_list)] = 1

    def check_brand_power(self):
        """
        :objective: identify items with low sales power(+) & high price
        """
        if self.type == 'train':
            # define top ten bpower items list
            bpower_list = self.train.마더코드.loc[(self.train.sales_power > self.train.sales_power.quantile(0.7)) &
                                              (self.train.판매단가 > self.train.판매단가.quantile(0.7))].unique()
            try:
                pd.DataFrame(bpower_list).to_pickle("../data/12/bpower_list.pkl")
            except:
                print("fail to write pickle")
        else:
            bpower_list = pd.read_pickle("../data/12/bpower_list.pkl")
            bpower_list = bpower_list.loc[:,0]
            bpower_list = bpower_list.to_list()

        self.train['bpower'] = 0
        self.train.bpower.loc[self.train.마더코드.isin(bpower_list)] = 1

    ############################
    ## Other characteristics
    ############################
    def check_men_items(self):
        """
        :objective: create a dummy variable to identify products for men
        """
        mens_category = ["의류", "이미용", "잡화", "속옷"]  # only for these categories
        self.train['men'] = 0
        self.train.men[self.train['상품군'].isin(mens_category) & self.train['상품명'].str.contains("남성")] = 1

    def check_luxury_items(self):
        """
        :**objective: create a dummy variable to identify products with selling price >= 490,000
        """
        self.train['luxury'] = 0
        self.train.loc[self.train['판매단가'] >= 490000, 'luxury'] = 1

    def check_pay(self):
        """
        :**objective: create 3 factor variable to identify payment methods ('ilsibul','muiza','none')
        """
        pay = []
        for i in range(0, len(self.train)):
            word = self.train['상품명'].iloc[i]
            if '(일)' in word or '일시불' in word:
                pay.append('ilsibul')
            elif '(무)' in word or '무이자' in word:
                pay.append('muiza')
            else:
                pay.append('none')
        self.train['pay'] = pay

    def get_dup_times(self):
        """
        :objective: get # of shows within the same category in a day
        """
        self.train['dup_times'] = 0
        dup_times_list = self.train.groupby(['ymd', '상품군']) \
            .show_id.nunique()
        for ymd_idx, cate_idx in dup_times_list.index:
            val = dup_times_list.loc[([(ymd_idx, cate_idx)])].values[0]
            self.train.dup_times.loc[(self.train.ymd == ymd_idx) & (self.train.상품군 == cate_idx)] = val

    def get_dup_times_smallc(self):
        """
        :objective: get # of shows within the same small_c in a day
        """
        self.train['dup_times_smallc'] = 0
        dup_times_small_list = self.train.groupby(['ymd', 'small_c']) \
            .show_id.nunique()
        for ymd_idx, cate_idx in dup_times_small_list.index:
            val = dup_times_small_list.loc[([(ymd_idx, cate_idx)])].values[0]
            self.train.dup_times_smallc.loc[(self.train.ymd == ymd_idx) & (self.train.small_c == cate_idx)] = val

    ############################
    ## Lag features
    ############################
    def get_lag_scode_price(self):
        """
        :**objective: get previous week scode price
        """
        self.train['lag_scode_price'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0)|((self.type == 'test')&(prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            groups = train_subset[['상품코드', '판매단가']].groupby(by='상품코드')
            grp = groups.agg({'판매단가': 'mean'}).reset_index()
            grp['week_num'] = curr_wk
            grp = grp.rename(columns={'판매단가': 'lag_scode_price_temp'})
            result = pd.merge(left=grp, right=self.train, on=['week_num', '상품코드'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_scode_price'] = result.loc[
                result.week_num == curr_wk, 'lag_scode_price_temp']

    def get_lag_scode_count(self):
        """
        :**objective: get previous week scode onair count
        """
        self.train['lag_scode_count'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            grp = train_subset.groupby(by='상품코드').apply(lambda x: x.show_id.nunique())
            grp = grp.to_frame(name='lag_scode_count_temp')
            grp['week_num'] = curr_wk
            result = pd.merge(left=grp, right=self.train, on=['week_num', '상품코드'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_scode_count'] = \
                result.loc[result.week_num == curr_wk, 'lag_scode_count_temp']

    def get_lag_mcode_price(self):
        """
        :**objective: get previous week mcode price
        """
        self.train['lag_mcode_price'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            groups = train_subset[['마더코드', '판매단가']].groupby(by='마더코드')
            grp = groups.agg({'판매단가': 'mean'}).reset_index()
            grp['week_num'] = curr_wk
            grp = grp.rename(columns={'판매단가': 'lag_mcode_price_temp'})
            result = pd.merge(left=grp, right=self.train, on=['week_num', '마더코드'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_mcode_price'] = result.loc[
                result.week_num == curr_wk, 'lag_mcode_price_temp']

    def get_lag_mcode_count(self):
        """
        :**objective: get previous week mcode onair count
        """
        self.train['lag_mcode_count'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            grp = train_subset.groupby(by='마더코드').apply(lambda x: x.show_id.nunique())
            grp = grp.to_frame(name='lag_mcode_count_temp')
            grp['week_num'] = curr_wk
            result = pd.merge(left=grp, right=self.train, on=['week_num', '마더코드'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_mcode_count'] = result.loc[
                result.week_num == curr_wk, 'lag_mcode_count_temp']

    def get_lag_bigcat_price(self):
        """
        :**objective: get previous week bigcat price
        """
        self.train['lag_bigcat_price'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            groups = train_subset[['상품군', '판매단가']].groupby(by='상품군')
            grp = groups.agg({'판매단가': 'mean'}).reset_index()
            grp['week_num'] = curr_wk
            grp = grp.rename(columns={'판매단가': 'lag_bigcat_price_temp'})
            result = pd.merge(left=grp, right=self.train, on=['week_num', '상품군'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_bigcat_price'] = result.loc[
                result.week_num == curr_wk, 'lag_bigcat_price_temp']

    def get_lag_bigcat_count(self):
        """
        :**objective: get previous week bigcat onair count
        """
        self.train['lag_bigcat_count'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            grp = train_subset.groupby(by='상품군').apply(lambda x: x.show_id.nunique())
            grp = grp.to_frame(name='lag_bigcat_count_temp')
            grp['week_num'] = curr_wk
            result = pd.merge(left=grp, right=self.train, on=['week_num', '상품군'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_bigcat_count'] = result.loc[
                result.week_num == curr_wk, 'lag_bigcat_count_temp']

    def get_lag_bigcat_price_day(self):
        """
        :**objective: get previous day bigcat price
        """
        self.train['lag_bigcat_price_day'] = 0
        daynums = self.train['ymd'].unique()
        for i in range(0, len(daynums)):
            if i == 0:
                continue
            curr_wk = daynums[i]
            prev_wk = daynums[i - 1]
            prev_wk_selector = (self.train['ymd'] == prev_wk)
            train_subset = self.train[prev_wk_selector]
            groups = train_subset[['상품군', '판매단가']].groupby(by='상품군')
            grp = groups.agg({'판매단가': 'mean'}).reset_index()
            grp['ymd'] = curr_wk
            grp = grp.rename(columns={'판매단가': 'lag_bigcat_price_day_temp'})
            result = pd.merge(left=grp, right=self.train, on=['ymd', '상품군'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['ymd'] == curr_wk, 'lag_bigcat_price_day'] = result.loc[
                result.ymd == curr_wk, 'lag_bigcat_price_day_temp']

    def get_lag_bigcat_count_day(self):
        """
        :**objective: get previous day bigcat onair count
        """
        self.train['lag_bigcat_count_day'] = 0
        daynums = self.train['ymd'].unique()
        for i in range(0, len(daynums)):
            if i == 0:
                continue
            curr_wk = daynums[i]
            prev_wk = daynums[i - 1]
            prev_wk_selector = (self.train['ymd'] == prev_wk)
            train_subset = self.train[prev_wk_selector]
            grp = train_subset.groupby(by='상품군').apply(lambda x: x.show_id.nunique())
            grp = grp.to_frame(name='lag_bigcat_count_day_temp')
            grp['ymd'] = curr_wk
            result = pd.merge(left=grp, right=self.train, on=['ymd', '상품군'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['ymd'] == curr_wk, 'lag_bigcat_count_day'] = result.loc[
                result.ymd == curr_wk, 'lag_bigcat_count_day_temp']

    def get_lag_small_c_price(self):
        """
        :**objective: get previous week small_c price
        """
        self.train['lag_small_c_price'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            groups = train_subset[['small_c', '판매단가']].groupby(by='small_c')
            grp = groups.agg({'판매단가': 'mean'}).reset_index()
            grp['week_num'] = curr_wk
            grp = grp.rename(columns={'판매단가': 'lag_small_c_price_temp'})
            result = pd.merge(left=grp, right=self.train, on=['week_num', 'small_c'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_small_c_price'] = result.loc[
                result.week_num == curr_wk, 'lag_small_c_price_temp']

    def get_lag_small_c_count(self):
        """
        :**objective: get previous week small_c onair count
        """
        self.train['lag_small_c_count'] = 0
        weeknums = self.train['week_num'].unique()
        for num in weeknums:
            curr_wk = num
            prev_wk = curr_wk - 1
            prev_wk_selector = (self.train['week_num'] == prev_wk)
            if (prev_wk == 0) | ((self.type == 'test') & (prev_wk == 22)):
                continue
            train_subset = self.train[prev_wk_selector]
            grp = train_subset.groupby(by='small_c').apply(lambda x: x.show_id.nunique())
            grp = grp.to_frame(name='lag_small_c_count_temp')
            grp['week_num'] = curr_wk
            result = pd.merge(left=grp, right=self.train, on=['week_num', 'small_c'], how='right')
            result.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)  ##
            result = result.reset_index()
            # merge
            self.train.loc[self.train['week_num'] == curr_wk, 'lag_small_c_count'] = result.loc[
                result.week_num == curr_wk, 'lag_small_c_count_temp']

    def get_lag_all_price_show(self):
        """
        :**objective: get previous week small_c mean price
        """
        self.train['lag_all_price_show'] = 0
        daynums = self.train['show_id'].unique()
        for i in range(0, len(daynums)):
            if i == 0:
                continue
            curr_wk = daynums[i]
            prev_wk = daynums[i - 1]
            prev_wk_selector = (self.train['show_id'] == prev_wk)
            train_subset = self.train[prev_wk_selector]
            mean_price = train_subset.판매단가.mean()
            # merge
            self.train.loc[self.train['show_id'] == curr_wk, 'lag_all_price_show'] = mean_price

    def get_lag_all_price_day(self):
        """
        :**objective: get previous day small_c mean price
        """
        self.train['lag_all_price_day'] = 0
        daynums = self.train['show_id'].unique()
        for i in range(0, len(daynums)):
            if i == 0:
                continue
            curr_wk = daynums[i]
            prev_wk = daynums[i - 1]
            prev_wk_selector = (self.train['show_id'] == prev_wk)
            train_subset = self.train[prev_wk_selector]
            mean_price = train_subset.판매단가.mean()
            # merge
            self.train.loc[self.train['show_id'] == curr_wk, 'lag_all_price_day'] = mean_price

    def get_lag_sales(self):
        """
        :objective: get sales lag 1,2 for weekend cases and lag 1 to 5 for weekday cases; pointwisely
        :return: pd.DataFrame - including lag_sales_wk_i (i = 1,2), lag_sales_wd_i (i = 1,2,3,4,5)
        """
        # use 2019-december data to get lag vars if self.type == 'test'
        if self.type != 'train':
            self.train['day_hour'] = self.train.days.astype(str) + '/' + self.train.hours.astype(str)
            if self.not_divided:
                full_train = pd.read_pickle(FEATURED_DATA_DIR + "train_fin_light_ver.pkl")
                lag_cols = ['day_hour'] + full_lag_col
            else:
                full_train = pd.read_pickle(FEATURED_DATA_DIR + "train_v2.pkl")
                lag_cols = ['day_hour'] + lag_wd + lag_wk

            if self.counterfactual:
                train_cf = full_train.loc[
                    (full_train.months == 6) | ((full_train.months == 7) & (full_train.days.isin([1, 2])))]
                train_cf['day_hour'] = (train_cf.days - 2).astype(str) + '/' + train_cf.hours.astype(str)
                train_cf_lags = train_cf[lag_cols].groupby(['day_hour']).mean()
                train_cf_lags.reset_index(inplace=True)

                self.train = pd.merge(left=self.train, right=train_cf_lags[lag_cols], how='left',
                                      on=['day_hour'])
                self.train.drop(['day_hour'], axis=1, inplace=True)
            else:
                # extract 2019-dec data
                train_dec = full_train.loc[(full_train.months == 12)]
                train_dec.loc[train_dec.상품군 == '건강기능'].취급액 = 1.112*(train_dec.loc[train_dec.상품군 == '건강기능'].취급액)
                train_dec.loc[train_dec.상품군 == '농수축'].취급액 = 1.141*(train_dec.loc[train_dec.상품군 == '농수축'].취급액)
                train_dec.loc[train_dec.상품군 == '이미용'].취급액 = 0.994*(train_dec.loc[train_dec.상품군 == '이미용'].취급액)
                train_dec.loc[train_dec.상품군 == '의류'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '의류'].취급액)
                train_dec.loc[train_dec.상품군 == '속옷'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '속옷'].취급액)
                train_dec.loc[train_dec.상품군 == '잡화'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '잡화'].취급액)
                train_dec.loc[train_dec.상품군 == '생활용품'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '생활용품'].취급액)
                train_dec.loc[train_dec.상품군 == '가전'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '가전'].취급액)
                train_dec.loc[train_dec.상품군 == '주방'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '주방'].취급액)
                train_dec.loc[train_dec.상품군 == '침구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '침구'].취급액)
                train_dec.loc[train_dec.상품군 == '가구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '가구'].취급액)

                # extract 2019-jun data
                train_jun = full_train.loc[(full_train.months == 6) | ((full_train.months == 7)  & (full_train.days.isin([1,2])))]
                train_jun.loc[train_jun.상품군 == '건강기능'].취급액 = 1.12*(train_jun.loc[train_jun.상품군 == '건강기능'].취급액)
                train_jun.loc[train_jun.상품군 == '농수축'].취급액 = 1.307*(train_jun.loc[train_jun.상품군 == '농수축'].취급액)
                train_jun.loc[train_jun.상품군 == '이미용'].취급액 = 0.897*(train_jun.loc[train_jun.상품군 == '이미용'].취급액)
                train_jun.loc[train_jun.상품군 == '의류'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '의류'].취급액)
                train_jun.loc[train_jun.상품군 == '속옷'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '속옷'].취급액)
                train_jun.loc[train_jun.상품군 == '잡화'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '잡화'].취급액)
                train_jun.loc[train_jun.상품군 == '생활용품'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '생활용품'].취급액)
                train_jun.loc[train_jun.상품군 == '가전'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '가전'].취급액)
                train_jun.loc[train_jun.상품군 == '주방'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '주방'].취급액)
                train_jun.loc[train_jun.상품군 == '침구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '침구'].취급액)
                train_jun.loc[train_jun.상품군 == '가구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '가구'].취급액)

                # define new 'day_hour' column to match df
                # lag values of train_dec will be injected by day_hour
                # need to extract one day from december data since Dec 1st is Sunday but June 1st is Monday
                train_dec['day_hour'] = (train_dec.days - 1).astype(str) + '/' + train_dec.hours.astype(str)
                train_jun['day_hour'] = (train_jun.days - 2).astype(str) + '/' + train_jun.hours.astype(str)

                train_dec_lags = train_dec[lag_cols].groupby(['day_hour']).mean()
                train_dec_lags.reset_index(inplace=True)
                train_jun_lags = train_jun[lag_cols].groupby(['day_hour']).mean()
                train_jun_lags.reset_index(inplace=True)

                train_lags = pd.concat([train_dec_lags, train_jun_lags]).groupby('day_hour').mean()
                train_lags.reset_index(inplace=True)
                self.train = pd.merge(left=self.train, right=train_lags[lag_cols], how='left',
                                 on=['day_hour'])
                self.train.drop(['day_hour'], axis=1, inplace=True)

        else:
            if self.not_divided:
                df_wkwd = self.train
                for i in [1,2,5,7]:  # lags for 1,5 diff
                    temp_df = df_wkwd[['ymd', 'hours', '취급액']]
                    temp_df = pd.DataFrame(temp_df.groupby(['ymd', 'hours'])['취급액'].sum())
                    temp_df = pd.DataFrame(temp_df.groupby(level=1)['취급액'].shift(i))
                    df_wkwd = pd.merge(right=temp_df, left=df_wkwd, on=['ymd', 'hours'], how='left')
                    df_wkwd = df_wkwd.rename(columns={'취급액_x': '취급액', '취급액_y': 'lag_sales_' + str(i)})
                self.train = df_wkwd

            else:
                df_wk = self.train[self.train.weekends == 1]  # weekend case
                for i in range(1, 3): # lags for 1,2 diff
                    temp_df = df_wk[['ymd', 'hours', '취급액']]
                    temp_df = pd.DataFrame(temp_df.groupby(['ymd', 'hours'])['취급액'].sum())
                    temp_df = pd.DataFrame(temp_df.groupby(level=1)['취급액'].shift(i))
                    df_wk = pd.merge(right=temp_df, left=df_wk, on=['ymd', 'hours'], how='left')
                    df_wk = df_wk.rename(columns={'취급액_x': '취급액', '취급액_y': 'lag_sales_wk_' + str(i)})

                df_wd = self.train[self.train.weekends == 0]  # weekday case
                for i in range(1, 6): # lags for 1,5 day diff
                    temp_df = df_wd[['ymd', 'hours', '취급액']]
                    temp_df = pd.DataFrame(temp_df.groupby(['ymd', 'hours'])['취급액'].sum())
                    temp_df = pd.DataFrame(temp_df.groupby(level=1)['취급액'].shift(i))
                    df_wd = pd.merge(right=temp_df, left=df_wd, on=['ymd', 'hours'], how='left')
                    df_wd = df_wd.rename(columns={'취급액_x': '취급액', '취급액_y': 'lag_sales_wd_' + str(i)})

                self.train = pd.concat([df_wd, df_wk], join='outer', sort=False)


    def get_rolling_means(self):
        """
        :objective: compute rolling means by 상품군 for 7/14/21/28 days
        :param: df - pd.DataFrame
        :return: pd.DataFrme - including rolling_mean_i (i=7,14,21,28)
        """
        # stack 2019-12 data to get lag vars if self.type == 'test'
        if self.type != "train":
            full_train = pd.read_pickle(FEATURED_DATA_DIR + "train_v2.pkl")
            
            # extract 2019-dec data
            train_dec = full_train.loc[(full_train.months == 12)]
            train_dec.loc[train_dec.상품군 == '건강기능'].취급액 = 1.112*(train_dec.loc[train_dec.상품군 == '건강기능'].취급액)
            train_dec.loc[train_dec.상품군 == '농수축'].취급액 = 1.141*(train_dec.loc[train_dec.상품군 == '농수축'].취급액)
            train_dec.loc[train_dec.상품군 == '이미용'].취급액 = 0.994*(train_dec.loc[train_dec.상품군 == '이미용'].취급액)
            train_dec.loc[train_dec.상품군 == '의류'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '의류'].취급액)
            train_dec.loc[train_dec.상품군 == '속옷'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '속옷'].취급액)
            train_dec.loc[train_dec.상품군 == '잡화'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '잡화'].취급액)
            train_dec.loc[train_dec.상품군 == '생활용품'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '생활용품'].취급액)
            train_dec.loc[train_dec.상품군 == '가전'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '가전'].취급액)
            train_dec.loc[train_dec.상품군 == '주방'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '주방'].취급액)
            train_dec.loc[train_dec.상품군 == '침구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '침구'].취급액)
            train_dec.loc[train_dec.상품군 == '가구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '가구'].취급액)
            
            # extract 2019-jun data
            train_jun = full_train.loc[(full_train.months == 6) | ((full_train.months == 7)  & (full_train.days.isin([1,2])))]
            train_jun.loc[train_jun.상품군 == '건강기능'].취급액 = 1.12*(train_jun.loc[train_jun.상품군 == '건강기능'].취급액)
            train_jun.loc[train_jun.상품군 == '농수축'].취급액 = 1.307*(train_jun.loc[train_jun.상품군 == '농수축'].취급액)
            train_jun.loc[train_jun.상품군 == '이미용'].취급액 = 0.897*(train_jun.loc[train_jun.상품군 == '이미용'].취급액)
            train_jun.loc[train_jun.상품군 == '의류'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '의류'].취급액)
            train_jun.loc[train_jun.상품군 == '속옷'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '속옷'].취급액)
            train_jun.loc[train_jun.상품군 == '잡화'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '잡화'].취급액)
            train_jun.loc[train_jun.상품군 == '생활용품'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '생활용품'].취급액)
            train_jun.loc[train_jun.상품군 == '가전'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '가전'].취급액)
            train_jun.loc[train_jun.상품군 == '주방'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '주방'].취급액)
            train_jun.loc[train_jun.상품군 == '침구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '침구'].취급액)
            train_jun.loc[train_jun.상품군 == '가구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '가구'].취급액)
            
            train_dec.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            train_dec['days'] = train_dec.days - 1
            lag_cols = ['days', '상품군', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_21', 'rolling_mean_28']
            train_dec_lags = train_dec[lag_cols].groupby(['days', '상품군']).mean()
            train_dec_lags.reset_index(inplace=True)

            train_jun.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            train_jun['days'] = train_jun.days - 2
            train_jun_lags = train_jun[lag_cols].groupby(['days', '상품군']).mean()
            train_jun_lags.reset_index(inplace=True)
            
            train_lags = pd.concat([train_dec_lags, train_jun_lags]).groupby(['days', '상품군']).mean()
            train_lags.reset_index(inplace=True)
            
            if self.counterfactual :
                train_cf = full_train.loc[(full_train.months == 6) | ((full_train.months == 7)  & (full_train.days.isin([1,2])))]
                train_cf.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
                train_cf['days'] = train_cf.days - 2
                train_cf_lags = train_cf[lag_cols].groupby(['days', '상품군']).mean()
                train_cf_lags.reset_index(inplace=True)
                self.train = pd.merge(left=self.train, right=train_cf_lags[lag_cols], how='left',
                      on=['days', '상품군'])
            else :
                self.train = pd.merge(left=self.train, right=train_lags[lag_cols], how='left',
                                  on=['days', '상품군'])
        else:
            for i in [7, 14, 21, 28]:
                self.train['rolling_mean_' + str(i)] = 0
                for time_slot in self.train.ymd.unique():
                    time_slot = pd.to_datetime(time_slot)
                    for category in self.train.상품군.unique():
                         self.train['rolling_mean_' + str(i)].loc[(self.train.ymd == time_slot) &
                                                                  (self.train.상품군 == category)] =\
                                self.train[(self.train.방송일시 >= time_slot - np.timedelta64(i,'D')) &
                                    (self.train.방송일시 < time_slot) & (self.train.상품군 == category)].취급액.mean()

    def get_mean_sales_origin(self):
        """
        :objective: compute mean sales of original codes
        :param: df - pd.DataFrame
        :return: pd.DataFrme - including mean_sales_origin
        """
        # stack 2019-12 data to get lag vars if self.type == 'test'
        if self.type != 'train':
            full_train = pd.read_pickle("../data/20/train_fin_light_ver.pkl")
            # extract 2019-dec data
            train_dec = full_train.loc[(full_train.months == 12)]
            train_dec.loc[train_dec.상품군 == '건강기능'].취급액 = 1.112*(train_dec.loc[train_dec.상품군 == '건강기능'].취급액)
            train_dec.loc[train_dec.상품군 == '농수축'].취급액 = 1.141*(train_dec.loc[train_dec.상품군 == '농수축'].취급액)
            train_dec.loc[train_dec.상품군 == '이미용'].취급액 = 0.994*(train_dec.loc[train_dec.상품군 == '이미용'].취급액)
            train_dec.loc[train_dec.상품군 == '의류'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '의류'].취급액)
            train_dec.loc[train_dec.상품군 == '속옷'].취급액 = 0.792*(train_dec.loc[train_dec.상품군 == '속옷'].취급액)
            train_dec.loc[train_dec.상품군 == '잡화'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '잡화'].취급액)
            train_dec.loc[train_dec.상품군 == '생활용품'].취급액 = 1.357*(train_dec.loc[train_dec.상품군 == '생활용품'].취급액)
            train_dec.loc[train_dec.상품군 == '가전'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '가전'].취급액)
            train_dec.loc[train_dec.상품군 == '주방'].취급액 = 1.227*(train_dec.loc[train_dec.상품군 == '주방'].취급액)
            train_dec.loc[train_dec.상품군 == '침구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '침구'].취급액)
            train_dec.loc[train_dec.상품군 == '가구'].취급액 = 0.665*(train_dec.loc[train_dec.상품군 == '가구'].취급액)
            
            # extract 2019-jun data
            train_jun = full_train.loc[(full_train.months == 6) | ((full_train.months == 7)  & (full_train.days.isin([1,2])))]
            train_jun.loc[train_jun.상품군 == '건강기능'].취급액 = 1.12*(train_jun.loc[train_jun.상품군 == '건강기능'].취급액)
            train_jun.loc[train_jun.상품군 == '농수축'].취급액 = 1.307*(train_jun.loc[train_jun.상품군 == '농수축'].취급액)
            train_jun.loc[train_jun.상품군 == '이미용'].취급액 = 0.897*(train_jun.loc[train_jun.상품군 == '이미용'].취급액)
            train_jun.loc[train_jun.상품군 == '의류'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '의류'].취급액)
            train_jun.loc[train_jun.상품군 == '속옷'].취급액 = 0.752*(train_jun.loc[train_jun.상품군 == '속옷'].취급액)
            train_jun.loc[train_jun.상품군 == '잡화'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '잡화'].취급액)
            train_jun.loc[train_jun.상품군 == '생활용품'].취급액 = 1.351*(train_jun.loc[train_jun.상품군 == '생활용품'].취급액)
            train_jun.loc[train_jun.상품군 == '가전'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '가전'].취급액)
            train_jun.loc[train_jun.상품군 == '주방'].취급액 = 0.946*(train_jun.loc[train_jun.상품군 == '주방'].취급액)
            train_jun.loc[train_jun.상품군 == '침구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '침구'].취급액)
            train_jun.loc[train_jun.상품군 == '가구'].취급액 = 1.052*(train_jun.loc[train_jun.상품군 == '가구'].취급액)
            
            history_sales = pd.concat([train_dec, train_jun]).groupby('original_c').취급액.mean().to_frame()
            history_sales.reset_index(inplace=True)
            history_sales = history_sales.rename(columns = {'취급액':'mean_sales_origin'})
            
            if self.counterfactual:
                train_cf = full_train.loc[(full_train.months == 6) | ((full_train.months == 7)  & (full_train.days.isin([1,2])))]
                history_sales_cf = train_cf.groupby('original_c').취급액.mean().to_frame()
                history_sales_cf.reset_index(inplace=True)
                history_sales_cf = history_sales_cf.rename(columns = {'취급액':'mean_sales_origin'})
                self.train = pd.merge(left=self.train, right=history_sales_cf, on='original_c', how="left")
            else :
                self.train = pd.merge(left=self.train, right=history_sales, on='original_c', how="left")
        else:
            month_ori = self.train[['months','original_c','취급액']].groupby(['months','original_c']).mean()
            month_ori.reset_index(inplace=True)
            month_ori = month_ori.rename(columns = {'취급액':'mean_sales_origin'})
            self.train = pd.merge(left = self.train, right = month_ori, on = ['months','original_c'], how = 'left')
   
    def get_ts_pred(self, type='Prophet'):
        """
        :objective: get sales estimation using time-series model
        :param type: str - 'Prophet'
        """
        if type == 'Prophet':
            ts_pred = pd.read_pickle("../data/12/prophet_pred.pkl")
        self.train = pd.merge(left=self.train, right=ts_pred[['ymd', 'hours', 'ts_pred']], on=['ymd', 'hours'], sort=False,
                 how='left')

    ############################
    ## External information
    ############################
    def add_categories(self):
        """
        :objective: add category columns
        :return: pandas dataframe
        """

        if self.type == 'test':
            categories = pd.read_excel("../data/01/2020sales_test_added.xlsx")
            categories.상품코드 = categories.상품코드.dropna().astype(int).astype(str).str.zfill(6)
            categories.방송일시 = pd.to_datetime(categories.방송일시, format="%Y/%m/%d %H:%M")
            categories.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            
            self.train = pd.merge(left=self.train, right=categories[['방송일시', '상품코드', 'small_c', 'small_c_code',
                                                                     'middle_c', 'middle_c_code', 'big_c',
                                                                     'big_c_code', 'original_c']],
                                  how='left', on=['방송일시', '상품코드'], sort=False)
        else:
            categories = pd.read_excel("../data/01/2019sales_added.xlsx")
            categories.rename(columns={' 취급액 ': '취급액'}, inplace=True)
            categories.상품코드 = categories.상품코드.dropna().astype(int).astype(str).str.zfill(6)
            categories.방송일시 = pd.to_datetime(categories.방송일시, format="%Y/%m/%d %H:%M")
            categories.sort_values(['방송일시', '상품코드'], ascending=[True, True], inplace=True)
            
            self.train = pd.merge(left=self.train, right=categories[['방송일시', '상품코드', 'small_c', 'small_c_code',
                                                                     'middle_c', 'middle_c_code', 'big_c',
                                                                     'big_c_code', 'original_c']],
                                  how='left', on=['방송일시', '상품코드'], sort=False)
            
    def get_season_items(self):
        """
        :objective: create dummy vars(spring,summer,fall,winter) for seasonal items
        """
        if self.type == "train":
            with open("../data/11/seasonal.json", encoding='UTF8') as json_file:
                seasonal_items = json.load(json_file)
        else:
            with open("../data/11/seasonal_test.json", encoding='UTF8') as json_file:
                seasonal_items = json.load(json_file)

        self.train['spring'] = 0
        self.train['summer'] = 0
        self.train['fall'] = 0
        self.train['winter'] = 0
        self.train.spring.loc[self.train['small_c'].isin(seasonal_items['spring'])] = 1
        self.train.summer.loc[self.train['small_c'].isin(seasonal_items['summer'])] = 1
        self.train.fall.loc[self.train['small_c'].isin(seasonal_items['fall'])] = 1
        self.train.winter.loc[self.train['small_c'].isin(seasonal_items['winter'])] = 1

    def add_small_c_clickr(self):
        """
        :objective: add click ratio column (small_c)
        """
        if self.counterfactual:
            pass
        else :
            smallc_comb = pd.read_excel("../data/11/small_comb.xlsx")
            smallc_comb['ymd'] = pd.to_datetime(smallc_comb.date.astype(str)).dt.date
            self.train = pd.merge(left=self.train,
                                  right=smallc_comb[['small_c_code', 'ymd', 'small_click_r']],
                                  how='left', on=['small_c_code', 'ymd'], sort=False)

    def add_mid_c_clickr(self):
        """
        :objective: add click ratio column (mid_c)
        """
        if self.counterfactual:
            pass
        else :
            midc_comb = pd.read_excel("../data/11/mid_comb.xlsx")
            midc_comb['ymd'] = pd.to_datetime(midc_comb.date.astype(str)).dt.date
            midc_comb['middle_c_code'] = midc_comb['mid_c_code']
            self.train = pd.merge(left=self.train,
                                  right=midc_comb[['middle_c_code', 'ymd', 'mid_click_r']],
                                  how='left', on=['middle_c_code', 'ymd'], sort=False)

    def add_big_c_clickr(self):
        """
        :objective: add click ratio column (big_c)
        """
        if self.counterfactual:
            pass
        else :
            bigc_comb = pd.read_excel("../data/11/big_comb.xlsx")
            bigc_comb['ymd'] = pd.to_datetime(bigc_comb.date.astype(str)).dt.date
            self.train = pd.merge(left=self.train,
                                  right=bigc_comb[['big_c_code', 'ymd', 'big_click_r']],
                                  how='left', on=['big_c_code', 'ymd'], sort=False)

    def add_age_click_ratio(self):
        """
        :objective: add click ratio by age
        :return:
        """
        if self.counterfactual:
            pass
        else :
            age_click = pd.read_excel("../data/11/age_click.xlsx")
            age_click['ymd'] = pd.to_datetime(age_click.date.astype(str)).dt.date
            self.train = pd.merge(left=self.train,
                                  right=age_click[['cat_code', 'ymd', 'age30', 'age40', 'age50', 'age60above']],
                                  how='left', left_on=['small_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False)
            self.train = pd.merge(left=self.train,
                                  right=age_click[['cat_code', 'ymd', 'age30', 'age40', 'age50', 'age60above']],
                                  how='left', left_on=['middle_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False,
                                  suffixes=['_small', '_middle'])
            self.train = pd.merge(left=self.train,
                                  right=age_click[['cat_code', 'ymd', 'age30', 'age40', 'age50', 'age60above']],
                                  how='left', left_on=['big_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False)
            self.train.drop(['cat_code', 'cat_code_small', 'cat_code_middle'], axis=1, inplace=True)
            self.train = self.train.rename(
                columns={'age30': 'age30_big', 'age40': 'age40_big', 'age50': 'age50_big', 'age60above': 'age60above_big'})

    def add_device_click_ratio(self):
        """
        :objective: add click ratio by device type(mobile/pc)
        :return:
        """
        if self.counterfactual:
            pass
        else:            
            device_click = pd.read_excel("../data/11/dev_click.xlsx")
            device_click['ymd'] = pd.to_datetime(device_click.date.astype(str)).dt.date
            self.train = pd.merge(left=self.train,
                                  right=device_click[['cat_code', 'ymd', 'pc', 'mobile']],
                                  how='left', left_on=['small_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False)
            self.train = pd.merge(left=self.train,
                                  right=device_click[['cat_code', 'ymd', 'pc', 'mobile']],
                                  how='left', left_on=['middle_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False,
                                  suffixes=['_small', '_middle'])
            self.train = pd.merge(left=self.train,
                                  right=device_click[['cat_code', 'ymd', 'pc', 'mobile']],
                                  how='left', left_on=['big_c_code', 'ymd'], right_on=['cat_code', 'ymd'], sort=False)
            self.train.drop(['cat_code', 'cat_code_small', 'cat_code_middle'], axis=1, inplace=True)
            self.train = self.train.rename(columns={'pc': 'pc_big', 'mobile': 'mobile_big'})

    def get_weather(self):
        """
        :objective: get weather(rain, temp_diff info)
        """
        if self.type == 'train':
            weather = pd.read_excel("../data/11/weather_diff.xlsx")
        else:
            weather = pd.read_excel("../data/11/weather_diff_test.xlsx")
            weather.ymd = pd.to_datetime(weather.ymd, format="%Y-%m-%d")

        weather.ymd = weather.ymd.dt.date
        self.train = pd.merge(left=self.train,
                              right=weather[['ymd', 'rain', 'temp_diff_s']],
                              how='left', on=['ymd'], sort=False)
        # self.train.rain.loc[self.train.rain.isna()] = weather.rain.loc[len(weather) - 1]
        # self.train.temp_diff_s.loc[self.train.temp_diff_s.isna()] = weather.temp_diff_s.loc[len(weather) - 1]

    ############################
    ## Combine
    ############################

    def drop_na(self):
        """
        :objective: drop na rows and 취급액 == 50000(train)
                    drop rows with 판매단가 == 0(test)
        """
        if (self.type == 'test')|(self.type == 'hungarian'):
            self.train = self.train[self.train['판매단가'] != 0]
        elif self.type == 'train':
            self.train = self.train[self.train['취급액'].notna()]
            self.train = self.train[self.train['취급액'] != 50000]

    def price_to_rate(self):
        """
        :objective: get ratio of lag price to current selling price
        """
        lag_price_col = ['lag_mcode_price', 'lag_bigcat_price', 'lag_small_c_price', 'lag_bigcat_price_day',
                         'lag_all_price_show', 'lag_all_price_day']
        for col in lag_price_col:
            self.train[col] = self.train[col] / self.train['판매단가']

    def run_all(self):

        self.get_time()
        self.get_weekday()
        self.get_hours_inweek()
        self.get_holidays()
        self.get_red_days()
        self.get_weekends()
        self.get_min_start()

        self.filter_jappingt()
        self.fill_exposed_na()

        self.get_ymd()
        self.timeslot()
        self.get_show_id()
        self.get_min_range()
        self.add_showid_minran_to_train()

        self.drop_na()
        self.add_categories()

        self.get_primetime()
        self.check_smallc_primet()

        self.get_sales_power()
        self.freq_items()
        self.get_dup_times()
        self.get_dup_times_smallc()

        self.get_lag_scode_price()
        self.get_lag_scode_count()
        self.get_lag_mcode_price()
        self.get_lag_mcode_count()
        self.get_lag_bigcat_price()
        self.get_lag_bigcat_count()
        self.get_lag_bigcat_price_day()
        self.get_lag_bigcat_count_day()
        self.get_lag_small_c_price()
        self.get_lag_small_c_count()
        self.get_lag_all_price_show()
        self.get_lag_all_price_day()

        self.check_brand_power()
        self.check_steady_sellers()
        self.check_men_items()
        self.check_luxury_items()
        self.check_pay()

        self.get_season_items()

        self.add_small_c_clickr()
        self.add_mid_c_clickr()
        self.add_big_c_clickr()

        self.get_weather()
        self.price_to_rate()
        self.round_exposed()
        
        self.add_age_click_ratio()
        self.add_device_click_ratio()
        self.get_rolling_means()
        self.get_mean_sales_origin()
        self.get_lag_sales()
        self.get_ts_pred()
        print(self.train.shape, ": df shape")

        if not self.not_divided:
            featured_wd_lag = self.train.loc[self.train.weekends == 0]
            featured_wd_lag.drop(columns=lag_wk, inplace=True)
            featured_wk_lag = self.train.loc[self.train.weekends == 1]
            featured_wk_lag.drop(columns=lag_wd, inplace=True)

            if self.type == "train":
                self.train.to_pickle(FEATURED_DATA_DIR + "{}_v2.pkl".format(self.type))

            if not self.counterfactual:
                featured_wk_lag.to_pickle(FEATURED_DATA_DIR + "{}_fin_wk_lag.pkl".format(self.type))
                featured_wd_lag.to_pickle(FEATURED_DATA_DIR + "{}_fin_wd_lag.pkl".format(self.type))
        else:
            self.train.to_pickle(FEATURED_DATA_DIR + "{}_fin_light_ver.pkl".format(self.type))
        return self.train

    def run_hungarian(self):
        self.drop_na()
        self.get_time()
        self.get_weekday()
        self.get_hours_inweek()
        self.get_holidays()
        self.get_red_days()
        self.get_weekends()
        self.filter_jappingt()

        self.timeslot()
        self.get_show_id()
        self.get_min_range()

        self.get_primetime()
        self.check_smallc_primet()
        self.freq_items()

        self.check_brand_power()
        self.check_steady_sellers()
        self.check_men_items()
        self.check_luxury_items()
        self.check_pay()

        self.get_season_items()
        self.add_small_c_clickr()
        self.add_mid_c_clickr()
        self.add_big_c_clickr()
        self.get_weather()

        self.add_age_click_ratio()
        self.add_device_click_ratio()
        self.get_rolling_means()
        self.get_mean_sales_origin()

        self.get_lag_sales()
        self.get_ts_pred()
        print(self.train.shape, ": df shape")

        return self.train


if __name__ == "__main__":
    train_lag = Features(types="train", not_divided=False)
    train_lag.run_all()
    train_full = Features(types="train", not_divided=True)
    train_full.run_all()
    test_lag = Features(types="test", not_divided=False)
    test_lag.run_all()
    test_lag = Features(types="test", not_divided=True)
    test_lag.run_all()

