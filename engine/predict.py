# General imports
import warnings

warnings.filterwarnings("ignore")

# data
import pickle


from engine.utils import *


def predict():
    """
    :objective: run model on test data
    :return: pd.DataFrame, pd.DataFrame
    """
    # Load Models
    model_path = MODELS_DIR + 'lgbm_finalmodel_wd_all.bin'
    model_wd_all = pickle.load(open(model_path, 'rb'))

    model_path = MODELS_DIR + 'lgbm_finalmodel_wd_top.bin'
    model_wd_top = pickle.load(open(model_path, 'rb'))

    model_path = MODELS_DIR + 'lgbm_finalmodel_wk_all.bin'
    model_wk_all = pickle.load(open(model_path, 'rb'))

    model_path = MODELS_DIR + 'lgbm_finalmodel_wk_top.bin'
    model_wk_top = pickle.load(open(model_path, 'rb'))

    # wd
    test_wd_origin = load_df(FEATURED_DATA_DIR + 'test_fin_wd_lag.pkl').drop('index', axis=1)
    test_wd = test_wd_origin.copy()
    test_wd = run_preprocess(test_wd)
    test_wd = test_wd.drop(['show_id', TARGET], axis=1)
    test_wd_sort = test_wd.sort_values('mean_sales_origin', ascending=False)
    # Predict all observations
    pred_test_wd_all = model_wd_all.predict(test_wd)
    # Mixed DF (Top: 727개)
    test_mixed_wd = mixed_df(model_wd_top, test_wd_sort, test_wd, pred_test_wd_all, num_top=727)
    test_wd_origin[TARGET] = test_mixed_wd[TARGET]

    # wk
    test_wk_origin = load_df(FEATURED_DATA_DIR + 'test_fin_wk_lag.pkl').drop('index', axis=1)
    test_wk = test_wk_origin.copy()
    test_wk = run_preprocess(test_wk)
    test_wk = test_wk.drop(['show_id', TARGET], axis=1)
    test_wk_sort = test_wk.sort_values('mean_sales_origin', ascending=False)
    # Predict all observations
    pred_test_wk_all = model_wk_all.predict(test_wk)
    # Mixed DF (Top: 249개)
    test_mixed_wk = mixed_df(model_wk_top, test_wk_sort, test_wk, pred_test_wk_all, num_top=249)
    test_wk_origin[TARGET] = test_mixed_wk[TARGET]
    # two outputs
    return test_wd_origin, test_wk_origin


def submission(wd, wk):
    """
    create submission file
    :param wd: pd.DataFrame
    :param wk: pd.DataFrame
    :return:
    """
    test_final_wd = wd[['방송일시', '노출(분)', '마더코드', '상품코드', '상품명', '상품군', '판매단가', TARGET]]
    test_final_wk = wk[['방송일시', '노출(분)', '마더코드', '상품코드', '상품명', '상품군', '판매단가', TARGET]]
    test_final_full = pd.concat([test_final_wd, test_final_wk], axis=0)
    test_final_full.sort_values(['방송일시'], inplace=True)

    test_final_full.to_csv(SUBMISSION_DIR + 'submission.csv', index=False)


def mixed_df(model_top, top_df, val_all_df_x, preds_all, num_top):
    """
    :objective: mix two models' outputs
    :param model_top: LGBMRegressor
    :param top_df: df sorted by "mean sales origin"
    :param val_all_df_x: full df
    :param preds_all: predicted values
    :param num_top: index to split
    :return: pd.DataFrame
    """
    top_idx = set(top_df.iloc[:num_top, :].index)
    val_idx = set(val_all_df_x.index)
    top_in_val = list(val_idx.intersection(top_idx))

    val_copy = val_all_df_x.copy()
    val_copy[TARGET] = preds_all

    for i in top_in_val:
        val_copy[TARGET].loc[val_copy.index == i] = model_top.predict(val_all_df_x.loc[val_all_df_x.index == i])

    return val_copy


if __name__ == "__main__":
    test_wd_origin, test_wk_origin = predict()
    submission(test_wd_origin, test_wk_origin)
    print("finish to create submission files")


