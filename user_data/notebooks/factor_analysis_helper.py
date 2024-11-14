import pandas as pd
import numpy as np


def mad(df, n=3 * 1.4826):
    def filter_extreme_MAD(series, n):
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n * new_median, median + n * new_median)

    df = df.apply(lambda x: filter_extreme_MAD(x, n), axis=1)
    return df


def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def Quick_Factor_Return_N_IC(factor_df: pd.DataFrame, open_df: pd.DataFrame, n: int, name='', time_frame=''):
    return_n = open_df.pct_change(n).shift(-n - 1)
    result = factor_df.corrwith(return_n, axis=1, method='spearman').dropna(how='all')
    report = {
        'name': name,
        'time frame': time_frame,
        'IC mean': round(result.mean(), 4),
        'IC std': round(result.std(), 4),
        'IR': round(result.mean() / result.std(), 4),
        'IC>0': round(len(result[result > 0].dropna()) / len(result), 4),
        'ABS_IC>2%': round(len(result[abs(result) > 0.02].dropna()) / len(result), 4),
    }
    return result, report


def Factor_Group_Analysis(factor_df: pd.DataFrame, open_df: pd.DataFrame, group_count: int, change_position_period: int, factor_name: str):
    pair_list = factor_df.columns.to_list()
    return_df = open_df.pct_change().shift(-2).dropna(how='all', axis=0).stack()
    # Factor DF must be standardized
    group_df = factor_df.stack().to_frame('factor')
    group_df['current_return'] = return_df
    group_df.reset_index(inplace=True)
    group_df.columns = ['date', 'pair', 'factor', 'current_return']
    datetime_period = sorted(group_df.date.drop_duplicates().tolist())
    turnover_ratio = pd.DataFrame()
    group_return = pd.DataFrame()

    for i in range(0, len(datetime_period) - 1, change_position_period):
        single = group_df[group_df['date'] == datetime_period[i]].sort_values(by='factor')
        single.loc[:, 'group'] = pd.qcut(single.factor, group_count, list(range(1, group_count + 1))).to_list()
        group_dict = {}
        for j in range(1, group_count + 1):
            group_dict[j] = single[single.group == j].pair.to_list()

        turnover_ratio_temp = []
        if i == 0:
            temp_group_dict = group_dict
        else:
            for j in range(1, group_count + 1):
                turnover_ratio_temp.append(
                    len(list(set(temp_group_dict[j]).difference(set(group_dict[j])))) / len(set(temp_group_dict[j])))
            turnover_ratio = pd.concat([turnover_ratio,
                                        pd.DataFrame(turnover_ratio_temp,
                                                     index=['G{}'.format(j) for j in list(range(1, group_count + 1))],
                                                     columns=[datetime_period[i]]).T],
                                       axis=0)
            temp_group_dict = group_dict

        if i < len(datetime_period) - change_position_period:
            period = group_df[group_df.date.isin(datetime_period[i:i + change_position_period])]
        else:
            period = group_df[group_df.date.isin(datetime_period[i:])]

        group_return_temp = []
        for j in range(1, group_count + 1):
            group_return_temp.append(
                period[period.pair.isin(group_dict[j])].set_index(['date', 'pair']).current_return.unstack('pair').mean(
                    axis=1))
        group_return = pd.concat([group_return, pd.DataFrame(group_return_temp, index=['G{}'.format(j) for j in list(
            range(1, group_count + 1))]).T], axis=0)
        print('\r 当前：{} / 总量：{}'.format(i, len(datetime_period)), end='')

    group_return.dropna(how='all', axis=0, inplace=True)
    group_return['Benchmark'] = group_return.mean(axis=1)
    group_return = (group_return + 1).cumprod()

    group_weekly_ret = (group_return.iloc[-1] ** (24 * 7 / len(group_return)) - 1)
    group_weekly_ret -= group_weekly_ret.Benchmark
    group_weekly_ret = group_weekly_ret.drop('Benchmark').to_frame('weekly_ret')
    group_weekly_ret['group'] = list(range(1, group_count + 1))

    corr_value = round(group_weekly_ret.corr(method='spearman').iloc[0, 1], 4)

    return group_weekly_ret, corr_value