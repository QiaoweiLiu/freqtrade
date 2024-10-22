# Factors Generator

# 1. Generate Useful Factors (DataFrame => Dataframe)
# 2. Single Factors validation (Time Scheduled)
# 2.1 Normalization
# 2.2 IC IR Check
# 2.3 Memorize the useful Factors
# Factors Name must fit freqtrade AI
import pandas as pd
import talib.abstract as ta
import os


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


def mad(df, n=3 * 1.4826):
    # MAD:中位数去极值
    def filter_extreme_MAD(series, n):
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n * new_median, median + n * new_median)

    # 离群值处理
    df = df.apply(lambda x: filter_extreme_MAD(x, n), axis=1)

    return df


def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


class FactorsGenerator:
    def __init__(self, time_frame, is_run_live=True):
        self.factor_methods = [] if is_run_live else self.get_factor_methods()
        self.time_frame = time_frame

    def generate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        for factor_method in self.factor_methods:
            factor = getattr(self, factor_method)
            df = self.calculate_single_factor(factor, factor_method, df)
            freqtrade_ai_factor_name = factor_method.replace('factor_', '%-')
            df.rename(columns={factor_method: freqtrade_ai_factor_name}, inplace=True)
        return df

    def load_factors_from_file(self):
        factors_info_df = pd.read_csv(f'./factors_info_{self.time_frame}.csv')
        if factors_info_df is not None and not factors_info_df.empty:
            self.factor_methods = factors_info_df['factor_method'].tolist()

    def validate_factors(self, df_dict: dict) -> None:
        factor_methods = self.get_factor_methods()
        self.factor_methods = []
        factors_info = []

        # Check is factors files exist
        for factor_method in factor_methods:
            # Combo all pairs factors
            all_pair_factor_df = pd.DataFrame()
            all_pair_open_df = pd.DataFrame()
            for pair in list(df_dict.keys()):
                # df = df_dict[pair].copy()
                factor = getattr(self, factor_method)
                factor_df = self.calculate_single_factor(factor, factor_method, df_dict[pair].copy())
                factor_df.rename(columns={factor_method: pair}, inplace=True)
                open_df = df_dict[pair].copy()
                open_df.rename(columns={'open': pair}, inplace=True)
                if all_pair_factor_df.shape[0] == 0:
                    all_pair_factor_df = factor_df[pair].to_frame()
                    all_pair_open_df = open_df[pair].to_frame()
                else:
                    all_pair_factor_df = all_pair_factor_df.join(factor_df[pair].to_frame())
                    all_pair_open_df = all_pair_open_df.join(open_df[pair].to_frame())
            # Normalization
            standardized_factor_df = standardize(mad(all_pair_factor_df))
            # IC IR Check
            target_n_map = {
                '5m': 6,
                '15m': 4,
                '1h': 2
            }
            result, report = Quick_Factor_Return_N_IC(standardized_factor_df, all_pair_open_df,
                                                      target_n_map[self.time_frame], factor_method, self.time_frame)
            print(report)
            if abs(report['IC mean']) > 0.03 and abs(report['IR']) > 0.2:
                self.factor_methods.append(factor_method)
                factors_info.append({'factor_method': factor_method, 'IC': report['IC mean'], 'IR': report['IR']})
        factors_info_df = pd.DataFrame.from_records(factors_info)
        print(factors_info_df)
        factors_info_df.to_csv(f'./factors_info_{self.time_frame}.csv')

    def calculate_single_factor(self, func, name, df: pd.DataFrame) -> pd.DataFrame:
        res = func(df)
        return res

    def get_factor_methods(self) -> list:
        return (list(filter(lambda m: m.startswith("factor") and callable(getattr(self, m)),
                            dir(self))))

    def factor_mom_4(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mom_4'] = (dataframe['close'] - dataframe['close'].shift(4)) / dataframe['close'].shift(4)
        return dataframe

    def factor_mom_12(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mom_12'] = (dataframe['close'] - dataframe['close'].shift(12)) / dataframe['close'].shift(12)
        return dataframe

    def factor_mom_24(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mom_24'] = (dataframe['close'] - dataframe['close'].shift(24)) / dataframe['close'].shift(24)
        return dataframe

    def factor_mom_72(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mom_72'] = (dataframe['close'] - dataframe['close'].shift(72)) / dataframe['close'].shift(72)
        return dataframe

    # def factor_mom_168(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    #     dataframe['factor_mom_168'] = (dataframe['close'] - dataframe['close'].shift(168)) / dataframe['close'].shift(168)
    #     return dataframe

    def factor_adx_12(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_adx_12'] = ta.ADX(dataframe, timeperiod=12)
        return dataframe

    def factor_adx_24(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_adx_24'] = ta.ADX(dataframe, timeperiod=24)
        return dataframe

    def factor_adx_72(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_adx_72'] = ta.ADX(dataframe, timeperiod=72)
        return dataframe

    def factor_rsi_12(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_rsi_12'] = ta.RSI(dataframe, timeperiod=12)
        return dataframe

    def factor_rsi_24(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_rsi_24'] = ta.RSI(dataframe, timeperiod=24)
        return dataframe

    def factor_rsi_72(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_rsi_72'] = ta.RSI(dataframe, timeperiod=72)
        return dataframe

    def factor_mfi_12(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mfi_12'] = ta.MFI(dataframe, timeperiod=12)
        return dataframe

    def factor_mfi_24(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mfi_24'] = ta.MFI(dataframe, timeperiod=24)
        return dataframe

    def factor_mfi_72(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_mfi_72'] = ta.MFI(dataframe, timeperiod=72)
        return dataframe

    def factor_obv(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_obv'] = ta.OBV(dataframe)
        return dataframe

    def factor_roc_12(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_roc_12'] = ta.ROC(dataframe, timeperiod=12)
        return dataframe

    def factor_roc_24(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_roc_24'] = ta.ROC(dataframe, timeperiod=24)
        return dataframe

    def factor_roc_72(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['factor_roc_72'] = ta.ROC(dataframe, timeperiod=72)
        return dataframe
