import pandas as pd
import talib.abstract as ta


class TAFactors:
    def __init__(self, time_frame='1h'):
        self.time_frame = time_frame
        self.factors_list = []

    def get_factor_methods(self) -> list:
        return (list(filter(lambda m: m.startswith("factor") and callable(getattr(self, m)),
                            dir(self))))

    @staticmethod
    def factor_ADX(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ADX'] = ta.ADX(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_ADXR(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ADXR'] = ta.ADX(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_APO(self, df: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0):
        df['factor_APO'] = ta.APO(df, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        return df

    @staticmethod
    def factor_AROON(self, df: pd.DataFrame, timeperiod=14):
        df['factor_AROON_DOWN'], df['factor_AROON_UP'] = ta.AROON(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_AROONOSC(self, df: pd.DataFrame, timeperiod=14):
        df['factor_AROONOSC'] = ta.AROONOSC(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_BOP(self, df: pd.DataFrame):
        df['factor_BOP'] = ta.BOP(df)
        return df

    @staticmethod
    def factor_CCI(self, df: pd.DataFrame, timeperiod=14):
        df['factor_CCI'] = ta.CCI(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_CMO(self, df: pd.DataFrame, timeperiod=14):
        df['factor_CMO'] = ta.CMO(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_DX(self, df: pd.DataFrame, timeperiod=14):
        df['factor_DX'] = ta.DX(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_MFI(self, df: pd.DataFrame, timeperiod=14):
        df['factor_MFI'] = ta.MFI(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_MINUS_DI(self, df: pd.DataFrame, timeperiod=14):
        df['factor_MINUS_DI'] = ta.MINUS_DI(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_MINUS_DM(self, df: pd.DataFrame, timeperiod=14):
        df['factor_MINUS_DM'] = ta.MINUS_DM(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_MOM(self, df: pd.DataFrame, timeperiod=14):
        df['factor_MMO'] = ta.MOM(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_PLUS_DI(self, df: pd.DataFrame, timeperiod=14):
        df['factor_PLUS_DI'] = ta.PLUS_DI(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_PLUS_DM(self, df: pd.DataFrame, timeperiod=14):
        df['factor_PLUS_DM'] = ta.PLUS_DM(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_PPO(self, df: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0):
        df['factor_PPO'] = ta.PPO(df, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        return df

    @staticmethod
    def factor_ROC(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ROC'] = ta.ROC(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_ROCP(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ROCP'] = ta.ROCP(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_ROCR(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ROCR'] = ta.ROCR(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_ROCR100(self, df: pd.DataFrame, timeperiod=14):
        df['factor_ROCR100'] = ta.ROCR100(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_RSI(self, df: pd.DataFrame, timeperiod=14):
        df['factor_RSI'] = ta.RSI(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_STOCH(self, df: pd.DataFrame, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        df['factor_STOCH_SLOWK'], df['factor_STOCK_SLOWD'] = ta.STOCH(df, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)
        return df

    @staticmethod
    def factor_STOCHF(self, df: pd.DataFrame, fastk_period=5, fastd_period=3, fastd_matype=0):
        df['factor_STOCHF_FASTK'], df['factor_STOCHF_FASTD'] = ta.STOCHF(df, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
        return df

    @staticmethod
    def factor_STOCHRSI(self, df: pd.DataFrame, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        df['factor_STOCHRSI_FASTK'], df['factor_STOCHRSI_FASTD'] = ta.STOCHRSI(df, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
        return df

    @staticmethod
    def factor_TRIX(self, df: pd.DataFrame, timeperiod=14):
        df['factor_TRIX'] = ta.TRIX(df, timeperiod=timeperiod)
        return df

    @staticmethod
    def factor_ULTOSC(self, df: pd.DataFrame, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        df['factor_ULTOSC'] = ta.ULTOSC(df, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        return df

    @staticmethod
    def factor_WILLR(self, df: pd.DataFrame, timeperiod=14):
        df['factor_WILLR'] = ta.WILLR(df, timeperiod=timeperiod)
        return df





