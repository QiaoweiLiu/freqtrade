import logging
from datetime import datetime
import datetime
from typing import Optional
import pandas as pd
from pandas import DataFrame
import time

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    informative
)
import talib.abstract as ta
from technical import qtpylib

logger = logging.getLogger(__name__)


def factor_open(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = dataframe['open']
    return dataframe

def factor_rocr(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ROCR(dataframe, timeperiod=52)
    return dataframe

def factor_rocr100(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ROCR100(dataframe, timeperiod=26)
    return dataframe

def factor_roc(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ROC(dataframe, timeperiod=27)
    return dataframe

def factor_trix(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.TRIX(dataframe, timeperiod=42)
    return dataframe

def factor_trix1(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.TRIX(dataframe, timeperiod=13)
    return dataframe

def factor_cci(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.CCI(dataframe, timeperiod=20)
    return dataframe

def factor_rsi(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.RSI(dataframe, timeperiod=49)
    return dataframe

def factor_adx(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ADX(dataframe, timeperiod=34)
    return dataframe

def factor_dx(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.DX(dataframe, timeperiod=48)
    return dataframe

def factor_adxr(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ADXR(dataframe, timeperiod=34)
    return dataframe

def factor_ppo(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.PPO(dataframe, fastperiod=42, slowperiod=35)
    return dataframe

def factor_bop(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.BOP(dataframe)
    return dataframe

def factor_minus_di(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.MINUS_DI(dataframe, timeperiod=40)
    return dataframe

def factor_ultosc(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ULTOSC(dataframe, timeperiod1=59, timeperiod2=51, timeperiod=58)
    return dataframe

def factor_ultosc1(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ULTOSC(dataframe, timeperiod1=28, timeperiod2=6, timeperiod=27)
    return dataframe

def factor_mfi(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.MFI(dataframe, timeperiod=38)
    return dataframe

def factor_plus_di(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.PLUS_DI(dataframe, timeperiod=56)
    return dataframe

def factor_plus_dm(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.PLUS_DM(dataframe, timeperiod=58)
    return dataframe

def factor_willr(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.WILLR(dataframe, timeperiod=28)
    return dataframe

def factor_aroonosc(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.AROONOSC(dataframe, timeperiod=26)
    return dataframe

def factor_aroondown(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name], _ = ta.AROON(dataframe, timeperiod=51)
    return dataframe

def factor_aroonup(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    _, dataframe[pair_name] = ta.AROON(dataframe, timeperiod=51)
    return dataframe

def factor_natr(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.NATR(dataframe, timeperiod=12)
    return dataframe

def factor_bop(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.BOP(dataframe)
    return dataframe

def factor_adosc(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ADOSC(dataframe, fastperiod=7, slowperiod=6)
    return dataframe

def factor_adosc1(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.ADOSC(dataframe, fastperiod=35, slowperiod=32)
    return dataframe

def factor_ad(dataframe: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dataframe[pair_name] = ta.AD(dataframe)
    return dataframe

def mad(df, n=3 * 1.4826):
    def filter_extreme_MAD(series, n):
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n * new_median, median + n * new_median)

    df = df.apply(lambda x: filter_extreme_MAD(x, n), axis=1)
    return df


def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


class MultiFactorsStrategyV1(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    trailing_stop = False
    timeframe = '5m'
    stoploss = -1

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    pair_data_dict_5m = {}
    pair_data_dict_1h = {}
    factors_df_dict = {}
    target_symbols = {}
    factors_ic_dict = {
        'NATR': -0.0037,
        'ROCR100': 0.0379,
        'TRIX': -0.0975,
        'PPO': 0.0106,
        'DX': 0.0079,
        'ADX': 0.0482,
        'MINUS_DI': -0.0095,
        'ADXR': 0.0677,
        'ULTOSC': -0.0701,
        'ULTOSC1': -0.0306,
        'MFI': -0.0137,
        'PLUS_DI': -0.0121,
        'BOP': -0.0318,
        'PLUS_DM': -0.002,
        'AROONOSC': 0.0023,
        'WILLR': -0.0121,
        'ADOSC': 0.0233,
        'ADOSC1': 0.0632,
        'AD': -0.063
        }

    def is_candles_prepared(self):
        all_data_collected = all((pair in self.pair_data_dict_1h)
                                 for pair in self.dp.current_whitelist())
        return all_data_collected

    def calculate_target_symbols(self, current_time: datetime):
        target_symbols_key = current_time.replace(minute=0, second=0, microsecond=0)
        if self.target_symbols.get(target_symbols_key):
            return self.target_symbols[target_symbols_key]['short'], self.target_symbols[target_symbols_key]['long']
        else:
            if not self.is_candles_prepared():
                return [], []
            factors_df_dict = {}
            for key, value in self.pair_data_dict_1h.items():
                candles = value.copy()
                candles.set_index(['date'], inplace=True)
                if len(factors_df_dict.keys()) == 0:
                    # Hard code parameter first for research
                    # factors_df_dict['NATR'] = factor_natr(candles, key)[key].to_frame()
                    factors_df_dict['ROCR100'] = factor_rocr100(candles, key)[key].to_frame()
                    factors_df_dict['TRIX'] = factor_trix(candles, key)[key].to_frame()
                    # factors_df_dict['PPO'] = factor_ppo(candles, key)[key].to_frame()
                    # factors_df_dict['DX'] = factor_dx(candles, key)[key].to_frame()
                    factors_df_dict['ADX'] = factor_adx(candles, key)[key].to_frame()
                    # factors_df_dict['MINUS_DI'] = factor_minus_di(candles, key)[key].to_frame()
                    factors_df_dict['ADXR'] = factor_adxr(candles, key)[key].to_frame()
                    factors_df_dict['ULTOSC'] = factor_ultosc(candles, key)[key].to_frame()
                    factors_df_dict['ULTOSC1'] = factor_ultosc1(candles, key)[key].to_frame()
                    # factors_df_dict['MFI'] = factor_mfi(candles, key)[key].to_frame()
                    # factors_df_dict['PLUS_DI'] = factor_plus_di(candles, key)[key].to_frame()
                    factors_df_dict['BOP'] = factor_bop(candles, key)[key].to_frame()
                    # factors_df_dict['PLUS_DM'] = factor_plus_dm(candles, key)[key].to_frame()
                    # factors_df_dict['AROONOSC'] = factor_aroonosc(candles, key)[key].to_frame()
                    # factors_df_dict['WILLR'] = factor_willr(candles, key)[key].to_frame()
                    factors_df_dict['ADOSC'] = factor_adosc(candles, key)[key].to_frame()
                    factors_df_dict['ADOSC1'] = factor_adosc1(candles, key)[key].to_frame()
                    factors_df_dict['AD'] = factor_ad(candles, key)[key].to_frame()
                else:
                    # factors_df_dict['NATR'] = factors_df_dict['NATR'].join(factor_natr(candles, key)[key].to_frame())
                    factors_df_dict['ROCR100'] = factors_df_dict['ROCR100'].join(
                        factor_rocr100(candles, key)[key].to_frame())
                    factors_df_dict['TRIX'] = factors_df_dict['TRIX'].join(factor_trix(candles, key)[key].to_frame())
                    # factors_df_dict['DX'] = factors_df_dict['DX'].join(factor_dx(candles, key)[key].to_frame())
                    # factors_df_dict['PPO'] = factors_df_dict['PPO'].join(factor_ppo(candles, key)[key].to_frame())
                    factors_df_dict['ADX'] = factors_df_dict['ADX'].join(factor_adx(candles, key)[key].to_frame())
                    # factors_df_dict['MINUS_DI'] = factors_df_dict['MINUS_DI'].join(factor_minus_di(candles, key)[key].to_frame())
                    factors_df_dict['ADXR'] = factors_df_dict['ADXR'].join(factor_adxr(candles, key)[key].to_frame())
                    factors_df_dict['ULTOSC'] = factors_df_dict['ULTOSC'].join(factor_ultosc(candles, key)[key].to_frame())
                    factors_df_dict['ULTOSC1'] = factors_df_dict['ULTOSC1'].join(factor_ultosc1(candles, key)[key].to_frame())
                    # factors_df_dict['MFI'] = factors_df_dict['MFI'].join(factor_mfi(candles, key)[key].to_frame())
                    # factors_df_dict['PLUS_DI'] = factors_df_dict['PLUS_DI'].join(factor_plus_di(candles, key)[key].to_frame())
                    factors_df_dict['BOP'] = factors_df_dict['BOP'].join(factor_bop(candles, key)[key].to_frame())
                    # factors_df_dict['PLUS_DM'] = factors_df_dict['PLUS_DM'].join(
                    #     factor_plus_dm(candles, key)[key].to_frame())
                    # factors_df_dict['AROONOSC'] = factors_df_dict['AROONOSC'].join(
                    #     factor_aroonosc(candles, key)[key].to_frame())
                    # factors_df_dict['WILLR'] = factors_df_dict['WILLR'].join(factor_willr(candles, key)[key].to_frame())
                    factors_df_dict['ADOSC'] = factors_df_dict['ADOSC'].join(factor_adosc(candles, key)[key].to_frame())
                    factors_df_dict['ADOSC1'] = factors_df_dict['ADOSC1'].join(factor_adosc1(candles, key)[key].to_frame())
                    factors_df_dict['AD'] = factors_df_dict['AD'].join(factor_ad(candles, key)[key].to_frame())
            standardized_factors_df_dict = {}
            for key in factors_df_dict.keys():
                standardized_factors_df_dict[key] = standardize(factors_df_dict[key])

            # Combine Factors
            combine_factor_df = pd.DataFrame()
            for key in standardized_factors_df_dict.keys():
                ic = self.factors_ic_dict[key]
                if combine_factor_df.shape[0] == 0:
                    combine_factor_df = standardized_factors_df_dict[key] * (-1 if ic < 0 else 1)
                else:
                    combine_factor_df = combine_factor_df + (standardized_factors_df_dict[key] * (-1 if ic < 0 else 1))

            max_open_trades = self.config['max_open_trades']
            if self.dp.runmode == 'live':
                ordered_symbols = combine_factor_df.iloc[-1].sort_values().index.to_list()
            else:
                ordered_symbols = combine_factor_df.loc[(current_time.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')].sort_values().index.to_list()
            self.target_symbols[target_symbols_key] = {
                'long': ordered_symbols[-max_open_trades:],
                'short': ordered_symbols[:max_open_trades]
            }
            return ordered_symbols[:max_open_trades], ordered_symbols[-max_open_trades:]

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.update_pair_data(metadata['pair'], '1h', dataframe)
        return dataframe

    def update_pair_data(self, pair, time_frame, df):
        if time_frame == '5m':
            self.pair_data_dict_5m[pair] = df
        elif time_frame == '1h':
            df = df.rename(columns={'close_1h': 'close'})
            df = df.rename(columns={'open_1h': 'open'})
            df = df.rename(columns={'high_1h': 'high'})
            df = df.rename(columns={'low_1h': 'low'})
            df = df.rename(columns={'volume_1h': 'volume'})
            self.pair_data_dict_1h[pair] = df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
                dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        dataframe["rsi"] = ta.RSI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                    # (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value))
                    (dataframe["tema"] <= dataframe["bb_middleband"])  # Guard: tema below BB middle
                    & (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                    # (qtpylib.crossed_below(dataframe["rsi"], self.short_rsi.value))
                    (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
                    & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_short",
        ] = 1

        return dataframe

    def confirm_trade_entry(
            self,
            pair: str,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            current_time: datetime,
            entry_tag: Optional[str],
            side: str,
            **kwargs,
    ) -> bool:
        max_open_trades = self.config['max_open_trades']
        short_symbols, long_symbols = self.calculate_target_symbols(current_time)
        logger.info('Short symbols: %s', short_symbols)
        logger.info('Long symbols: %s', long_symbols)
        num_shorts, num_longs, has_pair_pos = self.get_num_pos(pair, side == 'short')
        if num_longs >= max_open_trades // 2 or num_shorts >= max_open_trades // 2 or has_pair_pos:
            return False

        if side == 'short' and pair not in short_symbols:
            return False
        if side == 'long' and pair not in long_symbols:
            return False
        return True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                #     (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
                #     & (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
                    (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                #     (qtpylib.crossed_below(dataframe["rsi"], self.exit_short_rsi.value))
                #     &
                #     # Guard: tema below BB middle
                #     (dataframe["tema"] <= dataframe["bb_middleband"])
                    (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1

        return dataframe

    def confirm_trade_exit(
            self,
            pair: str,
            trade: Trade,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            exit_reason: str,
            current_time: datetime,
            **kwargs,
    ) -> bool:
        pos_minutes = (current_time - trade.open_date_utc.replace(minute=0, second=0,
                                                                  microsecond=0)).total_seconds() // 60
        if pos_minutes < 60 * 4:
            time.sleep(10)
            return False
        short_symbols, long_symbols = self.calculate_target_symbols(current_time)
        if trade.is_short and pair not in short_symbols:
            return True
        if not trade.is_short and pair not in long_symbols:
            return True
        time.sleep(10)
        return False

    def get_num_pos(self, pair, is_short):
        has_pair_pos = False
        # 当前持仓
        open_trades = Trade.get_trades_proxy(is_open=True)
        # 计算 各个方向已经开单的币的数量
        num_shorts, num_longs = 0, 0
        for open_trade in open_trades:
            if open_trade.pair == pair and open_trade.is_short is is_short:
                has_pair_pos = True
            # 记录各个方向已开单币数量
            if open_trade.is_short:
                num_shorts += 1
            else:
                num_longs += 1
        return num_shorts, num_longs, has_pair_pos
