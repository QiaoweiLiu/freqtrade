import logging
import time
import datetime
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair, informative  # noqa
from factors_generator import FactorsGenerator

logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    pair_data_dict_5m = {}
    pair_data_dict_15m = {}
    pair_data_dict_1h = {}

    factor_generator_5m = FactorsGenerator('5m')
    factor_generator_15m = FactorsGenerator('15m')
    factor_generator_1h = FactorsGenerator('1h')

    factors_validated_time = None

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    timeframe = '5m'

    plot_config = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Up_or_down": {
                "&s-up_or_down": {"color": "green"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 800
    can_short = True

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    def update_pair_data(self, pair, time_frame, df):
        if time_frame == '5m':
            self.pair_data_dict_5m[pair] = df
        elif time_frame == '15m':
            df = df.rename(columns={'close_15m': 'close'})
            df = df.rename(columns={'open_15m': 'open'})
            df = df.rename(columns={'high_15m': 'high'})
            df = df.rename(columns={'low_15m': 'low'})
            df = df.rename(columns={'volume_15m': 'volume'})
            self.pair_data_dict_15m[pair] = df
        elif time_frame == '1h':
            df = df.rename(columns={'close_1h': 'close'})
            df = df.rename(columns={'open_1h': 'open'})
            df = df.rename(columns={'high_1h': 'high'})
            df = df.rename(columns={'low_1h': 'low'})
            df = df.rename(columns={'volume_1h': 'volume'})
            self.pair_data_dict_1h[pair] = df

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # Check whether factor needs to re-validate
        if self.factors_validated_time is not None and current_time - datetime.timedelta(hours=24) < self.factors_validated_time:
            return

        # Check whether data collected ready
        if not self.is_pair_data_prepared():
            return
        # Do validation again
        logger.info(self.pair_data_dict_5m)
        self.factor_generator_5m.validate_factors(self.pair_data_dict_5m)
        self.factor_generator_15m.validate_factors(self.pair_data_dict_15m)
        self.factor_generator_1h.validate_factors(self.pair_data_dict_1h)

        # Save the validated time and set factor_validation_ready to True
        self.factors_validated_time = current_time

    def is_pair_data_prepared(self):
        all_data_collected = all((pair in self.pair_data_dict_15m) and (pair in self.pair_data_dict_5m) and (pair in self.pair_data_dict_1h)
                                 for pair in self.dp.current_whitelist())
        return all_data_collected

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        # dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        # dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        # dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        # dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        # dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        #
        # bollinger = qtpylib.bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=period, stds=2.2
        # )
        # dataframe["bb_lowerband-period"] = bollinger["lower"]
        # dataframe["bb_middleband-period"] = bollinger["mid"]
        # dataframe["bb_upperband-period"] = bollinger["upper"]
        #
        # dataframe["%-bb_width-period"] = (
        #     dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        # ) / dataframe["bb_middleband-period"]
        # dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]
        #
        # dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        #
        # dataframe["%-relative_volume-period"] = (
        #     dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        # )
        if self.factors_validated_time is not None or True:
            tf = metadata["tf"]
            logger.info(tf)
            if tf == '5m':
                dataframe = self.factor_generator_5m.generate_factors(dataframe)
            elif tf == '15m':
                dataframe = self.factor_generator_15m.generate_factors(dataframe)
            elif tf == '1h':
                dataframe = self.factor_generator_1h.generate_factors(dataframe)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        # dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        # dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        self.freqai.class_names = ["down", "up"]
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-50) > dataframe["close"], "up", "down"
        )

        return dataframe

    @informative('15m')
    def populate_indicators_15m(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self.update_pair_data(metadata['pair'], '15m', dataframe)
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self.update_pair_data(metadata['pair'], '1h', dataframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA indicators to combine with the Freqai targets
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Bollinger Bands
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

        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        self.update_pair_data(metadata['pair'], '5m', dataframe)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                &
                (df["tema"] <= df["bb_middleband"])  # Guard: tema below BB middle
                & (df["tema"] > df["tema"].shift(1))  # Guard: tema is raising
                & (df["volume"] > 0)  # Make sure Volume is not 0
                & (df["do_predict"] == 1)  # Make sure Freqai is confident in the prediction
                &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df["&s-up_or_down"] == "up")
                # &
                # (self.factors_validated_time is not None)
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                &
                (df["tema"] > df["bb_middleband"])  # Guard: tema above BB middle
                & (df["tema"] < df["tema"].shift(1))  # Guard: tema is falling
                & (df["volume"] > 0)  # Make sure Volume is not 0
                & (df["do_predict"] == 1)  # Make sure Freqai is confident in the prediction
                &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df["&s-up_or_down"] == "down")
                # &
                # (self.factors_validated_time is not None)
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                &
                (df["tema"] > df["bb_middleband"])  # Guard: tema above BB middle
                & (df["tema"] < df["tema"].shift(1))  # Guard: tema is falling
                & (df["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df["rsi"], self.exit_short_rsi.value))
                &
                # Guard: tema below BB middle
                (df["tema"] <= df["bb_middleband"])
                & (df["tema"] > df["tema"].shift(1))  # Guard: tema is raising
                & (df["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1

        return df
