import logging
from typing import Optional
import shutil

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import datetime

from freqtrade.enums import CandleType
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair, informative  # noqa
from factors_generator import FactorsGenerator

logger = logging.getLogger(__name__)


class SuperMultiFactorsStrategy(IStrategy):
    pair_data_dict_5m = {}
    pair_data_dict_15m = {}
    pair_data_dict_1h = {}

    factor_generator_5m = FactorsGenerator('5m')
    factor_generator_15m = FactorsGenerator('15m')
    factor_generator_1h = FactorsGenerator('1h')

    factors_validated_time = None
    process_only_new_candles = True
    can_short = True
    startup_candle_count = 3000
    use_custom_stoploss = True
    is_validating_factors = False

    stoploss = -1
    timeframe = '5m'

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
        if self.factors_validated_time is not None:
            return

        # We don't revalidate in the live
        if self.dp.runmode != 'live':
            # Check whether data collected ready
            if not self.is_pair_data_prepared():
                return
            # Do validation again
            logger.info("Starting re-validating factors")
            self.is_validating_factors = True
            self.factor_generator_5m.validate_factors(self.pair_data_dict_5m)
            logger.info(f"5m timeframe factors {self.factor_generator_5m.factor_methods}")
            self.factor_generator_15m.validate_factors(self.pair_data_dict_15m)
            logger.info(f"15m timeframe factors {self.factor_generator_15m.factor_methods}")
            self.factor_generator_1h.validate_factors(self.pair_data_dict_1h)
            logger.info(f"1h timeframe factors {self.factor_generator_1h.factor_methods}")

            # Save the validated time and set factor_validation_ready to True
            self.factors_validated_time = current_time
            self.is_validating_factors = False
            # Set freqtrade ai model expire
            for pair in self.dp.current_whitelist():
                if self.freqai.dd.pair_dict.get(pair):
                    self.freqai.dd.pair_dict[pair] = self.freqai.dd.empty_pair_dict.copy()

            # Remove model file from disk
            model_folders = [x for x in self.freqai.dd.full_path.iterdir() if x.is_dir()]
            for model_folder in model_folders:
                shutil.rmtree(model_folder)

            logger.info("Re-validating factors succeed")
        else:
            logger.info("Running strategy in live, loading factors from csv")
            self.factor_generator_5m.load_factors_from_file()
            self.factor_generator_15m.load_factors_from_file()
            self.factor_generator_1h.load_factors_from_file()
            self.factors_validated_time = current_time

    def is_pair_data_prepared(self):
        all_data_collected = all((pair in self.pair_data_dict_15m) and (pair in self.pair_data_dict_5m) and (pair in self.pair_data_dict_1h)
                                 for pair in self.dp.current_whitelist())
        return all_data_collected

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        if self.factors_validated_time is not None and not self.is_validating_factors:
            tf = metadata["tf"]
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
        if self.factors_validated_time is not None and not self.is_validating_factors:
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
        self.freqai.class_names = ["down", "up"]
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-24) > dataframe["close"], "up", "down"
        )
        return dataframe

    @informative('15m')
    def populate_indicators_15m(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode != 'live':
            self.update_pair_data(metadata['pair'], '15m', dataframe)
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode != 'live':
            self.update_pair_data(metadata['pair'], '1h', dataframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        if self.factors_validated_time is not None and not self.is_validating_factors:
            dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe['atr_real'] = qtpylib.atr(dataframe, window=12 * 4)
        dataframe['atr'] = np.nan_to_num(dataframe['atr_real'] / ((dataframe['close'] + dataframe['open']) / 2))

        # TA indicators to combine with the Freqai targets
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=12, stds=2)
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

        if self.dp.runmode != 'live':
            self.update_pair_data(metadata['pair'], '5m', dataframe)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if self.factors_validated_time is None or self.is_validating_factors:
            return df

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                & (df["tema"] <= df["bb_middleband"])  # Guard: tema below BB middle
                & (df["tema"] > df["tema"].shift(1))  # Guard: tema is raising
                & (df["volume"] > 0)  # Make sure Volume is not 0
                & (df["do_predict"] == 1)  # Make sure Freqai is confident in the prediction
                &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df["&s-up_or_down"] == "up")
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                & (df["tema"] > df["bb_middleband"])  # Guard: tema above BB middle
                & (df["tema"] < df["tema"].shift(1))  # Guard: tema is falling
                & (df["volume"] > 0)  # Make sure Volume is not 0
                & (df["do_predict"] == 1)  # Make sure Freqai is confident in the prediction
                &
                # Only enter trade if Freqai thinks the trend is in this direction
                (df["&s-up_or_down"] == "down")
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if self.factors_validated_time is None or self.is_validating_factors:
            return df

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                & (df["tema"] > df["bb_middleband"])  # Guard: tema above BB middle
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

    # def confirm_trade_entry(
    #     self,
    #     pair: str,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     current_time: datetime,
    #     entry_tag: Optional[str],
    #     side: str,
    #     **kwargs,
    # ) -> bool:
    #     is_short = True if side == "short" else False
    #     # Max trade count in each side
    #     can_open_trades_len = int(self.config["max_open_trades"]) // 2
    #     num_shorts, num_longs, has_pair_pos = self.get_num_pos(pair, is_short)
    #     # Already has position of this pair
    #     if has_pair_pos:
    #         return False
    #
    #     if side == "long" and num_longs >= can_open_trades_len:
    #         return False
    #
    #     if side == "short" and num_shorts >= can_open_trades_len:
    #         return False
    #
    #     return True

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        pos_minutes = (current_time - trade.open_date_utc.replace(second=0, microsecond=0)).total_seconds() // 60

        atr_ratio = 3
        # if pos_minutes >= 120:
        #     if (trade.is_short and last_candle['&s-up_or_down'] == 'up') or (not trade.is_short and last_candle['&s-up_or_down'] == 'down'):
        #         atr_ratio = 2

        atr = last_candle["atr"]
        if current_profit > atr_ratio * atr:
            desired_stoploss = current_profit / 2
            return max(min(desired_stoploss, 0.01), 0.005)

        if current_profit < -atr_ratio * atr:
            return -atr


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

