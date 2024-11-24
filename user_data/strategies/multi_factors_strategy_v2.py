import time
from typing import Optional
import logging
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative
from datetime import datetime
import datetime
import pandas as pd
import pandas_ta as ta
logger = logging.getLogger(__name__)


def generate_function_eval_string(indicator_params_str: str) -> str:
    """
    Generate function evaluation string by pandas-ta indicator parameter string
    """
    indicator_params_list = indicator_params_str.split('_')
    first_param_name = ''

    for i, v in enumerate(indicator_params_list):
        if v.isdigit():
            first_param_name = indicator_params_list[i - 1]
            break
        elif v == 'mamode':
            first_param_name = indicator_params_list[i]
            break

    if first_param_name == '':
        return f'ta.{indicator_params_str[4:]}()'

    indicator_name = indicator_params_str[4:indicator_params_str.index(first_param_name) - 1]
    indicator_params = indicator_params_str[indicator_params_str.index(first_param_name):]

    indicator_params_list = indicator_params.split('_')
    indicator_params_str = ''

    for i, indicator_param in enumerate(indicator_params_list):
        indicator_params_str += indicator_param
        if indicator_param.isdigit():
            indicator_params_str += ','
        elif i < len(indicator_params_list) - 1 and indicator_params_list[i + 1] in ['length', 'lookback']:
            indicator_params_str += '_'
        elif indicator_param not in ['mamode', 'sma1', 'sma2', 'sma3', 'sma4'] and 'ma' in indicator_param:
            indicator_params_str += ','
        else:
            indicator_params_str += '='
    indicator_params_str = indicator_params_str[:-1]

    if 'mamode=' in indicator_params_str:
        mamode_index = indicator_params_str.index('mamode')
        mamode_value = indicator_params_str[mamode_index + 7:]
        indicator_params_str = str.replace(indicator_params_str, mamode_value, f"'{mamode_value}'")

    return f'ta.{indicator_name}({indicator_params_str})'


def mad(df, n=3 * 1.4826):
    def filter_extreme_MAD(series, n):
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n * new_median, median + n * new_median)

    df = df.apply(lambda x: filter_extreme_MAD(x, n), axis=1)
    return df


def standardize(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


class MultiFactorsStrategyV2(IStrategy):
    # Freqtrade Config
    INTERFACE_VERSION = 3
    can_short = True
    trailing_stop = False
    timeframe = '5m'
    stoploss = -1
    startup_candle_count = 80
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    # Strategy Custom Config

    # Cache all pairs candles
    pair_data_dict_5m = {}
    pair_data_dict_1h = {}

    # Long and short target symbols after combined factors sorted
    target_symbols = {}

    # IC value for factors
    factors_ic_dict = {'NATR_4': -0.0335,
 'CFO_19': -0.0501,
 'ROC_19': -0.0303,
 'CG_9': -0.0263,
 'CFO_9': -0.049,
 'DMP_5': -0.0225,
 'DMN_5': 0.0258,
 'EFI_16': -0.0455,
 'PGO_9': -0.0461,
 'BOP': -0.0449,
 'DMP_16': -0.0304,
 'DMN_16': 0.0299}

    # Factors mapping the pandas_ta indicators
    factors_pta_indicators_list = ['pta_adx_length_16_lensig_13_mamode_dema',
 'pta_adx_length_5_lensig_10_mamode_trima',
 'pta_bop',
 'pta_cfo_length_19',
 'pta_cfo_length_9',
 'pta_cg_length_9',
 'pta_efi_length_16_mamode_fwma',
 'pta_natr_length_4_mamode_dema',
 'pta_pgo_length_9',
 'pta_roc_length_19']

    def is_candles_prepared(self) -> bool:
        """
        Check all pairs 1h candles are prepared
        """
        # All pairs candle ready
        all_data_collected = all((pair in self.pair_data_dict_1h)
                                 for pair in self.dp.current_whitelist())

        # All pairs latest candle updated
        latest_pair_date_list = []
        for key in self.pair_data_dict_1h.keys():
            latest_date = self.pair_data_dict_1h[key].iloc[-1]['date']
            latest_pair_date_list.append(latest_date)

        return all_data_collected and len(set(latest_pair_date_list)) == 1

    def calculate_target_symbols(self, current_time: datetime) -> tuple[list, list]:
        """
        Calculate target long & short symbols
        """
        target_symbol_key = current_time.replace(minute=0, second=0, microsecond=0)
        if self.target_symbols.get(target_symbol_key):
            return self.target_symbols[target_symbol_key]['short'], self.target_symbols[target_symbol_key]['long']
        else:
            if not self.is_candles_prepared():
                return [], []

            # 1. Calculate factors
            factor_dict = {}
            for key, value in self.pair_data_dict_1h.items():
                candles = value.copy()
                candles.set_index(['date'], inplace=True)
                if len(factor_dict.keys()) == 0:
                    # Dynamic generate Factors by eval
                    for indicator_with_params in self.factors_pta_indicators_list:
                        eval_string = generate_function_eval_string(indicator_with_params)
                        eval_result = eval(f'candles.{eval_string}')
                        if type(eval_result) is pd.Series:
                            eval_result = eval_result.to_frame()
                        factor_names = eval_result.columns.tolist()
                        for factor_name in factor_names:
                            if factor_name in self.factors_ic_dict.keys():
                                single_factor = eval_result[factor_name].to_frame()
                                single_factor.columns = [key]
                                factor_dict[factor_name] = single_factor
                else:
                    for indicator_with_params in self.factors_pta_indicators_list:
                        eval_string = generate_function_eval_string(indicator_with_params)
                        eval_result = eval(f'candles.{eval_string}')
                        if type(eval_result) is pd.Series:
                            eval_result = eval_result.to_frame()
                        factor_names = eval_result.columns.tolist()
                        for factor_name in factor_names:
                            if factor_name in self.factors_ic_dict.keys():
                                single_factor = eval_result[factor_name].to_frame()
                                single_factor.columns = [key]
                                factor_dict[factor_name] = factor_dict[factor_name].join(single_factor)

            # 2. Standardize factors
            standardized_factors_dict = {}
            for key in factor_dict.keys():
                standardized_factors_dict[key] = standardize(factor_dict[key])

            # 3. Combine factors
            combine_factors_df = pd.DataFrame()
            for key in standardized_factors_dict.keys():
                ic = self.factors_ic_dict[key]
                # print(key)
                # print(standardized_factors_dict[key].iloc[-1])
                if combine_factors_df.shape[0] == 0:
                    combine_factors_df = standardized_factors_dict[key] * (-1 if ic < 0 else 1)
                else:
                    combine_factors_df = combine_factors_df + (standardized_factors_dict[key] * (-1 if ic < 0 else 1))

            # 4. Sort pairs by combined factor
            max_open_trades = self.config['max_open_trades'] // 2
            if self.dp.runmode in ['live', 'dry-run']:
                ordered_symbols = combine_factors_df.iloc[-1].dropna().sort_values().index.to_list()
            else:
                ordered_symbols = combine_factors_df.loc[
                    (current_time.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)).strftime(
                        '%Y-%m-%d %H:%M:%S')].dropna().sort_values().index.to_list()

            # 5. Set the long short targets into cache and return the value
            if len(ordered_symbols) == len(self.dp.current_whitelist()):
                self.target_symbols[target_symbol_key] = {
                    'long': ordered_symbols[-max_open_trades:],
                    'short': ordered_symbols[:max_open_trades]
                }
                return ordered_symbols[:max_open_trades], ordered_symbols[-max_open_trades:]
            else:
                return [], []

    def update_pair_data(self, pair: str, time_frame: str, df: pd.DataFrame) -> None:
        """
        Update cached pair candles data
        """
        if time_frame == '5m':
            self.pair_data_dict_5m[pair] = df
        elif time_frame == '1h':
            df = df.rename(columns={
                'close_1h': 'close',
                'open_1h': 'open',
                'high_1h': 'high',
                'low_1h': 'low',
                'volume_1h': 'volume'
            })
            self.pair_data_dict_1h[pair] = df

    def get_num_positions(self, pair: str, is_short: bool) -> tuple[int, int, bool]:
        """
        Check the pair has position and get total number of positions for long and short
        """
        has_pair_position = False
        open_trades = Trade.get_trades_proxy(is_open=True)
        num_shorts, num_longs = 0, 0
        for open_trade in open_trades:
            if open_trade.pair == pair and open_trade.is_short is is_short:
                has_pair_position = True
            if open_trade.is_short:
                num_shorts += 1
            else:
                num_longs += 1
        return num_shorts, num_longs, has_pair_position

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['tema'] = dataframe.ta.tema(length=12)
        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.update_pair_data(metadata['pair'], '1h', dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_symbols, long_symbols = self.calculate_target_symbols(datetime.datetime.utcnow())
        dataframe.loc[
            (
                    (metadata['pair'] in long_symbols)
                    &
                    (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                    (metadata['pair'] in short_symbols)
                    &
                    (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
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
        short_symbols, long_symbols = self.calculate_target_symbols(datetime.datetime.utcnow())
        logger.info('Short symbols: %s', short_symbols)
        logger.info('Long symbols: %s', long_symbols)
        num_shorts, num_longs, has_pair_pos = self.get_num_positions(pair, side == 'short')
        if has_pair_pos:
            return False
        if side == 'short' and (pair not in short_symbols or num_shorts >= max_open_trades // 2):
            return False
        if side == 'long' and (pair not in long_symbols or num_longs >= max_open_trades // 2):
            return False
        return True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_symbols, long_symbols = self.calculate_target_symbols(datetime.datetime.utcnow())

        dataframe.loc[
            (
                    (metadata['pair'] not in long_symbols)
                    &
                    (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
                    & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                    (metadata['pair'] not in short_symbols)
                    &
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
        short_symbols, long_symbols = self.calculate_target_symbols(datetime.datetime.utcnow())
        logger.info('Short symbols: %s', short_symbols)
        logger.info('Long symbols: %s', long_symbols)
        if not short_symbols or not long_symbols:
            return False
        if trade.is_short and pair not in short_symbols:
            return True
        if not trade.is_short and pair not in long_symbols:
            return True
        time.sleep(10)
        return False

