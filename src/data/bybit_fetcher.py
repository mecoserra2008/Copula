"""Bybit futures data fetcher with batched historical data retrieval."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import pandas as pd
from pybit.unified_trading import HTTP

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.validators import DataValidator, validate_or_raise

logger = get_logger(__name__)


class BybitDataFetcher:
    """Fetches historical OHLCV futures data from Bybit with batched retrieval."""

    # Bybit limits
    MAX_LIMIT = 1000  # Maximum candles per request
    RATE_LIMIT_DELAY = 0.12  # Seconds between requests (to stay under rate limits)

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        """
        Initialize Bybit futures data fetcher.

        Args:
            api_key: Bybit API key (optional for public data)
            api_secret: Bybit API secret (optional for public data)
            testnet: Whether to use testnet
        """
        config = get_config()

        self.api_key = api_key or config.bybit_api_key
        self.api_secret = api_secret or config.bybit_api_secret
        self.testnet = testnet or config.bybit_testnet

        # Initialize Bybit client
        self.client = HTTP(
            testnet=self.testnet,
            api_key=self.api_key if self.api_key else None,
            api_secret=self.api_secret if self.api_secret else None,
        )

        self.config = config
        logger.info(f"Initialized BybitDataFetcher (testnet={self.testnet}, category=linear)")

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "15",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick futures data from Bybit with batched retrieval.

        Args:
            symbol: Futures trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum number of candles per request (max 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If symbol is invalid or API call fails
        """
        # Validate symbol
        validate_or_raise(DataValidator.validate_symbol, symbol)

        # Set default times
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=self.config.default_days)

        logger.info(
            f"Fetching {symbol} futures klines: interval={interval}min, "
            f"from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
        )

        # Convert to milliseconds timestamp
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        all_data = []
        current_end = end_ms
        batch_count = 0

        # Calculate expected number of candles
        interval_minutes = self._parse_interval_to_minutes(interval)
        expected_candles = int((end_ms - start_ms) / (interval_minutes * 60 * 1000))

        logger.info(f"Expected ~{expected_candles} candles (will fetch in batches of {self.MAX_LIMIT})")

        # Fetch in batches going backwards from end_time
        while current_end > start_ms:
            try:
                batch_count += 1

                # Fetch batch
                response = self.client.get_kline(
                    category="linear",  # Futures
                    symbol=symbol,
                    interval=interval,
                    start=start_ms,
                    end=current_end,
                    limit=self.MAX_LIMIT,
                )

                if response["retCode"] != 0:
                    raise ValueError(
                        f"Bybit API error: {response['retMsg']}"
                    )

                klines = response["result"]["list"]

                if not klines:
                    logger.debug(f"No more data available before {datetime.fromtimestamp(current_end/1000)}")
                    break

                # Bybit returns data in reverse chronological order
                all_data.extend(klines)

                # Get the earliest timestamp from this batch
                earliest_timestamp = int(klines[-1][0])

                # Move the end point to before the earliest candle we just fetched
                current_end = earliest_timestamp - 1

                logger.debug(
                    f"Batch {batch_count}: Fetched {len(klines)} candles, "
                    f"total: {len(all_data)}, "
                    f"earliest: {datetime.fromtimestamp(earliest_timestamp/1000)}"
                )

                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)

                # Stop if we've gone before start_time
                if earliest_timestamp <= start_ms:
                    break

            except Exception as e:
                logger.error(f"Error fetching klines batch {batch_count}: {e}")
                raise

        if not all_data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )

        # Convert types
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["turnover"] = pd.to_numeric(df["turnover"])

        # Sort by timestamp (ascending) and remove duplicates
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Filter to requested time range
        df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]

        # Select and reorder columns
        df = df[["datetime", "timestamp", "open", "high", "low", "close", "volume", "turnover"]]

        logger.info(
            f"✓ Fetched {len(df)} candles for {symbol} in {batch_count} batches "
            f"({df['datetime'].min()} to {df['datetime'].max()})"
        )

        return df

    def _parse_interval_to_minutes(self, interval: str) -> int:
        """
        Parse interval string to minutes.

        Args:
            interval: Interval string (e.g., '15', 'D', 'W')

        Returns:
            Interval in minutes
        """
        if interval == "D":
            return 1440  # 24 hours
        elif interval == "W":
            return 10080  # 7 days
        elif interval == "M":
            return 43200  # 30 days (approximate)
        else:
            return int(interval)

    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "15",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading symbols
            interval: Kline interval in minutes
            start_time: Start datetime
            end_time: End datetime

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info(f"Fetching futures data for {len(symbols)} symbols")

        results = {}
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"\n[{i}/{len(symbols)}] Fetching {symbol}...")
                df = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                )
                results[symbol] = df
                logger.info(f"✓ {symbol}: {len(df)} candles")
            except Exception as e:
                logger.error(f"✗ Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(f"\n✓ Successfully fetched {successful}/{len(symbols)} symbols")

        return results

    def fetch_and_cache(
        self,
        symbol: str,
        interval: str = "15",
        days: int = 30,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch futures data with file system caching.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            days: Number of days to fetch
            force_refresh: Force refresh cached data

        Returns:
            DataFrame with OHLCV data
        """
        if not self.config.cache_enabled:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            return self.fetch_klines(symbol, interval, start_time, end_time)

        # Create cache directory
        cache_dir = self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path (include 'futures' to distinguish from spot)
        cache_file = cache_dir / f"futures_{symbol}_{interval}m_{days}d.parquet"

        # Check cache
        if cache_file.exists() and not force_refresh:
            # Check if cache is recent (within last hour)
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 3600:  # 1 hour
                logger.info(f"Loading {symbol} from cache (age: {cache_age:.0f}s)")
                return pd.read_parquet(cache_file)

        # Fetch fresh data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        df = self.fetch_klines(symbol, interval, start_time, end_time)

        # Save to cache
        if not df.empty:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {symbol} data to {cache_file}")

        return df

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get futures symbol information.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol info
        """
        try:
            response = self.client.get_instruments_info(
                category="linear",
                symbol=symbol,
            )

            if response["retCode"] != 0:
                raise ValueError(f"Bybit API error: {response['retMsg']}")

            return response["result"]["list"][0]

        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available futures symbols.

        Returns:
            List of available symbols
        """
        try:
            response = self.client.get_instruments_info(
                category="linear",
            )

            if response["retCode"] != 0:
                raise ValueError(f"Bybit API error: {response['retMsg']}")

            symbols = [item["symbol"] for item in response["result"]["list"]]
            logger.info(f"Found {len(symbols)} available futures symbols")

            return symbols

        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            raise
