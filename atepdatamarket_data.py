"""
Market Data Abstraction Layer
Provides unified interface for multiple data sources with robust error handling.
Supports fallback mechanisms and data validation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import ccxt
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

class DataSource(Enum):
    """Supported data sources"""
    YFINANCE = "yfinance"
    CCXT = "ccxt"
    ALPACA = "alpaca"  # Placeholder for future integration

@dataclass
class MarketDataRequest:
    """Structured data request with validation"""
    symbol: str
    source: DataSource
    interval: str
    start_date: datetime
    end_date: Optional[datetime] = None
    limit: Optional[int] = 1000
    
    def validate(self) -> bool:
        """Validate request parameters"""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.start_date > datetime.now():
            raise ValueError("Start date cannot be in the future")
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        if self.limit and self.limit > 10000:
            raise ValueError("Limit cannot exceed 10,000 records")
        return True

class MarketDataProvider:
    """Unified market data provider with fallback logic"""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange_cache = {}
        
        # Initialize data sources
        self._init_ccxt()
    
    def _init_ccxt(self) -> None:
        """Initialize CCXT exchanges with error handling"""
        try:
            self.binance = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000
            })
            self.exchange_cache['binance'] = self.binance
        except Exception as e:
            self.logger.warning(f"CCXT Binance initialization failed: {e}")
            self.binance = None
    
    def fetch_data(self, request: MarketDataRequest) -> pd.DataFrame:
        """
        Fetch market data with automatic retry and fallback logic
        Returns validated DataFrame with OHLCV data
        """
        request.validate()
        
        try:
            if request.source == DataSource.YFINANCE:
                return self._fetch_yfinance(request)
            elif request.source == DataSource.CCXT:
                return self._fetch_ccxt(request)
            else:
                raise ValueError(f"Unsupported data source: {request.source}")
        except Exception as primary_error:
            self.logger.error(f"Primary data fetch failed: {primary_error}")
            # Attempt fallback to alternative source
            return self._fetch_fallback(request, primary_error)
    
    def _fetch_yfinance(self, request: MarketDataRequest) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(request.symbol)
            df = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=request.interval
            )
            
            # Validate and