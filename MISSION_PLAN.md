# Autonomous Trading Evolution Platform (ATEP)

## Objective
A self-evolving AI platform designed to autonomously research, develop, and optimize trading strategies across diverse financial markets using reinforcement learning (RL) and generative models.

## Strategy
Research and implement using available tools.

## Execution Output
## SUMMARY:
I've architected and implemented the core foundation of the Autonomous Trading Evolution Platform (ATEP), focusing on modular, production-ready components with strict error handling, logging, and realistic constraints. The system includes: 1) Firebase-powered state management, 2) Market data abstraction layer, 3) Trading environment for RL, 4) Agent base architecture, 5) Strategy evolution pipeline, and 6) Comprehensive monitoring/logging.

## OUTPUT:

### FILE: atep/core/config.py
```python
"""
ATEP Configuration Manager
Handles environment variables, Firebase initialization, and platform-wide settings.
Critical: All sensitive data loaded from .env with fallbacks for development.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import firebase_admin
from firebase_admin import credentials, firestore, db
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Singleton configuration manager with Firebase integration"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._load_config()
            self._init_firebase()
            self._initialized = True
    
    def _load_config(self) -> None:
        """Load and validate configuration from environment"""
        self.firebase_credentials = os.getenv('FIREBASE_CREDENTIALS_PATH')
        self.firebase_database_url = os.getenv('FIREBASE_DATABASE_URL')
        self.firestore_collection = os.getenv('FIRESTORE_COLLECTION', 'atep_strategies')
        
        # Trading configuration
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.2'))
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
        
        # RL configuration
        self.state_size = int(os.getenv('STATE_SIZE', '50'))
        self.action_size = int(os.getenv('ACTION_SIZE', '3'))
        self.gamma = float(os.getenv('GAMMA', '0.99'))
        
        # Validation
        if not self.firebase_credentials:
            raise ValueError("FIREBASE_CREDENTIALS_PATH must be set in .env")
        if not os.path.exists(self.firebase_credentials):
            raise FileNotFoundError(f"Firebase credentials not found at {self.firebase_credentials}")
    
    def _init_firebase(self) -> None:
        """Initialize Firebase with error handling and validation"""
        try:
            cred = credentials.Certificate(self.firebase_credentials)
            
            # Initialize with both Firestore and Realtime Database
            firebase_admin.initialize_app(
                cred,
                {
                    'databaseURL': self.firebase_database_url,
                    'projectId': 'atep-platform'  # Will be overridden by credentials
                }
            )
            
            self.firestore_client = firestore.client()
            self.rtdb = db.reference()
            self.logger.info("Firebase initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Firebase initialization failed: {str(e)}")
            # Create mock clients for development (prevent crashes)
            self.firestore_client = None
            self.rtdb = None
            self.logger.warning("Running in offline mode - no Firebase connectivity")
    
    def get_strategy_collection(self):
        """Get Firestore collection for strategies with validation"""
        if self.firestore_client is None:
            raise ConnectionError("Firebase not initialized. Check credentials and network.")
        return self.firestore_client.collection(self.firestore_collection)
    
    def get_state_path(self, agent_id: str) -> str:
        """Get Realtime Database path for agent state"""
        return f"agents/{agent_id}/state"

config = Config()
```

### FILE: atep/data/market_data.py
```python
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