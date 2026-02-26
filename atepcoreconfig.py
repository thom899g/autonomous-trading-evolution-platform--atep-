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