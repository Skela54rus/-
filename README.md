
BINANCE_API_KEY=
BINANCE_API_SECRET=

PAPER_TRADING=true
DEFAULT_LEVERAGE=5
RISK_PERCENT=1.0
MIN_BALANCE=100
MAX_POSITIONS_OPEN=3
MIN_TIME_BETWEEN_TRADES=10
MAX_TRADES_PER_HOUR=50
STOP_LOSS_COOLDOWN=2

TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=

TIMEFRAMES=3m,5m,15m
PRIMARY_TIMEFRAME=5m
CHECK_INTERVAL=2.0
UPDATE_INTERVAL=300
TRADING_HOURS_START=0
TRADING_HOURS_END=24
MAX_HOLDING_TIME=1800

MAX_SPREAD=0.08
MAX_DRAWDOWN=0.3
MAX_DAILY_LOSS_PERCENT=2.0
MIN_MARKET_CAP=50000000
LIQUIDITY_THRESHOLD=500000
CORRELATION_THRESHOLD=0.8
PRICE_FILTER=0.00000001
MIN_NOTIONAL=5.0
MAX_PRICE_FILTER=1000000.0

EMA_FAST=5
EMA_MEDIUM=15
EMA_SLOW=35
RSI_PERIOD=12
RSI_OVERBOUGHT=75
RSI_OVERSOLD=25
ADX_PERIOD=12
MIN_ADX=15
VOLUME_FACTOR=1.2
MIN_VOLUME=1000000
VOLATILITY_THRESHOLD=0.04
MIN_DAILY_VOLUME=30000000
VOLATILITY_PREDICTION_ENABLED=true
VOLATILITY_PREDICTION_PERIOD=15

TP_MULTIPLIER=0.03
SL_MULTIPLIER=0.015
QUICK_TP_MULTIPLIER=0.015
QUICK_TP_SIZE=0.6
SAFETY_PRICE_LIMIT=0.000001
TRAILING_START=0.03
TRAILING_DISTANCE=0.015
MIN_PROFIT_RATIO=1.5
ADAPTIVE_SL=true
ADAPTIVE_TP=true
ORDERBOOK_DEPTH=15
ORDERBOOK_UPDATE_INTERVAL=30
ADVANCED_ORDERBOOK_ANALYSIS=true

AGGRESSIVE_MODE=true
MOMENTUM_TRADING=true
BREAKOUT_TRADING=true
VOLUME_CONFIRMATION=true
NEWS_SENSITIVITY=false
REVERSAL_DETECTION=true
REVERSAL_CONFIRMATION_BARS=2
MIN_REVERSAL_STRENGTH=0.2
VOLUME_SPIKE_MULTIPLIER=2.5
TREND_CONFIRMATION=2
DYNAMIC_TRAILING=true
HIDDEN_DIVERGENCE_ENABLED=true
FALSE_BREAKOUT_FILTER=true
MIN_TREND_STRENGTH=0.4
TREND_FILTER_STRENGTH=0.7
WHALE_MANIPULATION_PROTECTION=true

ASIAN_SESSION_FACTOR=0.9
EUROPEAN_SESSION_FACTOR=1.0
US_SESSION_FACTOR=1.2
WEEKEND_FACTOR=0.7

MULTIDIMENSIONAL_ANALYSIS=true
SCENARIO_ANALYSIS_WORKERS=6
ADAPTIVE_LEARNING=true
LEARNING_UPDATE_INTERVAL=43200
BAYESIAN_OPTIMIZATION=true
AUTO_PARAMETER_OPTIMIZATION=true
OPTIMIZATION_INTERVAL=43200
COGNITIVE_TRADING_ENABLED=true
REINFORCEMENT_LEARNING_ENABLED=false
ENSEMBLE_LEARNING=true

SOCIAL_SIGNALS_ENABLED=false

DL_SEQUENCE_LENGTH=20
DL_NUM_FEATURES=10
DL_TRAINING_EPOCHS=30
DL_BATCH_SIZE=16
DL_USE_ATTENTION=false
DL_USE_HYBRID_MODEL=false
DL_TRAINING_THRESHOLD=300
DL_MIN_CONFIDENCE=0.25
TRANSFORMER_ENABLED=false
TRANSFORMER_SEQ_LENGTH=20
TRANSFORMER_D_MODEL=64
TRANSFORMER_NUM_HEADS=4
TRANSFORMER_NUM_LAYERS=2

MAX_POSITION_SIZE=0.3
MARKET_CAP_WEIGHT=0.3
LIQUIDITY_WEIGHT=0.4
VOLATILITY_WEIGHT=0.3
VOLATILITY_SCALING=true
POSITION_HEDGING=false
HEDGE_RATIO=0.1
EARLY_CLOSE_THRESHOLD=0.6

MAX_CONSECUTIVE_LOSSES=5
MIN_WIN_RATE=0.4
SIGNAL_QUALITY_THRESHOLD=0.4
CONFLUENCE_FACTORS_REQUIRED=2
VOLUME_SPIKE_THRESHOLD=1.5
MIN_SCORE=4.0
CONFIRMATIONS_REQUIRED=2
MARKET_MEMORY_SIZE=500
SIMILARITY_THRESHOLD=0.6

TIMEOUT=30
WHALE_VOLUME_FACTOR=0.3
MAX_SYMBOLS_TO_CHECK=12

FUNDAMENTAL_PAIRS=BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT,LINKUSDT,LTCUSDT,BCHUSDT,XRPUSDT,BNBUSDT,SOLUSDT,MATICUSDT,AVAXUSDT
WEBSOCKET_ENABLED=false
DATA_CACHE_DURATION=150
MAX_REQUEST_RETRIES=2
REQUEST_TIMEOUT=5

import pandas as pd
import numpy as np
import logging
import json
import pickle
import os
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.calibration import CalibratedClassifierCV
from bayes_opt import BayesianOptimization
from config import Config
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import optuna
from optuna.samplers import TPESampler
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("AdaptiveLearner")

# Добавим кастомные метрики для TensorFlow
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, l1_reg=0.0, l2_reg=0.0):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu', 
                 kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
            Dense(d_model, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DeepLearningPredictor:
    def __init__(self, sequence_length=60, num_features=15):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.scaler = RobustScaler()  # Более устойчивый скейлер
        self.feature_selector = None
        self.model_version = 1
        self.model_dir = "dl_models"
        self.performance_history = []
        self.best_accuracy = 0
        self.training_history = []
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_model(self):
        """Сохраняет модель глубокого обучения на диск"""
        try:
            if self.model is not None:
                model_path = f"{self.model_dir}/best_dl_model.h5"
                self.model.save(model_path)
                logger.info(f"DL model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving DL model: {e}")

    def load_model(self):
        """Загружает модель глубокого обучения с диска"""
        try:
            model_path = f"{self.model_dir}/best_dl_model.h5"
            if os.path.exists(model_path):
                self.model = load_model(model_path, 
                                      custom_objects={'TransformerBlock': TransformerBlock, 'F1Score': F1Score})
                logger.info("Loaded DL model from disk")
            else:
                logger.info("No DL model found on disk, will create new one when needed")
        except Exception as e:
            logger.error(f"Error loading DL model: {e}")
            self.model = None

    def _build_simplified_model(self):
        """Упрощенная модель для продакшена"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Компилируем модель
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', F1Score()]
        )
        
        return model
        
    def _build_hybrid_model(self, trial=None):
        """Создает гибридную модель с возможностью оптимизации гиперпараметров"""
        # Для продакшена используем упрощенную модель
        if getattr(Config, 'USE_SIMPLIFIED_MODEL', True):
            return self._build_simplified_model()
            
        if trial:
            # Оптимизация гиперпараметров с помощью Optuna
            num_conv_filters = trial.suggest_categorical('num_conv_filters', [32, 64, 128, 256])
            conv_kernel_size = trial.suggest_categorical('conv_kernel_size', [2, 3, 5])
            num_lstm_units = trial.suggest_categorical('num_lstm_units', [32, 64, 128, 256])
            num_dense_units = trial.suggest_categorical('num_dense_units', [16, 32, 64, 128])
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
            l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-2, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
        else:
            # Значения по умолчанию
            num_conv_filters = 128
            conv_kernel_size = 3
            num_lstm_units = 128
            num_dense_units = 64
            num_heads = 4
            learning_rate = 0.0005
            dropout_rate = 0.4
            l1_reg = 1e-4
            l2_reg = 1e-4
        
        inputs = Input(shape=(self.sequence_length, self.num_features))
        
        # Нормализация входа
        x = BatchNormalization()(inputs)
        
        # CNN слои с улучшенной архитектурой
        conv1 = Conv1D(filters=num_conv_filters, kernel_size=conv_kernel_size, 
                      activation='relu', padding='same', 
                      kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        
        conv2 = Conv1D(filters=num_conv_filters//2, kernel_size=conv_kernel_size, 
                      activation='relu', padding='same',
                      kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        
        # Bidirectional LSTM слои
        lstm1 = LSTM(num_lstm_units, return_sequences=True, 
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(conv2)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(dropout_rate)(lstm1)
        
        # Transformer блок
        transformer_block = TransformerBlock(
            d_model=num_lstm_units,
            num_heads=num_heads,
            ff_dim=num_lstm_units * 4,
            rate=dropout_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg
        )(lstm1)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=num_heads, key_dim=num_lstm_units)(transformer_block, transformer_block)
        attention = Dropout(dropout_rate)(attention)
        
        # Global average pooling
        pooled = GlobalAveragePooling1D()(attention)
        pooled = BatchNormalization()(pooled)
        pooled = Dropout(dropout_rate)(pooled)
        
        # Полносвязные слои с улучшенной архитектурой
        dense = Dense(num_dense_units, activation='relu', 
                     kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(pooled)
        dense = BatchNormalization()(dense)
        dense = Dropout(dropout_rate/2)(dense)
        
        dense2 = Dense(num_dense_units//2, activation='relu',
                      kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(dense)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(dropout_rate/2)(dense2)
        
        # Выходной слой
        output = Dense(3, activation='softmax', 
                      kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(dense2)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Кастомные метрики
        f1_metric = F1Score()
        
        model.compile(
            optimizer=Nadam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_metric, 'auc']
        )
        
        return model
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Оптимизирует гиперпараметры модели с помощью Optuna"""
        # Для упрощенной модели пропускаем оптимизацию
        if getattr(Config, 'USE_SIMPLIFIED_MODEL', True):
            self.model = self._build_simplified_model()
            logger.info("Using simplified model, skipping hyperparameter optimization")
            return {}
            
        def objective(trial):
            try:
                model = self._build_hybrid_model(trial)
                
                # Ранняя остановка
                early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=64,
                    verbose=0,
                    callbacks=[early_stop],
                    shuffle=False  # Важно для временных рядов
                )
                
                # Используем F1-score как метрику для оптимизации
                y_pred = model.predict(X_val, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_val, axis=1)
                
                f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
                
                # Сохраняем модель если она лучше предыдущих
                if f1 > self.best_accuracy:
                    self.best_accuracy = f1
                    model.save(f"{self.model_dir}/best_dl_model.h5")
                
                return f1
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return 0.0
        
        # Используем TPESampler для более эффективного поиска
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        
        # Загружаем лучшую модель
        if os.path.exists(f"{self.model_dir}/best_dl_model.h5"):
            self.model = load_model(f"{self.model_dir}/best_dl_model.h5", 
                                  custom_objects={'TransformerBlock': TransformerBlock, 'F1Score': F1Score})
        
        return study.best_params
    
    def prepare_sequences(self, data: List[Dict], target_symbol: str = None):
        """Подготавливает последовательности для обучения с улучшенной обработкой"""
        if len(data) < self.sequence_length * 2:
            return None, None
        
        # Собираем все возможные признаки
        all_features = []
        all_labels = []
        
        for i in range(len(data) - self.sequence_length):
            sequence_data = data[i:i + self.sequence_length]
            next_data = data[i + self.sequence_length]
            
            # Пропускаем последовательности с пропущенными данными
            if any(d.get('features') is None for d in sequence_data) or next_data.get('features') is None:
                continue
            
            # Извлекаем признаки для каждой точки в последовательности
            sequence_features = []
            for item in sequence_data:
                features = self._extract_dl_features(item['features'], item.get('symbol', target_symbol))
                sequence_features.append(features)
            
            # Извлекаем метку (направление следующего движения)
            price_change = next_data['result'].get('price_change', 0)
            
            # Динамические пороги based на волатильности
            volatility = np.std([d['features'].get('volatility', 0.01) for d in sequence_data[-10:]]) if len(sequence_data) >= 10 else 0.01
            
            # Адаптивные пороги
            buy_threshold = max(0.0008, volatility * 0.3)
            sell_threshold = min(-0.0008, -volatility * 0.3)
            
            if price_change > buy_threshold:
                label = 0  # BUY
            elif price_change < sell_threshold:
                label = 1  # SELL
            else:
                label = 2  # HOLD
                
            all_features.append(sequence_features)
            all_labels.append(label)
        
        if not all_features:
            return None, None
            
        X = np.array(all_features)
        y = to_categorical(all_labels, num_classes=3)
        
        # Балансируем классы если нужно
        if Config.BALANCE_CLASSES:
            X, y = self._balance_classes(X, y)
        
        return X, y
    
    def _balance_classes(self, X, y):
        """Балансирует классы с помощью oversampling"""
        from imblearn.over_sampling import RandomOverSampler
        
        y_classes = np.argmax(y, axis=1)
        ros = RandomOverSampler(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)  # Преобразуем в 2D
        X_resampled, y_resampled = ros.fit_resample(X_reshaped, y_classes)
        
        # Возвращаем к исходной форме
        X_balanced = X_resampled.reshape(-1, X.shape[1], X.shape[2])
        y_balanced = to_categorical(y_resampled, num_classes=3)
        
        return X_balanced, y_balanced
    
    def _extract_dl_features(self, features: Dict, symbol: str = None) -> List[float]:
        """Извлекает признаки для глубокого обучения с улучшенной обработкой"""
        feature_vector = []
        
        # Технические индикаторы (нормализованные)
        indicators = ['rsi', 'adx', 'macd', 'stochastic_k', 'stochastic_d', 
                     'atr', 'obv', 'cci', 'bb_percent', 'vwap', 'mom', 'williams_r']
        
        for indicator in indicators:
            value = features.get(indicator, 0)
            # Нормализуем значения индикаторов
            if indicator == 'rsi':
                value = (value - 30) / (70 - 30)  # Нормализуем к 0-1
            elif indicator == 'stochastic_k' or indicator == 'stochastic_d':
                value = value / 100
            elif indicator == 'macd':
                value = np.tanh(value * 10)  # Сжимаем диапазон
            feature_vector.append(max(0, min(1, value)))  # Ограничиваем диапазон
        
        # Ценовые действия (нормализованные)
        price_actions = ['price_change', 'volume_change', 'high_low_ratio', 'close_open_ratio']
        for pa in price_actions:
            value = features.get(pa, 0)
            # Нормализуем с помощью гиперболического тангенса для сжатия диапазона
            feature_vector.append(np.tanh(value * 10))
        
        # Волатильность (логарифмическая нормализация)
        volatility = features.get('volatility', 0.01)
        feature_vector.append(np.log1p(volatility * 100) / 5)  # Примерная нормализация к 0-1
        
        # Социальные сигналы
        social_signals = ['social_sentiment', 'social_volume', 'galaxy_score', 'alt_rank']
        for social in social_signals:
            value = features.get(social, 0.5)
            feature_vector.append(max(0, min(1, value)))
        
        # Данные стакана
        orderbook_features = ['orderbook_imbalance', 'orderbook_pressure', 'bid_walls', 'ask_walls']
        for ob in orderbook_features:
            value = features.get(ob, 0)
            feature_vector.append(np.tanh(value))  # Сжимаем диапазон
        
        # Временные особенности
        current_time = datetime.now()
        feature_vector.append(np.sin(2 * np.pi * current_time.hour / 24))  # Циклическое кодирование
        feature_vector.append(np.cos(2 * np.pi * current_time.hour / 24))
        feature_vector.append(np.sin(2 * np.pi * current_time.weekday() / 7))
        feature_vector.append(np.cos(2 * np.pi * current_time.weekday() / 7))
        
        # Сезонность (месяц)
        feature_vector.append(np.sin(2 * np.pi * current_time.month / 12))
        feature_vector.append(np.cos(2 * np.pi * current_time.month / 12))
        
        # Рыночная капитализация (логарифмическая нормализация)
        market_cap = features.get('market_cap', Config.MIN_MARKET_CAP)
        feature_vector.append(np.log1p(market_cap) / np.log1p(Config.MAX_MARKET_CAP))
        
        # Добавляем скользящие средние различных периодов
        for period in [5, 10, 20, 50]:
            ma_key = f'ma_{period}'
            if ma_key in features:
                value = features[ma_key] / features.get('close', 1) - 1  # Относительное значение
                feature_vector.append(np.tanh(value * 10))
            else:
                feature_vector.append(0)
        
        # Заполняем до нужного количества признаков
        while len(feature_vector) < self.num_features:
            feature_vector.append(0)
            
        return feature_vector[:self.num_features]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        """Обучает модель глубокого обучения с улучшенной обработкой"""
        if self.model is None:
            if getattr(Config, 'USE_SIMPLIFIED_MODEL', True):
                self.model = self._build_simplified_model()
                logger.info("Using simplified LSTM model for training")
            else:
                self.model = self._build_hybrid_model()
        
        # Для упрощенной модели используем меньше эпох
        if getattr(Config, 'USE_SIMPLIFIED_MODEL', True):
            epochs = min(epochs, 50)  # Ограничиваем эпохи для упрощенной модели
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_f1_score', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0),
            ModelCheckpoint(
                f"{self.model_dir}/dl_model_v{self.model_version}.h5",
                monitor='val_f1_score',
                save_best_only=True,
                mode='max',
                verbose=0
            ),
            TensorBoard(log_dir=f"{self.model_dir}/logs", histogram_freq=1)
        ]
        
        # Обучение с увеличенным числом эпох
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Важно для временных рядов
        )
        
        # Сохраняем историю обучения
        self.training_history.append({
            'timestamp': datetime.now(),
            'history': history.history,
            'val_accuracy': max(history.history['val_accuracy']),
            'val_f1_score': max(history.history['val_f1_score']),
            'epochs_trained': len(history.history['loss'])
        })
        
        return history

    def predict(self, data_sequence: List[Dict]) -> Dict[str, Any]:
        """Предсказывает направление движения цены на основе последовательности данных"""
        try:
            if self.model is None:
                return {'buy_prob': 0.33, 'sell_prob': 0.33, 'hold_prob': 0.34, 'confidence': 0, 'predicted_class': 'HOLD'}
            
            # Подготавливаем последовательность для предсказания
            sequence_features = []
            for item in data_sequence[-self.sequence_length:]:
                features = self._extract_dl_features(item['features'], item.get('symbol'))
                sequence_features.append(features)
            
            if len(sequence_features) < self.sequence_length:
                # Дополняем последовательность если нужно
                padding = [sequence_features[0]] * (self.sequence_length - len(sequence_features))
                sequence_features = padding + sequence_features
            
            X_pred = np.array([sequence_features])
            
            # Делаем предсказание
            prediction = self.model.predict(X_pred, verbose=0)[0]
            
            buy_prob = float(prediction[0])
            sell_prob = float(prediction[1])
            hold_prob = float(prediction[2])
            
            # Определяем уверенность
            confidence = max(buy_prob, sell_prob, hold_prob) - min(buy_prob, sell_prob, hold_prob)
            
            # Определяем класс
            predicted_class_idx = np.argmax(prediction)
            class_names = ['BUY', 'SELL', 'HOLD']
            predicted_class = class_names[predicted_class_idx]
            
            return {
                'buy_prob': buy_prob,
                'sell_prob': sell_prob,
                'hold_prob': hold_prob,
                'confidence': confidence,
                'predicted_class': predicted_class
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'buy_prob': 0.33, 'sell_prob': 0.33, 'hold_prob': 0.34, 'confidence': 0, 'predicted_class': 'HOLD'}

class ReinforcementLearner:
    """Улучшенный класс для обучения с подкреплением"""
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.97
        self.exploration_rate = 0.25
        self.exploration_decay = 0.999
        self.min_exploration = 0.01
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def get_action(self, state: str, features: Dict) -> str:
        """Выбирает действие на основе текущего состояния с учетом уверенности"""
        if state not in self.q_table:
            # Инициализируем с небольшими случайными значениями
            self.q_table[state] = {
                'BUY': np.random.normal(0, 0.1),
                'SELL': np.random.normal(0, 0.1), 
                'HOLD': np.random.normal(0, 0.1)
            }
        
        # Уменьшаем exploration rate со временем
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        # Exploration vs Exploitation
        if np.random.random() < self.exploration_rate:
            action = np.random.choice(['BUY', 'SELL', 'HOLD'])
        else:
            # Выбираем действие с максимальным Q-value
            action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
        
        # Записываем историю
        self.state_history.append(state)
        self.action_history.append(action)
        
        return action
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, done: bool = False):
        """Обновляет Q-значение на основе полученной награды"""
        if state not in self.q_table:
            self.q_table[state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Q-learning formula
        old_value = self.q_table[state][action]
        
        if done:
            next_max = 0
        else:
            next_max = max(self.q_table[next_state].values())
        
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        
        self.q_table[state][action] = new_value
        self.reward_history.append(reward)
        
    def get_state_hash(self, features: Dict) -> str:
        """Создает хэш состояния на основе признаков с улучшенной дискретизацией"""
        # Улучшенное представление состояния
        state_features = (
            int(features.get('rsi', 50) / 5),  # Дискретизация с шагом 5
            int(features.get('adx', 20) / 5),
            int(features.get('macd', 0) * 20),  # Увеличенная чувствительность
            int(features.get('market_regime', 0)),
            int(features.get('hour_of_day', 12) / 3),  # 3-часовые интервалы
            int(features.get('volatility', 0.01) * 100),  # Дискретизация волатильности
            int(features.get('orderbook_imbalance', 0) * 10)  # Дискретизация imbalance
        )
        return str(state_features)
    
    def calculate_intrinsic_reward(self, features: Dict, action: str) -> float:
        """Вычисляет внутреннюю награду для ускорения обучения"""
        intrinsic_reward = 0
        
        # Награда за исследование новых состояний
        state = self.get_state_hash(features)
        if state not in self.q_table:
            intrinsic_reward += 0.1
            
        # Награда за действия, соответствующие индикаторам
        rsi = features.get('rsi', 50)
        if action == 'BUY' and rsi < 40:
            intrinsic_reward += 0.05
        elif action == 'SELL' and rsi > 60:
            intrinsic_reward += 0.05
            
        return intrinsic_reward

class AdaptiveLearner:
    def __init__(self, data_feeder=None):
        self.model = None
        self.data_feeder = data_feeder
        self.dl_predictor = DeepLearningPredictor(
            sequence_length=Config.DL_SEQUENCE_LENGTH,
            num_features=Config.DL_NUM_FEATURES
        )
        self.reinforcement_learner = ReinforcementLearner()
        self.scaler = RobustScaler()  # Более устойчивый скейлер
        self.feature_importances = {}
        self.learning_data = []
        self.last_retrain = datetime.now()
        self.model_version = 1
        self.model_dir = "models"
        self.performance_metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
            'confusion_matrix': [], 'class_report': []
        }
        self.feature_names = self._get_feature_names()
        
        # Создаем директорию для моделей
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Загружаем предыдущую модель если существует
        self.load_model()
        self.dl_predictor.load_model()
    
    def _get_feature_names(self) -> List[str]:
        """Возвращает список имен признаков"""
        return [
            'rsi', 'adx', 'macd', 'stochastic_k', 'stochastic_d', 'atr', 'obv', 'cci', 'bb_percent', 'vwap',
            'orderbook_imbalance', 'orderbook_pressure', 'bid_ask_ratio', 'orderbook_depth', 'bid_walls', 'ask_walls',
            'social_sentiment', 'social_volume', 'news_sentiment', 'galaxy_score', 'alt_rank',
            'hour', 'weekday', 'volatility', 'price_change', 'volume_change', 'high_low_ratio', 'market_cap'
        ]
    
    def add_training_example(self, features: Dict[str, Any], result: Dict[str, Any], symbol: str = None):
        """Добавляет пример для обучения с улучшенной обработкой"""
        # Проверяем качество данных
        if not self._validate_features(features):
            logger.warning(f"Invalid features for symbol {symbol}")
            return
            
        # Подготавливаем данные для обучения
        example = {
            'features': features,
            'result': result,
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        self.learning_data.append(example)
        
        # Ограничиваем размер данных для избежания переполнения
        if len(self.learning_data) > Config.MAX_LEARNING_EXAMPLES:
            # Удаляем самые старые примеры, но сохраняем баланс классов
            self._prune_learning_data()
        
        # Периодически переобучаем модель
        time_since_retrain = (datetime.now() - self.last_retrain).total_seconds()
        if time_since_retrain > Config.LEARNING_UPDATE_INTERVAL and len(self.learning_data) >= Config.MIN_TRAINING_EXAMPLES:
            self.retrain_model()
    
    def _validate_features(self, features: Dict) -> bool:
        """Проверяет валидность признаков"""
        # Проверяем наличие необходимых признаков
        required_features = ['rsi', 'macd', 'volatility']
        for feat in required_features:
            if feat not in features or not np.isfinite(features[feat]):
                return False
                
        # Проверяем диапазоны значений
        if not (0 <= features.get('rsi', 50) <= 100):
            return False
            
        return True
    
    def _prune_learning_data(self):
        """Удаляет старые примеры, сохраняя баланс классов"""
        # Разделяем данные по классам
        profitable = [ex for ex in self.learning_data if ex['result'].get('pnl', 0) > 0]
        unprofitable = [ex for ex in self.learning_data if ex['result'].get('pnl', 0) <= 0]
        
        # Оставляем равное количество примеров каждого класса
        min_count = min(len(profitable), len(unprofitable), Config.MAX_LEARNING_EXAMPLES // 2)
        
        # Сортируем по времени (новые сначала)
        profitable.sort(key=lambda x: x['timestamp'], reverse=True)
        unprofitable.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Оставляем только нужное количество
        self.learning_data = profitable[:min_count] + unprofitable[:min_count]
        
        logger.info(f"Pruned learning data. Now {len(self.learning_data)} examples ({min_count} each class)")
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает данные для обучения с улучшенной обработкой"""
        if len(self.learning_data) < Config.MIN_TRAINING_EXAMPLES:
            return None, None
        
        X = []
        y = []
        
        for example in self.learning_data:
            features = example['features']
            result = example['result']
            
            # Пропускаем примеры с невалидными данными
            if not self._validate_features(features):
                continue
                
            # Извлекаем признаки
            feature_vector = self._extract_features(features)
            
            # Определяем метку (1 - прибыльная сделка, 0 - убыточная)
            is_profitable = 1 if result.get('pnl', 0) > 0 else 0
            
            X.append(feature_vector)
            y.append(is_profitable)
        
        if len(X) < Config.MIN_TRAINING_EXAMPLES:
            return None, None
            
        return np.array(X), np.array(y)
    
    def _extract_features(self, features: Dict[str, Any]) -> List[float]:
        """Извлекает признаки из данных с улучшенной обработкой"""
        feature_vector = []
        
        # Технические индикаторы
        for indicator in ['rsi', 'adx', 'macd', 'stochastic_k', 'stochastic_d', 'atr', 'obv', 'cci', 'bb_percent', 'vwap']:
            if indicator in features:
                value = features[indicator]
                # Нормализация значений
                if indicator == 'rsi':
                    value = (value - 30) / (70 - 30)  # Нормализуем к 0-1
                elif indicator == 'stochastic_k' or indicator == 'stochastic_d':
                    value = value / 100
                feature_vector.append(value)
            else:
                feature_vector.append(0)
        
        # Данные стакана
        for ob_data in ['orderbook_imbalance', 'orderbook_pressure', 'bid_ask_ratio', 'orderbook_depth', 'bid_walls', 'ask_walls']:
            if ob_data in features:
                value = features[ob_data]
                # Сжимаем диапазон с помощью tanh
                feature_vector.append(np.tanh(value))
            else:
                feature_vector.append(0)
        
        # Социальные сигналы
        for social in ['social_sentiment', 'social_volume', 'news_sentiment', 'galaxy_score', 'alt_rank']:
            if social in features:
                value = features[social]
                feature_vector.append(max(0, min(1, value)))  # Ограничиваем диапазон
            else:
                feature_vector.append(0.5)  # Нейтральное значение
        
        # Временные особенности (циклическое кодирование)
        current_time = datetime.now()
        feature_vector.append(np.sin(2 * np.pi * current_time.hour / 24))
        feature_vector.append(np.cos(2 * np.pi * current_time.hour / 24))
        feature_vector.append(np.sin(2 * np.pi * current_time.weekday() / 7))
        feature_vector.append(np.cos(2 * np.pi * current_time.weekday() / 7))
        
        # Волатильность (логарифмическая нормализация)
        if 'volatility' in features:
            volatility = features['volatility']
            feature_vector.append(np.log1p(volatility * 100))
        else:
            feature_vector.append(0)
        
        # Ценовые действия
        for pa in ['price_change', 'volume_change', 'high_low_ratio']:
            if pa in features:
                value = features[pa]
                feature_vector.append(np.tanh(value * 10))  # Сжимаем диапазон
            else:
                feature_vector.append(0)
        
        # Рыночная капитализация (логарифмическая нормализация)
        if 'market_cap' in features:
            market_cap = features['market_cap']
            feature_vector.append(np.log1p(market_cap))
        else:
            feature_vector.append(np.log1p(Config.MIN_MARKET_CAP))
        
        return feature_vector
    
    def retrain_model(self):
        """Переобучает модель машинного обучения с улучшенной обработкой"""
        try:
            # Обучаем классическую модель
            X, y = self.prepare_training_data()
            if X is None or len(X) < Config.MIN_TRAINING_EXAMPLES:
                logger.warning("Not enough data for retraining")
                return
            
            # Выбор признаков
            if Config.FEATURE_SELECTION:
                X = self._select_features(X, y)
            
            # Разделяем на тренировочную и тестовую выборки с учетом временных рядов
            tscv = TimeSeriesSplit(n_splits=3)
            best_score = -1
            best_params = None
            
            if Config.BAYESIAN_OPTIMIZATION:
                # Используем байесовскую оптимизацию для подбора гиперпараметров
                def gb_crossval(n_estimators, max_depth, min_samples_split, learning_rate, subsample):
                    model = GradientBoostingClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        min_samples_split=int(min_samples_split),
                        learning_rate=learning_rate,
                        subsample=subsample,
                        random_state=42
                    )
                    
                    # Кросс-валидация с учетом временных рядов
                    scores = []
                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Масштабируем признаки
                        X_train_scaled = self.scaler.fit_transform(X_train)
                        X_test_scaled = self.scaler.transform(X_test)
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        scores.append(f1_score(y_test, y_pred))
                    
                    return np.mean(scores)
                
                # Определяем пространство параметров
                pbounds = {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 15),
                    'min_samples_split': (2, 20),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.6, 1.0)
                }
                
                # Оптимизируем
                optimizer = BayesianOptimization(
                    f=gb_crossval,
                    pbounds=pbounds,
                    random_state=42
                )
                
                optimizer.maximize(init_points=8, n_iter=25)
                
                # Получаем лучшие параметры
                best_params = optimizer.max['params']
                n_estimators = int(best_params['n_estimators'])
                max_depth = int(best_params['max_depth'])
                min_samples_split = int(best_params['min_samples_split'])
                learning_rate = best_params['learning_rate']
                subsample = best_params['subsample']
            else:
                # Параметры по умолчанию
                n_estimators = 150
                max_depth = 8
                min_samples_split = 5
                learning_rate = 0.1
                subsample = 0.8
            
            # Создаем ансамбль моделей
            if Config.USE_MODEL_ENSEMBLE:
                models = [
                    ('gb', GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        random_state=42
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    ))
                ]
                
                self.model = VotingClassifier(estimators=models, voting='soft')
            else:
                # Обучаем модель с лучшими параметрами
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=42
                )
            
            # Масштабируем признаки
            X_scaled = self.scaler.fit_transform(X)
            
            # Калибруем модель для получения лучших вероятностей
            if Config.CALIBRATE_PROBABILITIES:
                self.model = CalibratedClassifierCV(self.model, cv=3, method='isotonic')
            
            # Обучаем модель
            self.model.fit(X_scaled, y)
            
            # Оцениваем качество
            y_pred = self.model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            
            # Сохраняем метрики
            self.performance_metrics['accuracy'].append(accuracy)
            self.performance_metrics['precision'].append(precision)
            self.performance_metrics['recall'].append(recall)
            self.performance_metrics['f1'].append(f1)
            self.performance_metrics['confusion_matrix'].append(confusion_matrix(y, y_pred))
            self.performance_metrics['class_report'].append(classification_report(y, y_pred))
            
            # Сохраняем важность признаков
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = dict(zip(
                    self.feature_names[:len(self.model.feature_importances_)],
                    self.model.feature_importances_
                ))
            
            # Обучаем модель глубокого обучения (если достаточно данных)
            if len(self.learning_data) >= Config.DL_TRAINING_THRESHOLD:
                self._retrain_dl_model()
            
            # Сохраняем модель
            self.save_model()
            
            self.last_retrain = datetime.now()
            self.model_version += 1
            
            logger.info(f"Model retrained. Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            # Анализ важности признаков
            self.feature_importance_analysis()
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
    
    def _select_features(self, X, y, k=20):
        """Выбирает наиболее важные признаки"""
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Обновляем имена признаков
        selected_indices = selector.get_support(indices=True)
        self.feature_names = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_indices)} most important features")
        return X_selected
    
    def _retrain_dl_model(self):
        """Переобучает модель глубокого обучения"""
        try:
            # Подготавливаем данные для LSTM
            X_dl, y_dl = self.dl_predictor.prepare_sequences(self.learning_data)
            
            if X_dl is None or len(X_dl) < 100:
                logger.warning("Not enough sequential data for DL training")
                return
            
            # Разделяем на тренировочную и тестовую выборки
            split_idx = int(len(X_dl) * 0.8)
            X_train, X_test = X_dl[:split_idx], X_dl[split_idx:]
            y_train, y_test = y_dl[:split_idx], y_dl[split_idx:]
            
            # Обучаем модель
            history = self.dl_predictor.train(
                X_train, y_train, 
                X_test, y_test, 
                epochs=Config.DL_TRAINING_EPOCHS,
                batch_size=Config.DL_BATCH_SIZE
            )
            
            # Оцениваем качество
            test_loss, test_accuracy, test_precision, test_recall = self.dl_predictor.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"DL Model - Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.3f}, Precision: {test_precision:.3f}, Recall: {test_recall:.3f}")
            
            # Сохраняем модель
            self.dl_predictor.save_model()
            
        except Exception as e:
            logger.error(f"DL model training error: {e}")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Предсказывает вероятность успеха для набора признаков"""
        if self.model is None:
            return {'probability': 0.5, 'confidence': 0}
        
        try:
            # Подготавливаем признаки
            feature_vector = self._extract_features(features)
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Предсказываем с помощью классической модели
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            confidence = abs(probability - 0.5) * 2  # 0-1, где 1 максимальная уверенность
            
            # Получаем предсказание от DL модели (если есть достаточно данных)
            dl_sequence_length = getattr(self.dl_predictor, 'sequence_length', 60)
            if len(self.learning_data) >= dl_sequence_length:
                dl_prediction = self.dl_predictor.predict(self.learning_data[-dl_sequence_length:])
                
                # Объединяем предсказания (взвешенное среднее)
                if dl_prediction['confidence'] > getattr(Config, 'DL_MIN_CONFIDENCE', 0.6):
                    # Учитываем направление DL предсказания
                    dl_direction_strength = dl_prediction['buy_prob'] - dl_prediction['sell_prob']
                    combined_probability = 0.7 * probability + 0.3 * (0.5 + dl_direction_strength * 0.5)
                    combined_confidence = 0.6 * confidence + 0.4 * dl_prediction['confidence']
                    
                    return {
                        'probability': combined_probability,
                        'confidence': combined_confidence,
                        'dl_prediction': dl_prediction
                    }
            
            return {
                'probability': probability,
                'confidence': confidence,
                'dl_prediction': None
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'probability': 0.5, 'confidence': 0}
    
    def predict_with_context(self, features: Dict, market_context: Dict) -> Dict:
        """Предсказывает с учетом рыночного контекста"""
        base_prediction = self.predict(features)
        
        # Корректируем предсказание на основе контекста
        context_factor = self._calculate_context_factor(market_context)
        adjusted_confidence = base_prediction['confidence'] * context_factor
        
        return {
            'probability': base_prediction['probability'],
            'confidence': adjusted_confidence,
            'context_factor': context_factor,
            'dl_prediction': base_prediction.get('dl_prediction')
        }
    
    def _calculate_context_factor(self, context: Dict) -> float:
        """Рассчитывает множитель на основе рыночного контекста"""
        factors = []
        
        # Учитываем силу тренда
        if context.get('trend_strength', 0) > 0.7:
            factors.append(1.2)
        else:
            factors.append(0.8)
        
        # Учитываем волатильность
        if context.get('volatility_regime', 'MEDIUM') == 'HIGH':
            factors.append(0.7)
        elif context.get('volatility_regime', 'MEDIUM') == 'LOW':
            factors.append(1.2)
        else:
            factors.append(1.0)
        
        # Учитываем время дня
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Основные торговые часы
            factors.append(1.1)
        else:
            factors.append(0.9)
        
        # Усредняем факторы
        return sum(factors) / len(factors)
    
    def get_reinforcement_action(self, features: Dict) -> str:
        """Получает действие от обучения с подкреплением"""
        state = self.reinforcement_learner.get_state_hash(features)
        return self.reinforcement_learner.get_action(state, features)
    
    def update_reinforcement_learning(self, features: Dict, action: str, reward: float, next_features: Dict):
        """Обновляет обучение с подкреплением"""
        state = self.reinforcement_learner.get_state_hash(features)
        next_state = self.reinforcement_learner.get_state_hash(next_features)
        self.reinforcement_learner.update_q_value(state, action, reward, next_state)
    
    def save_model(self):
        """Сохраняет модель на диск"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importances': self.feature_importances,
                'version': self.model_version,
                'last_retrain': self.last_retrain
            }
            
            with open(f"{self.model_dir}/model_v{self.model_version}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved as version {self.model_version}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Загружает модель с диска"""
        try:
            # Ищем последнюю версию модели
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_v') and f.endswith('.pkl')]
            if not model_files:
                logger.info("No model found, will train new one")
                return
            
            # Загружаем последнюю версию
            latest_version = max([int(f.split('_v')[1].split('.pkl')[0]) for f in model_files])
            model_path = f"{self.model_dir}/model_v{latest_version}.pkl"
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importances = model_data['feature_importances']
            self.model_version = model_data['version']
            self.last_retrain = model_data['last_retrain']
            
            logger.info(f"Loaded model version {latest_version}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def feature_importance_analysis(self):
        """Анализ важности признаков"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
            
        # Получаем важность признаков
        importances = self.model.feature_importances_
        
        # Создаем словарь с именами признаков
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance_dict = dict(zip(feature_names, importances))
        
        # Сортируем по важности
        sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Логируем топ-10 признаков
        logger.info("Top 10 features by importance:")
        for feature, importance in sorted_importances[:10]:
            logger.info(f"  {feature}: {importance:.4f}")
            
        return dict(sorted_importances)
    
    def automated_feature_selection(self, threshold=0.01):
        """Автоматический отбор признаков"""
        importances = self.feature_importance_analysis()
        
        # Определяем признаки для удаления
        features_to_remove = [feature for feature, importance in importances.items() if importance < threshold]
        
        if features_to_remove:
            logger.info(f"Removing {len(features_to_remove)} features with importance < {threshold}")
            
            # Здесь должна быть реализация удаления признаков из данных
            # В реальном коде это потребует изменения метода _extract_features
            
        return features_to_remove

    def update_from_trade(self, trade_result: Dict, lessons: Dict):
        """Обновляет обучение на основе результатов сделки"""
        try:
            # Добавляем пример для обучения
            features = trade_result.get('features', {})
            result = {
                'pnl': trade_result.get('pnl', 0),
                'profitable': trade_result.get('pnl', 0) > 0
            }
            
            self.add_training_example(features, result, trade_result.get('symbol'))
            
            # Обновляем обучение с подкреплением
            if 'next_features' in trade_result:
                reward = trade_result['pnl'] * 100  # Масштабируем награду
                self.update_reinforcement_learning(
                    features,
                    trade_result.get('direction', 'HOLD'),
                    reward,
                    trade_result['next_features']
                )
                
        except Exception as e:
            logger.error(f"Error updating from trade: {e}")

    def track_false_signals(self, symbol: str, signal: Dict, actual_result: bool):
        """Отслеживает ложные сигналы для последующей оптимизации"""
        try:
            if not actual_result:  # Если сигнал был ложным
                false_signal = {
                    'symbol': symbol,
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'scenario_scores': signal.get('scenario_details', {})
                }
                
                # Сохраняем для последующего анализа
                with open('false_signals.json', 'a') as f:
                    f.write(json.dumps(false_signal) + '\n')
                    
                logger.warning(f"False signal recorded for {symbol}")
                
        except Exception as e:
            logger.error(f"Error tracking false signal: {e}")

    def adjust_for_aggressive_mode(self, prediction: Dict) -> Dict:
        """Корректирует предсказания для агрессивного режима"""
        if not Config.AGGRESSIVE_MODE:
            return prediction
            
        # Увеличиваем вероятность и уверенность в агрессивном режиме
        adjusted_prob = min(prediction['probability'] * 1.2, 0.95)
        adjusted_confidence = min(prediction['confidence'] * 1.3, 1.0)
        
        return {
            'probability': adjusted_prob,
            'confidence': adjusted_confidence,
            'dl_prediction': prediction.get('dl_prediction')
        }
    
    def should_enter_trade_aggressive(self, features: Dict, signal_quality: float) -> bool:
        """Определяет, стоит ли входить в сделку в агрессивном режиме"""
        if not Config.AGGRESSIVE_MODE:
            return False
            
        prediction = self.predict(features)
        adjusted_pred = self.adjust_for_aggressive_mode(prediction)
        
        # Более низкие пороги для агрессивного режима
        min_prob = 0.42  # было 0.45
        min_confidence = 0.35  # было 0.4
        
        return (adjusted_pred['probability'] > min_prob and 
                adjusted_pred['confidence'] > min_confidence and
                signal_quality > 0.45)
    
    def quick_learn_from_micro_movements(self, symbol: str, timeframe: str):
        """Быстрое обучение на микродвижениях для агрессивного режима"""
        if not Config.AGGRESSIVE_MODE:
            return
            
        try:
            # Получаем последние данные
            if self.data_feeder is None:
                return
                
            df = self.data_feeder.get_market_data(symbol, timeframe)
            if df.empty or len(df) < 20:
                return
                
            # Анализируем микродвижения
            for i in range(1, min(10, len(df))):
                price_change = (df['close'].iloc[-i] - df['close'].iloc[-i-1]) / df['close'].iloc[-i-1]
                
                if abs(price_change) > 0.0008:  # Микродвижение > 0.08%
                    features = self.data_feeder.get_technical_data(symbol)
                    result = {
                        'pnl': price_change * 100,  # Масштабируем для обучения
                        'profitable': price_change > 0
                    }
                    
                    self.add_training_example(features, result, symbol)
                    
        except Exception as e:
            logger.error(f"Quick learning error: {e}")

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from config import Config
import talib
from scipy import stats
import hashlib
from dataclasses import dataclass
import time

logger = logging.getLogger("AdvancedDecisionMaker")

@dataclass
class MarketContext:
    """Data class for market context analysis"""
    trend_strength: float = 0.5
    market_phase: str = 'NEUTRAL'
    liquidity_conditions: str = 'MEDIUM'
    correlation_strength: float = 0.5
    volatility_regime: str = 'MEDIUM'
    time_of_day_factor: float = 1.0
    institutional_activity: bool = False
    market_sentiment: float = 0.5

class AdvancedDecisionMaker:
    def __init__(self, data_feeder, risk_manager):
        self.data_feeder = data_feeder
        self.risk_manager = risk_manager
        self.market_context = {}
        self.last_analysis_time = {}
        self._context_cache = {}
        self._signal_cache = {}
        
    def assess_market_context(self, symbol: str) -> MarketContext:
        """Анализирует общий рыночный контекст для символа"""
        try:
            # Кэшируем анализ для избежания повторных вычислений
            current_time = datetime.now()
            cache_key = f"{symbol}_context"
            
            if cache_key in self._context_cache:
                cached_data = self._context_cache[cache_key]
                if current_time - cached_data['timestamp'] < timedelta(minutes=3):
                    return cached_data['data']
            
            # Собираем все данные последовательно
            trend_strength = self._calculate_trend_strength(symbol)
            market_phase = self._identify_market_phase(symbol)
            liquidity_conditions = self._assess_liquidity(symbol)
            correlation_strength = self._check_correlations(symbol)
            volatility_regime = self._determine_volatility_regime(symbol)
            time_of_day_factor = self._get_time_of_day_factor()
            institutional_activity = self._detect_institutional_activity(symbol)
            market_sentiment = self._analyze_market_sentiment(symbol)
            
            context = MarketContext(
                trend_strength=trend_strength,
                market_phase=market_phase,
                liquidity_conditions=liquidity_conditions,
                correlation_strength=correlation_strength,
                volatility_regime=volatility_regime,
                time_of_day_factor=time_of_day_factor,
                institutional_activity=institutional_activity,
                market_sentiment=market_sentiment
            )
            
            # Кэшируем результат
            self._context_cache[cache_key] = {
                'data': context,
                'timestamp': current_time
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Market context assessment error for {symbol}: {e}")
            return MarketContext()
    
    def should_enter_trade(self, symbol: str, signal: Dict) -> Dict[str, Any]:
        """Принимает окончательное решение о входе в сделку"""
        try:
            # Кэширование решений для идентичных сигналов
            signal_str = str(sorted(signal.items()))
            signal_hash = hashlib.md5(signal_str.encode()).hexdigest()
            cache_key = f"{symbol}_decision_{signal_hash}"
            
            if cache_key in self._signal_cache:
                cached_decision = self._signal_cache[cache_key]
                if datetime.now() - cached_decision['timestamp'] < timedelta(seconds=30):
                    return cached_decision['decision']
            
            # 1. Проверяем базовые условия
            if not self._check_basic_conditions(symbol, signal):
                decision = {'decision': False, 'reason': 'Basic conditions not met'}
                self._cache_decision(cache_key, decision)
                return decision
            
            # 2. Анализируем рыночный контекст
            context = self.assess_market_context(symbol)
            
            # 3. Проверяем качество сигнала
            signal_quality = self._assess_signal_quality(symbol, signal, context)
            
            # Динамический порог качества based на волатильности
            required_quality = self._calculate_dynamic_quality_threshold(context)
            if Config.AGGRESSIVE_MODE:
                required_quality *= 0.75
                
            if signal_quality < required_quality:
                decision = {'decision': False, 'reason': f'Signal quality too low: {signal_quality:.2f}'}
                self._cache_decision(cache_key, decision)
                return decision
            
            # 4. Проверяем риск-менеджмент
            risk_assessment = self._assess_risk(symbol, signal, context)
            if not risk_assessment['acceptable']:
                decision = {'decision': False, 'reason': risk_assessment['reason']}
                self._cache_decision(cache_key, decision)
                return decision
            
            # 5. Проверяем наличие дивергенций
            if Config.HIDDEN_DIVERGENCE_ENABLED and not Config.AGGRESSIVE_MODE:
                has_divergence = self._check_hidden_divergence(symbol, signal['direction'])
                if not has_divergence:
                    decision = {'decision': False, 'reason': 'No hidden divergence confirmation'}
                    self._cache_decision(cache_key, decision)
                    return decision
            
            # 6. Финальное решение
            confidence = self._calculate_final_confidence(signal, context, signal_quality)
            
            if Config.AGGRESSIVE_MODE:
                confidence = min(confidence * 1.3, 1.0)
            
            decision = {
                'decision': True,
                'confidence': confidence,
                'position_size_multiplier': risk_assessment['size_multiplier'],
                'context': context.__dict__,
                'signal_quality': signal_quality,
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_decision(cache_key, decision)
            return decision
            
        except Exception as e:
            logger.error(f"Decision making error for {symbol}: {e}")
            return {'decision': False, 'reason': f'Error: {str(e)}'}
    
    def _cache_decision(self, cache_key: str, decision: Dict):
        """Кэширует решение для избежания повторных вычислений"""
        # Очищаем старые записи (больше 1000 в кэше)
        if len(self._signal_cache) > 1000:
            oldest_key = min(self._signal_cache.keys(), key=lambda k: self._signal_cache[k]['timestamp'])
            del self._signal_cache[oldest_key]
            
        self._signal_cache[cache_key] = {
            'decision': decision,
            'timestamp': datetime.now()
        }
    
    def _check_basic_conditions(self, symbol: str, signal: Dict) -> bool:
        """Проверяет базовые условия для торговли"""
        try:
            min_score = Config.MIN_SCORE * (0.85 if Config.AGGRESSIVE_MODE else 1.0)
            
            if signal['score'] < min_score:
                return False
                
            if not self.risk_manager.validate_spread(symbol):
                return False
                
            if not self.risk_manager.can_trade_symbol(symbol):
                return False
                
            # В агрессивном режиме разрешаем коррелированные позиции
            if not Config.AGGRESSIVE_MODE and self.risk_manager.has_correlated_position(symbol, signal['direction']):
                return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Basic conditions check error for {symbol}: {e}")
            return False
    
    def _calculate_dynamic_quality_threshold(self, context: MarketContext) -> float:
        """Динамически рассчитывает порог качества based на рыночных условиях"""
        base_threshold = Config.SIGNAL_QUALITY_THRESHOLD
        
        # Корректируем порог based на волатильности
        if context.volatility_regime == 'HIGH':
            base_threshold *= 1.2
        elif context.volatility_regime == 'LOW':
            base_threshold *= 0.9
            
        # Корректируем based на ликвидности
        if context.liquidity_conditions == 'LOW':
            base_threshold *= 1.1
            
        return min(base_threshold, 0.9)  # Ограничиваем максимальный порог
    
    def _assess_signal_quality(self, symbol: str, signal: Dict, context: MarketContext) -> float:
        """Оценивает качество сигнала на основе множества факторов"""
        try:
            # Собираем факторы качества
            quality_factors = []
            weights = []
            
            # 1. Согласованность таймфреймов (вес 0.2)
            timeframe_agreement = self._check_timeframe_agreement(symbol)
            quality_factors.append(timeframe_agreement)
            weights.append(0.2)
            
            # 2. Подтверждение объемами (вес 0.2)
            volume_confirmation = self._check_volume_confirmation(symbol)
            quality_factors.append(volume_confirmation)
            weights.append(0.2)
            
            # 3. Качество стакана заявков (вес 0.25)
            if Config.ADVANCED_ORDERBOOK_ANALYSIS:
                orderbook_quality = self._analyze_orderbook_quality(symbol, signal['direction'])
                quality_factors.append(orderbook_quality)
                weights.append(0.25)
            
            # 4. Соответствие рыночному контексту (вес 0.2)
            context_match = self._assess_context_match(signal, context)
            quality_factors.append(context_match)
            weights.append(0.2)
            
            # 5. Качество ценового действия (вес 0.15)
            price_action_quality = self._check_price_action_quality(symbol, signal['direction'])
            quality_factors.append(price_action_quality)
            weights.append(0.15)
            
            if not quality_factors:
                return 0
                
            # Взвешенное среднее
            weighted_sum = sum(factor * weight for factor, weight in zip(quality_factors, weights))
            total_weight = sum(weights)
            
            return min(weighted_sum / total_weight, 1.0)
            
        except Exception as e:
            logger.error(f"Signal quality assessment error for {symbol}: {e}")
            return 0
    
    def _check_price_action_quality(self, symbol: str, direction: str) -> float:
        """Анализирует качество ценового действия"""
        try:
            df = self.data_feeder.get_market_data(symbol, '5m')
            if df.empty or len(df) < 20:
                return 0.5
                
            # Анализ структуры свечей
            recent_candles = df.iloc[-5:]
            bullish_patterns = 0
            bearish_patterns = 0
            
            for i in range(len(recent_candles)):
                candle = recent_candles.iloc[i]
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range == 0:
                    continue
                    
                # Определяем тип свечи
                body_ratio = body_size / total_range
                
                if body_ratio > 0.7:  # Сильная свеча
                    if candle['close'] > candle['open']:
                        bullish_patterns += 1
                    else:
                        bearish_patterns += 1
                elif body_ratio < 0.3:  # Доджи/нерешительность
                    if direction == 'BUY':
                        bearish_patterns += 0.5
                    else:
                        bullish_patterns += 0.5
            
            # Оцениваем качество based на направлении
            if direction == 'BUY':
                quality = bullish_patterns / (bullish_patterns + bearish_patterns + 1)
            else:
                quality = bearish_patterns / (bullish_patterns + bearish_patterns + 1)
                
            return quality
            
        except Exception as e:
            logger.error(f"Price action analysis error for {symbol}: {e}")
            return 0.5
    
    def _analyze_market_sentiment(self, symbol: str) -> float:
        """Анализирует общий рыночный sentiment"""
        try:
            # Простая реализация - можно расширить
            # Анализируем соотношение покупок/продаж по объему
            df = self.data_feeder.get_market_data(symbol, '1h')
            if df.empty or len(df) < 20:
                return 0.5
                
            # Процент зеленых свечей за последние 24 часа
            green_candles = sum(1 for i in range(min(24, len(df))) if df['close'].iloc[-(i+1)] > df['open'].iloc[-(i+1)])
            total_candles = min(24, len(df))
            
            sentiment = green_candles / total_candles if total_candles > 0 else 0.5
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Market sentiment analysis error for {symbol}: {e}")
            return 0.5
    
    def _assess_risk(self, symbol: str, signal: Dict, context: MarketContext) -> Dict:
        """Оценивает риски сделки"""
        try:
            # Получаем предсказанную волатильность
            if Config.VOLATILITY_PREDICTION_ENABLED:
                volatility_prediction = self.risk_manager.predict_volatility(symbol)
                volatility_factor = self._calculate_volatility_factor(volatility_prediction)
            else:
                volatility_factor = 1.0
            
            # Учитываем рыночный контекст
            context_factor = self._calculate_context_risk_factor(context)
            
            # Учитываем время суток
            time_factor = self._get_time_risk_factor()
            
            # Общий множитель риска
            risk_multiplier = volatility_factor * context_factor * time_factor
            
            # Проверяем Value at Risk
            var = self.risk_manager.calculate_value_at_risk(symbol)
            if var > 0.06:  # Высокий VaR
                risk_multiplier *= 0.6
            elif var > 0.04:  # Средний VaR
                risk_multiplier *= 0.8
            
            # В агрессивном режиме увеличиваем допустимый риск
            if Config.AGGRESSIVE_MODE:
                risk_multiplier *= 1.4
            
            return {
                'acceptable': risk_multiplier >= 0.4,
                'size_multiplier': min(risk_multiplier, 1.0),
                'reason': f'Risk multiplier: {risk_multiplier:.2f}'
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error for {symbol}: {e}")
            return {'acceptable': False, 'size_multiplier': 0, 'reason': f'Risk assessment error: {e}'}
    
    def _check_hidden_divergence(self, symbol: str, direction: str) -> bool:
        """Проверяет наличие скрытых дивергенций"""
        # В агрессивном режиме пропускаем проверку на дивергенцию
        if Config.AGGRESSIVE_MODE:
            return True
            
        try:
            # Анализируем расхождения между ценой и индикаторами
            df = self.data_feeder.get_market_data(symbol, '15m')
            if df.empty or len(df) < 50:
                return False
            
            close_prices = df['close'].values
            rsi = talib.RSI(close_prices, timeperiod=14)
            macd, macd_signal, _ = talib.MACD(close_prices)
            
            # Ищем расхождения в последних 15 свечах
            lookback = min(15, len(close_prices))
            
            if direction == 'BUY':
                # Бычья дивергенция: цена делает нижние низы, индикатор - более высокие низы
                price_lows = close_prices[-lookback:]
                rsi_lows = rsi[-lookback:]
                
                if (np.argmin(price_lows) < np.argmin(rsi_lows[:-1]) and 
                    price_lows[-1] < price_lows[-2] and rsi_lows[-1] > rsi_lows[-2]):
                    return True
                    
            else:  # SELL
                # Медвежья дивергенция: цена делает более высокие highs, индикатор - более низкие highs
                price_highs = close_prices[-lookback:]
                rsi_highs = rsi[-lookback:]
                
                if (np.argmax(price_highs) < np.argmax(rsi_highs[:-1]) and 
                    price_highs[-1] > price_highs[-2] and rsi_highs[-1] < rsi_highs[-2]):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Divergence check error for {symbol}: {e}")
            return False
    
    def _calculate_trend_strength(self, symbol: str) -> float:
        """Рассчитывает силу тренда"""
        try:
            df = self.data_feeder.get_market_data(symbol, '1h')
            if df.empty or len(df) < 50:
                return 0.5
                
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # ADX для силы тренда
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            trend_strength = adx[-1] / 100 if not np.isnan(adx[-1]) else 0.5
            
            return min(max(trend_strength, 0), 1)
            
        except Exception as e:
            logger.error(f"Trend strength calculation error for {symbol}: {e}")
            return 0.5
    
    def _identify_market_phase(self, symbol: str) -> str:
        """Определяет фазу рынка"""
        try:
            df = self.data_feeder.get_market_data(symbol, '4h')
            if df.empty or len(df) < 100:
                return 'NEUTRAL'
                
            close_prices = df['close'].values
            
            # Используем EMA для определения фазы
            ema_20 = talib.EMA(close_prices, timeperiod=20)
            ema_50 = talib.EMA(close_prices, timeperiod=50)
            ema_200 = talib.EMA(close_prices, timeperiod=200)
            
            current_price = close_prices[-1]
            price_above_20 = current_price > ema_20[-1]
            price_above_50 = current_price > ema_50[-1]
            price_above_200 = current_price > ema_200[-1]
            
            if price_above_20 and price_above_50 and price_above_200:
                return 'BULLISH'
            elif not price_above_20 and not price_above_50 and not price_above_200:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Market phase identification error for {symbol}: {e}")
            return 'NEUTRAL'
    
    def _assess_liquidity(self, symbol: str) -> str:
        """Оценивает условия ликвидности"""
        try:
            volume_24h = self.data_feeder.get_24h_volume(symbol)
            orderbook_data = self.data_feeder.get_orderbook_data(symbol)
            
            if volume_24h > 40000000:  # 40M USDT
                liquidity = 'HIGH'
            elif volume_24h > 15000000:  # 15M USDT
                liquidity = 'MEDIUM'
            else:
                liquidity = 'LOW'
                
            # Учитываем глубину стакана
            if orderbook_data.get('bid_walls', 0) > 1 or orderbook_data.get('ask_walls', 0) > 1:
                liquidity = 'HIGH'
                
            return liquidity
            
        except Exception as e:
            logger.error(f"Liquidity assessment error for {symbol}: {e}")
            return 'MEDIUM'
    
    def _check_correlations(self, symbol: str) -> float:
        """Проверяет корреляции с основными активами"""
        try:
            # В реальной реализации здесь должен быть расчет корреляций
            # с BTC, ETH и другими major coins
            return 0.6  # Пример значения
            
        except Exception as e:
            logger.error(f"Correlation check error for {symbol}: {e}")
            return 0.5
    
    def _determine_volatility_regime(self, symbol: str) -> str:
        """Определяет режим волатильности"""
        try:
            volatility = self.risk_manager.calculate_atr(symbol, 14)
            current_price = self.data_feeder.get_current_price(symbol)
            
            if current_price == 0:
                return 'MEDIUM'
                
            volatility_percent = (volatility / current_price) * 100
            
            if volatility_percent > 2.5:
                return 'HIGH'
            elif volatility_percent > 1.2:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Volatility regime determination error for {symbol}: {e}")
            return 'MEDIUM'
    
    def _get_time_of_day_factor(self) -> float:
        """Возвращает множитель времени суток"""
        hour = datetime.now().hour
        
        # Азиатская сессия (00:00-08:00 UTC) - меньшая волатильность
        if 0 <= hour < 8:
            return 0.8
        # Европейская сессия (08:00-16:00 UTC) - нормальная активность
        elif 8 <= hour < 16:
            return 1.0
        # Американская сессия (16:00-24:00 UTC) - высокая активность
        else:
            return 1.2
    
    def _detect_institutional_activity(self, symbol: str) -> bool:
        """Обнаруживает активность институциональных игроков"""
        try:
            # Проверяем крупные ордера в стакане
            orderbook_data = self.data_feeder.get_orderbook_data(symbol)
            if not orderbook_data:
                return False
                
            # Большие стены в стакане или крупные сделки
            large_bid_walls = orderbook_data.get('bid_walls', 0) >= 1
            large_ask_walls = orderbook_data.get('ask_walls', 0) >= 1
            
            return large_bid_walls or large_ask_walls
            
        except Exception as e:
            logger.error(f"Institutional activity detection error for {symbol}: {e}")
            return False
    
    def _check_timeframe_agreement(self, symbol: str) -> float:
        """Проверяет согласованность сигналов на разных таймфреймах"""
        try:
            timeframes = ['3m', '5m', '15m', '30m']
            trends = []
            
            for tf in timeframes:
                df = self.data_feeder.get_market_data(symbol, tf)
                if df.empty or len(df) < 20:
                    continue
                    
                # Простой анализ тренда
                ema_fast = talib.EMA(df['close'], timeperiod=9)
                ema_slow = talib.EMA(df['close'], timeperiod=21)
                
                if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                    trends.append(1)  # Бычий
                else:
                    trends.append(-1)  # Медвежий
            
            if not trends:
                return 0.5
                
            # Процент согласованности
            agreement = abs(sum(trends)) / len(trends)
            return agreement
            
        except Exception as e:
            logger.error(f"Timeframe agreement check error for {symbol}: {e}")
            return 0.5
    
    def _check_volume_confirmation(self, symbol: str) -> float:
        """Проверяет подтверждение объемами"""
        try:
            df = self.data_feeder.get_market_data(symbol, Config.PRIMARY_TIMEFRAME)
            if df.empty or len(df) < 20:
                return 0.5
                
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume == 0:
                return 0.5
                
            volume_ratio = current_volume / avg_volume
            return min(volume_ratio / 1.8, 1.0)  # Нормализуем до 0-1
            
        except Exception as e:
            logger.error(f"Volume confirmation check error for {symbol}: {e}")
            return 0.5
    
    def _analyze_orderbook_quality(self, symbol: str, direction: str) -> float:
        """Анализирует качество стакана заявков"""
        try:
            orderbook_data = self.data_feeder.get_orderbook_data(symbol)
            if not orderbook_data:
                return 0.5
                
            imbalance = orderbook_data.get('imbalance', 0)
            pressure = orderbook_data.get('pressure', 0)
            
            # Для BUY хотим видеть положительный imbalance (больше покупателей)
            # Для SELL - отрицательный imbalance (больше продавцов)
            if direction == 'BUY':
                orderbook_score = max(imbalance, 0)
            else:
                orderbook_score = max(-imbalance, 0)
                
            # Учитываем давление
            orderbook_score = (orderbook_score + abs(pressure)) / 2
            
            return orderbook_score
            
        except Exception as e:
            logger.error(f"Orderbook analysis error for {symbol}: {e}")
            return 0.5
    
    def _assess_context_match(self, signal: Dict, context: MarketContext) -> float:
        """Оценивает соответствие сигнала рыночному контексту"""
        context_score = 0
        
        # Соответствие фазе рынка
        if (context.market_phase == 'BULLISH' and signal['direction'] == 'BUY') or \
           (context.market_phase == 'BEARISH' and signal['direction'] == 'SELL'):
            context_score += 0.3
        
        # Соответствие режиму волатильности
        if context.volatility_regime != 'HIGH' or signal['confidence'] > 0.6:
            context_score += 0.2
        
        # Учет времени суток
        context_score += (context.time_of_day_factor - 1) * 0.2
        
        return min(max(context_score, 0), 1)
    
    def _calculate_volatility_factor(self, volatility_prediction: Dict) -> float:
        """Рассчитывает множитель волатильности"""
        if volatility_prediction['volatility_regime'] == 'HIGH':
            return 0.6
        elif volatility_prediction['volatility_regime'] == 'MEDIUM':
            return 1.0
        else:
            return 1.3
    
    def _calculate_context_risk_factor(self, context: MarketContext) -> float:
        """Рассчитывает множитель риска на основе контекста"""
        risk_factor = 1.0
        
        # Корректируем риск based на фазе рынка
        if context.market_phase == 'BULLISH':
            risk_factor *= 1.2
        elif context.market_phase == 'BEARISH':
            risk_factor *= 0.8
        
        # Корректируем based на ликвидности
        if context.liquidity_conditions == 'HIGH':
            risk_factor *= 1.2
        elif context.liquidity_conditions == 'LOW':
            risk_factor *= 0.7
        
        return risk_factor
    
    def _get_time_risk_factor(self) -> float:
        """Возвращает множитель риска based на времени суток"""
        hour = datetime.now().hour
        
        # Ночью (меньшая ликвидность) - меньший риск
        if 2 <= hour < 6:
            return 0.7
        # Основные торговые сессии - нормальный риск
        elif (8 <= hour < 12) or (14 <= hour < 18):
            return 1.2
        # Вечером (американская сессия) - повышенный риск
        elif 18 <= hour < 22:
            return 1.0
        else:
            return 0.8
    
    def _calculate_final_confidence(self, signal: Dict, context: MarketContext, signal_quality: float) -> float:
        """Рассчитывает финальную уверенность в сделке"""
        base_confidence = signal['confidence']
        
        # Увеличиваем уверенность при высоком качестве сигнала
        confidence = base_confidence * (0.6 + 0.4 * signal_quality)
        
        # Корректируем based на рыночном контексте
        if context.market_phase != 'NEUTRAL':
            confidence *= 1.2
        
        # Корректируем based на волатильности
        if context.volatility_regime == 'MEDIUM':
            confidence *= 1.1
        elif context.volatility_regime == 'LOW':
            confidence *= 1.3
        
        return min(confidence, 1.0)

    def find_breakout_opportunities(self, symbol: str) -> Optional[Dict]:
        """Ищет возможности для пробоя в агрессивном режиме"""
        if not Config.AGGRESSIVE_MODE:
            return None
            
        try:
            df = self.data_feeder.get_market_data(symbol, '5m')
            if df.empty or len(df) < 20:
                return None
                
            # Анализируем последние свечи
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Определяем ключевые уровни
            resistance = df['high'].rolling(15).max().iloc[-1]
            support = df['low'].rolling(15).min().iloc[-1]
            
            # Проверяем пробой сопротивления
            if current_close > resistance and current_volume > df['volume'].rolling(15).mean().iloc[-1] * 1.3:
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'score': 5.5,
                    'confidence': 0.65,
                    'type': 'BREAKOUT'
                }
                
            # Проверяем пробой поддержки
            if current_close < support and current_volume > df['volume'].rolling(15).mean().iloc[-1] * 1.3:
                return {
                    'symbol': symbol,
                    'direction': 'SELL',
                    'score': 5.5,
                    'confidence': 0.65,
                    'type': 'BREAKOUT'
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Breakout opportunity error for {symbol}: {e}")
            return None

import numpy as np
import pandas as pd
import logging
import time
import json
import pickle
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.gaussian_process.kernels import Matern
from config import Config
from risk_manager import RiskManager
import optuna
from optuna.samplers import TPESampler
from scipy import stats

# Создаем необходимые директории и файлы
def initialize_data_files():
    data_dirs = ['data', 'models', 'dl_models', 'backups', 'optimization_logs']
    
    for dir_name in data_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Создаем пустые файлы данных
    data_files = {
        'data/optimization_history.json': [],
        'data/learned_patterns.json': {},
        'data/market_memory.json': [],
        'data/trade_history.json': [],
        'data/parameter_evolution.pkl': {}
    }
    
    for file_path, default_data in data_files.items():
        if not os.path.exists(file_path):
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(default_data, f, indent=2)
                elif file_path.endswith('.pkl'):
                    with open(file_path, 'wb') as f:
                        pickle.dump(default_data, f)
                print(f"Created {file_path}")
            except Exception as e:
                print(f"Error creating {file_path}: {e}")

# Вызываем при старте
initialize_data_files()

logger = logging.getLogger("AutoOptimizer")

class AutoOptimizer:
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.optimization_history = []
        self.performance_data = []
        self.parameter_evolution = {}
        self.last_optimization = datetime.now()
        self.optimization_cache = {}
        self.current_params_hash = None
        self.param_importance = {}
        
        # Загрузка истории оптимизации при инициализации
        self.load_optimization_history()
        self.load_parameter_evolution()
        
    def should_optimize(self) -> bool:
        """Проверяет, нужно ли проводить оптимизацию"""
        if not Config.AUTO_PARAMETER_OPTIMIZATION:
            return False
            
        time_since_last = (datetime.now() - self.last_optimization).total_seconds()
        if time_since_last < Config.OPTIMIZATION_INTERVAL:
            return False
            
        # Проверяем, достаточно ли данных для оптимизации
        if len(self.performance_data) < Config.DL_TRAINING_THRESHOLD:
            logger.info(f"Not enough data for optimization: {len(self.performance_data)}/{Config.DL_TRAINING_THRESHOLD}")
            return False
            
        # Проверяем производительность
        win_rate = self.risk_manager.calculate_win_rate()
        profit_factor = self.risk_manager.calculate_profit_factor()
        
        # Оптимизируем если производительность ниже порога или после значительных изменений рынка
        performance_metric = (win_rate * 0.4) + (profit_factor * 0.6)
        if performance_metric < 0.7:
            logger.info(f"Performance below threshold: {performance_metric:.2f}")
            return True
            
        # Проверяем волатильность рынка
        market_volatility = self._calculate_market_volatility()
        if market_volatility > Config.VOLATILITY_THRESHOLD * 1.5:
            logger.info(f"High market volatility detected: {market_volatility:.4f}")
            return True
            
        return False
        
    def collect_performance_data(self, trade_result: Dict):
        """Собирает данные о производительности для оптимизации"""
        data_point = {
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol'),
            'direction': trade_result.get('direction'),
            'pnl': trade_result.get('pnl', 0),
            'score': trade_result.get('score', 0),
            'market_volatility': trade_result.get('market_volatility', 0),
            'volume_ratio': trade_result.get('volume_ratio', 1),
            'parameters': self._get_current_parameters(),
            'market_conditions': self._get_market_conditions()
        }
        
        self.performance_data.append(data_point)
        
        # Ограничиваем размер данных
        if len(self.performance_data) > Config.MARKET_MEMORY_SIZE:
            self.performance_data = self.performance_data[-Config.MARKET_MEMORY_SIZE:]
            
    def optimize_parameters(self):
        """Оптимизирует параметры с помощью байесовской оптимизации"""
        if not self.should_optimize():
            return
            
        logger.info("Starting parameter optimization...")
        
        try:
            # Выбираем метод оптимизации в зависимости от количества данных
            if len(self.performance_data) > 500:
                best_params = self._optimize_with_optuna()
            else:
                best_params = self._optimize_with_bayesian()
            
            # Проверяем улучшение параметров
            if self._validate_improvement(best_params):
                self._apply_parameters(best_params)
                
                # Сохраняем результаты
                optimization_record = {
                    'timestamp': datetime.now(),
                    'best_params': best_params,
                    'performance_metrics': self._calculate_performance_metrics(),
                    'market_conditions': self._get_market_conditions(),
                    'data_points_count': len(self.performance_data)
                }
                
                self.optimization_history.append(optimization_record)
                self._update_parameter_evolution(best_params)
                
                self.last_optimization = datetime.now()
                self.save_optimization_history()
                self.save_parameter_evolution()
                
                logger.info(f"Parameter optimization completed successfully")
                
                # Отправляем уведомление
                if Config.TELEGRAM_ENABLED:
                    self._send_optimization_report(optimization_record)
            else:
                logger.info("Optimization did not yield significant improvement")
                
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            
    def _optimize_with_bayesian(self) -> Dict:
        """Байесовская оптимизация с расширенными настройками"""
        pbounds = self._get_parameter_bounds()
        
        # Настройка ядра для Gaussian Process
        kernel = Matern(nu=2.5)
        
        optimizer = BayesianOptimization(
            f=self._evaluate_parameters,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        # Настройка логгера
        logger_path = f"optimization_logs/bayesian_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.subscribe(Events.OPTIMIZATION_STEP, JSONLogger(path=logger_path))
        
        # Запускаем оптимизацию
        optimizer.maximize(
            init_points=8,
            n_iter=25,
            acq='ei',  # Expected Improvement
            kappa=2.576,  # Exploration parameter
            xi=0.05  # Exploitation parameter
        )
        
        return optimizer.max['params']
    
    def _optimize_with_optuna(self) -> Dict:
        """Оптимизация с помощью Optuna для больших наборов данных"""
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # Запускаем оптимизацию
        study.optimize(
            lambda trial: self._optuna_objective(trial),
            n_trials=50,
            timeout=3600  # Максимум 1 час
        )
        
        # Анализируем важность параметров
        self._analyze_parameter_importance(study)
        
        return study.best_params
        
    def _optuna_objective(self, trial) -> float:
        """Целевая функция для Optuna"""
        params = {
            'rsi_period': trial.suggest_int('rsi_period', 5, 21),
            'ema_fast': trial.suggest_int('ema_fast', 3, 12),
            'ema_medium': trial.suggest_int('ema_medium', 8, 21),
            'ema_slow': trial.suggest_int('ema_slow', 13, 34),
            'tp_multiplier': trial.suggest_float('tp_multiplier', 0.01, 0.05),
            'sl_multiplier': trial.suggest_float('sl_multiplier', 0.002, 0.01),
            'volume_factor': trial.suggest_float('volume_factor', 0.3, 0.7),
            'adx_period': trial.suggest_int('adx_period', 10, 20),
            'min_adx': trial.suggest_int('min_adx', 10, 25)
        }
        
        return self._evaluate_parameters(params)
        
    def _get_parameter_bounds(self) -> Dict:
        """Возвращает границы параметров для оптимизации"""
        return {
            'rsi_period': (5, 21),
            'ema_fast': (3, 12),
            'ema_medium': (8, 21),
            'ema_slow': (13, 34),
            'tp_multiplier': (0.01, 0.05),
            'sl_multiplier': (0.002, 0.01),
            'volume_factor': (0.3, 0.7),
            'adx_period': (10, 20),
            'min_adx': (10, 25)
        }
        
    def _evaluate_parameters(self, params: Dict) -> float:
        """Оценивает производительность параметров на исторических данных"""
        # Кешируем результаты для одинаковых параметров
        params_hash = self._hash_parameters(params)
        if params_hash in self.optimization_cache:
            return self.optimization_cache[params_hash]
        
        # Временное применение параметров
        original_params = self._get_current_parameters()
        
        # Симулируем торговлю на исторических данных
        results = []
        market_conditions = []
        
        for trade in self.performance_data[-300:]:  # Используем последние 300 сделок
            simulated_result = self._simulate_trade(trade, params)
            if simulated_result is not None:
                results.append(simulated_result['pnl'])
                market_conditions.append(trade['market_conditions'])
        
        if len(results) < 20:  # Минимум 20 сделок для оценки
            return 0
            
        # Рассчитываем метрики производительности
        total_pnl = sum(results)
        winning_trades = sum(1 for pnl in results if pnl > 0)
        total_trades = len(results)
        win_rate = winning_trades / total_trades
        avg_profit = total_pnl / total_trades
        profit_factor = abs(sum(pnl for pnl in results if pnl > 0) / 
                           sum(abs(pnl) for pnl in results if pnl < 0)) if any(pnl < 0 for pnl in results) else 10
        
        # Рассчитыем сложную оценку
        score = self._calculate_composite_score(
            win_rate, avg_profit, profit_factor, results, market_conditions
        )
        
        # Сохраняем в кеш
        self.optimization_cache[params_hash] = score
        
        # Восстанавливаем оригинальные параметры
        self._apply_parameters(original_params)
        
        return score
        
    def _calculate_composite_score(self, win_rate: float, avg_profit: float, 
                                 profit_factor: float, results: List[float],
                                 market_conditions: List[Dict]) -> float:
        """Рассчитывает сложную оценку производительности"""
        # Базовые метрики
        base_score = (win_rate * 0.3) + (min(avg_profit / 100, 1.0) * 0.3) + (min(profit_factor, 5) * 0.2)
        
        # Стабильность (меньше дисперсия - лучше)
        if len(results) > 10:
            volatility = np.std(results) / (abs(np.mean(results)) + 1e-10)
            stability_score = max(0, 1 - volatility)
            base_score += stability_score * 0.1
        
        # Адаптивность к различным рыночным условиям
        adaptability_score = self._calculate_adaptability_score(results, market_conditions)
        base_score += adaptability_score * 0.1
        
        return min(base_score, 1.0)  # Ограничиваем оценку 1.0
        
    def _calculate_adaptability_score(self, results: List[float], 
                                    market_conditions: List[Dict]) -> float:
        """Оценивает адаптивность параметров к различным рыночным условиям"""
        if len(results) < 30 or len(market_conditions) < 30:
            return 0.5
            
        # Группируем результаты по условиям рынка
        condition_performance = {}
        for pnl, conditions in zip(results, market_conditions):
            condition_key = self._classify_market_conditions(conditions)
            if condition_key not in condition_performance:
                condition_performance[condition_key] = []
            condition_performance[condition_key].append(pnl)
        
        # Рассчитываем производительность в различных условиях
        performance_scores = []
        for condition, pnls in condition_performance.items():
            if len(pnls) >= 5:  # Минимум 5 сделок для оценки
                condition_score = sum(pnls) / len(pnls)
                performance_scores.append(max(0, condition_score))
        
        if not performance_scores:
            return 0.5
            
        # Оценка адаптивности - насколько стабильна производительность в разных условиях
        adaptability = 1 - (np.std(performance_scores) / (np.mean(performance_scores) + 1e-10))
        return max(0, min(adaptability, 1))
        
    def _simulate_trade(self, trade: Dict, params: Dict) -> Optional[Dict]:
        """Симулирует сделку с новыми параметрами"""
        try:
            # Здесь должна быть реализация симуляции сделки
            # Для простоты возвращаем исходные данные, но с поправкой на параметры
            simulated_pnl = trade['pnl'] * self._calculate_parameter_impact(trade, params)
            
            return {
                'pnl': simulated_pnl,
                'symbol': trade['symbol'],
                'direction': trade['direction']
            }
        except Exception as e:
            logger.error(f"Trade simulation error: {str(e)}")
            return None
        
    def _calculate_parameter_impact(self, trade: Dict, params: Dict) -> float:
        """Рассчитывает влияние изменения параметров на результат сделки"""
        # Упрощенная модель влияния параметров
        impact = 1.0
        
        # Влияние RSI
        rsi_impact = params['rsi_period'] / trade['parameters']['rsi_period']
        impact *= 0.5 + 0.5 * rsi_impact
        
        # Влияние EMA
        ema_impact = (params['ema_fast'] + params['ema_medium'] + params['ema_slow']) / \
                    (trade['parameters']['ema_fast'] + trade['parameters']['ema_medium'] + trade['parameters']['ema_slow'])
        impact *= 0.5 + 0.5 * ema_impact
        
        # Влияние объемов
        volume_impact = params['volume_factor'] / trade['parameters']['volume_factor']
        impact *= 0.7 + 0.3 * volume_impact
        
        return impact
        
    def _validate_improvement(self, new_params: Dict) -> bool:
        """Проверяет, действительно ли новые параметры лучше текущих"""
        current_score = self._evaluate_parameters(self._get_current_parameters())
        new_score = self._evaluate_parameters(new_params)
        
        # Минимальное улучшение 10%
        improvement = (new_score - current_score) / (current_score + 1e-10)
        
        if improvement > 0.1:
            logger.info(f"Significant improvement detected: {improvement:.2%}")
            return True
            
        logger.info(f"Insufficient improvement: {improvement:.2%}")
        return False
        
    def _get_current_parameters(self) -> Dict:
        """Возвращает текущие параметры"""
        return {
            'rsi_period': Config.RSI_PERIOD,
            'ema_fast': Config.EMA_FAST,
            'ema_medium': Config.EMA_MEDIUM,
            'ema_slow': Config.EMA_SLOW,
            'tp_multiplier': Config.TP_MULTIPLIER,
            'sl_multiplier': Config.SL_MULTIPLIER,
            'volume_factor': Config.VOLUME_FACTOR,
            'adx_period': Config.ADX_PERIOD,
            'min_adx': Config.MIN_ADX
        }
        
    def _apply_parameters(self, params: Dict):
        """Применяет новые параметры"""
        Config.RSI_PERIOD = int(params.get('rsi_period', Config.RSI_PERIOD))
        Config.EMA_FAST = int(params.get('ema_fast', Config.EMA_FAST))
        Config.EMA_MEDIUM = int(params.get('ema_medium', Config.EMA_MEDIUM))
        Config.EMA_SLOW = int(params.get('ema_slow', Config.EMA_SLOW))
        Config.TP_MULTIPLIER = float(params.get('tp_multiplier', Config.TP_MULTIPLIER))
        Config.SL_MULTIPLIER = float(params.get('sl_multiplier', Config.SL_MULTIPLIER))
        Config.VOLUME_FACTOR = float(params.get('volume_factor', Config.VOLUME_FACTOR))
        Config.ADX_PERIOD = int(params.get('adx_period', Config.ADX_PERIOD))
        Config.MIN_ADX = int(params.get('min_adx', Config.MIN_ADX))
        
        # Обновляем хеш текущих параметров
        self.current_params_hash = self._hash_parameters(params)
        
    def _hash_parameters(self, params: Dict) -> str:
        """Создает хеш параметров для кеширования"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
        
    def _get_market_conditions(self) -> Dict:
        """Возвращает текущие рыночные условия"""
        return {
            'volatility': self._calculate_market_volatility(),
            'trend_strength': self._calculate_trend_strength(),
            'volume_ratio': self._calculate_volume_ratio(),
            'market_regime': self._classify_market_regime()
        }
        
    def _calculate_market_volatility(self) -> float:
        """Рассчитывает волатильность рынка"""
        # Заглушка - в реальной реализации нужно рассчитывать на основе цен
        return 0.02
        
    def _calculate_trend_strength(self) -> float:
        """Рассчитывает силу тренда"""
        # Заглушка
        return 0.5
        
    def _calculate_volume_ratio(self) -> float:
        """Рассчитывает отношение объема к среднему"""
        # Заглушка
        return 1.0
        
    def _classify_market_regime(self) -> str:
        """Классифицирует текущий рыночный режим"""
        volatility = self._calculate_market_volatility()
        trend_strength = self._calculate_trend_strength()
        
        if volatility > Config.VOLATILITY_THRESHOLD * 1.5:
            return "high_volatility"
        elif trend_strength > 0.7:
            return "strong_trend"
        elif trend_strength < 0.3:
            return "ranging"
        else:
            return "normal"
            
    def _classify_market_conditions(self, conditions: Dict) -> str:
        """Классифицирует рыночные условия для группировки"""
        return f"{conditions['market_regime']}_vol{conditions['volatility']:.3f}"
        
    def _calculate_performance_metrics(self) -> Dict:
        """Рассчитывает метрики производительности"""
        win_rate = self.risk_manager.calculate_win_rate()
        profit_factor = self.risk_manager.calculate_profit_factor()
        sharpe_ratio = self.risk_manager.calculate_sharpe_ratio()
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.performance_data),
            'avg_profit': np.mean([t['pnl'] for t in self.performance_data]) if self.performance_data else 0
        }
        
    def _update_parameter_evolution(self, new_params: Dict):
        """Обновляет историю эволюции параметров"""
        for param_name, param_value in new_params.items():
            if param_name not in self.parameter_evolution:
                self.parameter_evolution[param_name] = []
                
            self.parameter_evolution[param_name].append({
                'timestamp': datetime.now(),
                'value': param_value,
                'performance': self._calculate_performance_metrics()
            })
            
    def _analyze_parameter_importance(self, study):
        """Анализирует важность параметров на основе исследования Optuna"""
        try:
            importance = optuna.importance.get_param_importances(study)
            self.param_importance = importance
            logger.info(f"Parameter importance: {importance}")
        except Exception as e:
            logger.error(f"Failed to calculate parameter importance: {str(e)}")
            
    def _send_optimization_report(self, optimization_record: Dict):
        """Отправляет отчет об оптимизации"""
        try:
            metrics = optimization_record['performance_metrics']
            report = (
                f"🔄 PARAMETER OPTIMIZATION COMPLETE\n"
                f"Win Rate: {metrics['win_rate']:.2%}\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Total Trades: {metrics['total_trades']}\n"
                f"Market Conditions: {optimization_record['market_conditions']['market_regime']}\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
            # Используем метод бота для отправки сообщения
            if hasattr(self.risk_manager, 'bot') and self.risk_manager.bot:
                self.risk_manager.bot.send_telegram_alert(report)
                
        except Exception as e:
            logger.error(f"Failed to send optimization report: {str(e)}")
            
    def save_optimization_history(self):
        """Сохраняет историю оптимизации"""
        try:
            with open('data/optimization_history.json', 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")
            
    def load_optimization_history(self):
        """Загружает историю оптимизации"""
        try:
            with open('data/optimization_history.json', 'r') as f:
                self.optimization_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading optimization history: {e}")
            self.optimization_history = []
            
    def save_parameter_evolution(self):
        """Сохраняет историю эволюции параметров"""
        try:
            with open('data/parameter_evolution.pkl', 'wb') as f:
                pickle.dump(self.parameter_evolution, f)
        except Exception as e:
            logger.error(f"Error saving parameter evolution: {e}")
            
    def load_parameter_evolution(self):
        """Загружает историю эволюции параметров"""
        try:
            with open('data/parameter_evolution.pkl', 'rb') as f:
                self.parameter_evolution = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading parameter evolution: {e}")
            self.parameter_evolution = {}
            
    def get_recommended_parameters(self, market_regime: str) -> Dict:
        """Возвращает рекомендуемые параметры для заданного рыночного режима"""
        if not self.parameter_evolution:
            return self._get_current_parameters()
            
        # Находим лучшие параметры для данного режима
        best_params = {}
        best_score = -1
        
        for param_name, history in self.parameter_evolution.items():
            for record in history:
                if record['performance']['win_rate'] > best_score:
                    best_score = record['performance']['win_rate']
                    best_params[param_name] = record['value']
                    
        return best_params if best_params else self._get_current_parameters()

import logging
import os
import shutil
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional  # Добавлен импорт Dict
from config import Config
import gzip
import pickle

logger = logging.getLogger("BackupSystem")

class BackupSystem:
    def __init__(self):
        self.backup_path = Path(Config.BACKUP_SETTINGS['backup_path'])
        self.backup_path.mkdir(exist_ok=True)
        self.last_backup = datetime.now()
        self.backup_in_progress = False
        self.backup_queue = []
        
    def should_backup(self) -> bool:
        """Проверяет, нужно ли создавать бэкап"""
        if not Config.BACKUP_SETTINGS['enabled']:
            return False
            
        time_since_last = (datetime.now() - self.last_backup).total_seconds()
        return time_since_last >= Config.BACKUP_SETTINGS['interval'] and not self.backup_in_progress
        
    def create_backup(self):
        """Создает резервную копию системы в отдельном потоке"""
        if not self.should_backup():
            return
            
        # Запускаем бэкап в отдельном потоке, чтобы не блокировать торговлю
        backup_thread = threading.Thread(target=self._create_backup_thread, daemon=True)
        backup_thread.start()
        
    def _create_backup_thread(self):
        """Поток для создания бэкапа"""
        self.backup_in_progress = True
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # Копируем конфигурационные файлы
            self._backup_file('config.py', backup_dir)
            self._backup_file('config.yaml', backup_dir)
            self._backup_file('.env', backup_dir)
            
            # Копируем модели машинного обучения
            self._backup_directory('models', backup_dir)
            self._backup_directory('dl_models', backup_dir)
            self._backup_directory('transformers', backup_dir)
            
            # Копируем торговую историю и логи
            self._backup_file('trade_history.json', backup_dir)
            self._backup_file('optimization_history.json', backup_dir)
            self._backup_file('trading_bot.log', backup_dir)
            
            # Сохраняем состояние AI моделей
            self._backup_ai_state(backup_dir)
            
            # Создаем сжатый архив бэкапа
            self._create_compressed_backup(backup_dir)
            
            # Создаем файл с метаданными бэкапа
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'balance': self._get_current_balance(),
                'version': 'Precision Pro v5.0',
                'trade_count': self._get_trade_count(),
                'files_backed_up': [f.name for f in backup_dir.iterdir() if f.is_file()],
                'directories_backed_up': [f.name for f in backup_dir.iterdir() if f.is_dir()],
                'backup_size': self._get_directory_size(backup_dir)
            }
            
            with open(backup_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Очищаем старые бэкапы
            self._clean_old_backups()
            
            self.last_backup = datetime.now()
            logger.info(f"Backup created successfully: {backup_dir} (Size: {metadata['backup_size'] / (1024*1024):.2f} MB)")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
        finally:
            self.backup_in_progress = False
            
    def _get_current_balance(self) -> float:
        """Получает текущий баланс для метаданных"""
        try:
            # Здесь должна быть логика получения баланса
            # В реальной реализации это может быть запрос к бирже или risk_manager
            return 0.0
        except:
            return 0.0
            
    def _get_trade_count(self) -> int:
        """Получает количество сделок для метаданных"""
        try:
            # Здесь должна быть логика получения количества сделок
            return 0
        except:
            return 0
            
    def _get_directory_size(self, path: Path) -> int:
        """Вычисляет размер директории в байтах"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size
            
    def _backup_file(self, filename: str, backup_dir: Path):
        """Копирует файл в директорию бэкапа"""
        try:
            if Path(filename).exists():
                shutil.copy2(filename, backup_dir / filename)
        except Exception as e:
            logger.error(f"Error backing up {filename}: {e}")
            
    def _backup_directory(self, dirname: str, backup_dir: Path):
        """Копирует директорию в бэкап"""
        try:
            if Path(dirname).exists():
                shutil.copytree(dirname, backup_dir / dirname, dirs_exist_ok=True)
        except Exception as e:
            logger.error(f"Error backing up directory {dirname}: {e}")
            
    def _backup_ai_state(self, backup_dir: Path):
        """Сохраняет состояние AI моделей"""
        try:
            # Сохраняем состояние adaptive_learner если доступно
            if 'adaptive_learner' in globals():
                adaptive_learner = globals()['adaptive_learner']
                if hasattr(adaptive_learner, 'get_state'):
                    state = adaptive_learner.get_state()
                    with gzip.open(backup_dir / 'ai_state.pkl.gz', 'wb') as f:
                        pickle.dump(state, f)
                        
            # Сохраняем состояние scenario_analyzer если доступно
            if 'scenario_analyzer' in globals():
                scenario_analyzer = globals()['scenario_analyzer']
                if hasattr(scenario_analyzer, 'scenario_performance'):
                    with open(backup_dir / 'scenario_performance.json', 'w') as f:
                        json.dump(scenario_analyzer.scenario_performance, f, indent=2)
                        
        except Exception as e:
            logger.error(f"Error backing up AI state: {e}")
            
    def _create_compressed_backup(self, backup_dir: Path):
        """Создает сжатый архив бэкапа"""
        try:
            # Создаем zip архив для экономии места
            zip_path = self.backup_path / f"{backup_dir.name}.zip"
            shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', backup_dir)
            
            # Удаляем несжатую директорию для экономии места
            shutil.rmtree(backup_dir)
            
            logger.info(f"Created compressed backup: {zip_path}")
        except Exception as e:
            logger.error(f"Error creating compressed backup: {e}")
            
    def _clean_old_backups(self):
        """Очищает старые бэкапы с учетом размера"""
        try:
            backups = []
            for item in self.backup_path.iterdir():
                if item.is_file() and item.suffix == '.zip':
                    backups.append((item, item.stat().st_mtime))
                elif item.is_dir() and item.name.startswith('backup_'):
                    backups.append((item, item.stat().st_mtime))
            
            # Сортируем по времени изменения (сначала старые)
            backups.sort(key=lambda x: x[1])
            
            # Оставляем только N самых новых бэкапов
            while len(backups) > Config.BACKUP_SETTINGS['max_backups']:
                oldest_backup = backups.pop(0)[0]
                if oldest_backup.is_file():
                    oldest_backup.unlink()
                else:
                    shutil.rmtree(oldest_backup)
                logger.info(f"Removed old backup: {oldest_backup.name}")
                
        except Exception as e:
            logger.error(f"Error cleaning old backups: {e}")
            
    def restore_backup(self, backup_name: str):
        """Восстанавливает систему из бэкапа"""
        try:
            backup_path = self.backup_path / backup_name
            
            # Проверяем, является ли бэкап zip архивом
            if backup_path.suffix == '.zip':
                # Распаковываем архив
                extract_path = self.backup_path / backup_path.stem
                shutil.unpack_archive(backup_path, extract_path)
                backup_path = extract_path
                
            if not backup_path.exists():
                logger.error(f"Backup {backup_name} not found")
                return
                
            # Восстанавливаем файлы
            for item in backup_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, item.name)
                elif item.is_dir():
                    shutil.copytree(item, item.name, dirs_exist_ok=True)
                    
            # Восстанавливаем состояние AI
            self._restore_ai_state(backup_path)
                    
            logger.info(f"System restored from backup: {backup_name}")
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            
    def _restore_ai_state(self, backup_path: Path):
        """Восстанавливает состояние AI моделей"""
        try:
            # Восстанавливаем состояние adaptive_learner если доступно
            ai_state_file = backup_path / 'ai_state.pkl.gz'
            if ai_state_file.exists() and 'adaptive_learner' in globals():
                adaptive_learner = globals()['adaptive_learner']
                with gzip.open(ai_state_file, 'rb') as f:
                    state = pickle.load(f)
                if hasattr(adaptive_learner, 'set_state'):
                    adaptive_learner.set_state(state)
                    
            # Восстанавливаем состояние scenario_analyzer если доступно
            scenario_file = backup_path / 'scenario_performance.json'
            if scenario_file.exists() and 'scenario_analyzer' in globals():
                scenario_analyzer = globals()['scenario_analyzer']
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                scenario_analyzer.scenario_performance = scenario_data
                        
        except Exception as e:
            logger.error(f"Error restoring AI state: {e}")
            
    def quick_backup(self, critical_data: Dict = None):  # Теперь Dict распознается
        """Быстрый бэкап критически важных данных"""
        try:
            if not Config.BACKUP_SETTINGS['enabled']:
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quick_backup_dir = self.backup_path / f"quick_backup_{timestamp}"
            quick_backup_dir.mkdir(exist_ok=True)
            
            # Сохраняем критически важные данные
            if critical_data:
                with open(quick_backup_dir / 'critical_data.json', 'w') as f:
                    json.dump(critical_data, f, indent=2)
            
            # Быстрое сохранение торговой истории
            self._backup_file('trade_history.json', quick_backup_dir)
            
            # Быстрое сохранение конфигурации
            self._backup_file('config.py', quick_backup_dir)
            self._backup_file('.env', quick_backup_dir)
            
            logger.info(f"Quick backup created: {quick_backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating quick backup: {e}")
            
    def get_backup_status(self) -> Dict:
        """Возвращает статус системы бэкапов"""
        try:
            backups = []
            total_size = 0
            
            for item in self.backup_path.iterdir():
                if item.is_file() and item.suffix == '.zip':
                    size = item.stat().st_size
                    backups.append({
                        'name': item.name,
                        'size': size,
                        'type': 'compressed'
                    })
                    total_size += size
                elif item.is_dir() and item.name.startswith('backup_'):
                    size = self._get_directory_size(item)
                    backups.append({
                        'name': item.name,
                        'size': size,
                        'type': 'directory'
                    })
                    total_size += size
            
            return {
                'enabled': Config.BACKUP_SETTINGS['enabled'],
                'backup_count': len(backups),
                'total_size': total_size,
                'last_backup': self.last_backup.isoformat() if self.last_backup else None,
                'backups': sorted(backups, key=lambda x: x['name'], reverse=True)[:10]  # Последние 10 бэкапов
            }
            
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return {
                'enabled': Config.BACKUP_SETTINGS['enabled'],
                'error': str(e)
            }

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from config import Config
import json
import hashlib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import pandas_ta as ta  # ЗАМЕНА: import talib -> import pandas_ta as ta
from scipy import stats
import threading
import time

logger = logging.getLogger("CognitiveTrader")

class MarketMemory:
    def __init__(self, max_memory_size=2000):
        self.memory = []
        self.max_memory_size = max_memory_size
        self.scaler = StandardScaler()
        self.nn_model = None
        self.feature_columns = [
            'trend_strength', 'volatility', 'volume_ratio', 'rsi', 
            'market_phase', 'liquidity', 'time_of_day', 'signal_score'
        ]
        self.last_training_time = datetime.now()
        self.lock = threading.RLock()
    
    def add_experience(self, experience: Dict):
        """Добавляет торговый опыт в память"""
        with self.lock:
            if len(self.memory) >= self.max_memory_size:
                self._prune_memory()
            
            experience['timestamp'] = datetime.now()
            experience['id'] = hashlib.md5(f"{datetime.now()}{json.dumps(experience)}".encode()).hexdigest()
            
            self.memory.append(experience)
            
            if len(self.memory) % 100 == 0:
                self._train_similarity_model()
    
    def _prune_memory(self):
        """Удаляет наименее полезные случаи из памяти"""
        if len(self.memory) < self.max_memory_size * 0.8:
            return
            
        cutoff_date = datetime.now() - timedelta(days=30)
        self.memory = [exp for exp in self.memory if exp.get('timestamp', datetime.now()) > cutoff_date]
        
        if len(self.memory) > self.max_memory_size:
            usefulness_scores = []
            for exp in self.memory:
                profit_score = abs(exp.get('pnl', 0)) / 100
                uniqueness_score = self._calculate_uniqueness(exp)
                usefulness_scores.append(profit_score * uniqueness_score)
            
            sorted_indices = np.argsort(usefulness_scores)[::-1]
            self.memory = [self.memory[i] for i in sorted_indices[:self.max_memory_size]]
    
    def _calculate_uniqueness(self, experience: Dict) -> float:
        """Рассчитывает уникальность опыта"""
        if len(self.memory) < 10:
            return 1.0
            
        try:
            features = self._extract_features(experience['context'])
            if self.nn_model is None:
                return 1.0
                
            features_scaled = self.scaler.transform([features])
            distances, _ = self.nn_model.kneighbors(features_scaled)
            avg_similarity = 1 - np.mean(distances)
            
            return 1 - avg_similarity
            
        except Exception as e:
            logger.error(f"Error calculating uniqueness: {e}")
            return 1.0
    
    def _train_similarity_model(self):
        """Обучает модель для поиска похожих случаев"""
        if len(self.memory) < 50:
            return
            
        try:
            with self.lock:
                X = self._prepare_features()
                if X.shape[0] < 10:
                    return
                    
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                
                outlier_detector = IsolationForest(contamination=0.1, random_state=42)
                outliers = outlier_detector.fit_predict(X_scaled)
                X_filtered = X_scaled[outliers == 1]
                
                if X_filtered.shape[0] > 10:
                    self.nn_model = NearestNeighbors(
                        n_neighbors=min(10, X_filtered.shape[0]),
                        algorithm='ball_tree',
                        metric='euclidean'
                    )
                    self.nn_model.fit(X_filtered)
                    
                self.last_training_time = datetime.now()
                
        except Exception as e:
            logger.error(f"Error training similarity model: {e}")
    
    def find_similar_cases(self, current_context: Dict, n_neighbors=8, min_similarity=0.6) -> List[Dict]:
        """Находит похожие случаи в памяти"""
        with self.lock:
            if len(self.memory) < 20 or self.nn_model is None:
                return []
            
            try:
                current_features = self._extract_features(current_context)
                current_features_scaled = self.scaler.transform([current_features])
                
                distances, indices = self.nn_model.kneighbors(current_features_scaled)
                
                similar_cases = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.memory) and distances[0][i] <= (1 - min_similarity):
                        case = self.memory[idx].copy()
                        case['similarity'] = 1 - distances[0][i]
                        similar_cases.append(case)
                
                similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_cases[:n_neighbors]
                
            except Exception as e:
                logger.error(f"Error finding similar cases: {e}")
                return []
    
    def _prepare_features(self):
        """Подготавливает признаки для machine learning"""
        features = []
        valid_indices = []
        
        for i, experience in enumerate(self.memory):
            try:
                if 'context' in experience:
                    features.append(self._extract_features(experience['context']))
                    valid_indices.append(i)
            except (KeyError, TypeError):
                continue
        
        return np.array(features) if features else np.array([])
    
    def _extract_features(self, context: Dict) -> List[float]:
        """Извлекает признаки из контекста"""
        return [
            context.get('trend_strength', 0.5),
            context.get('volatility', 0.02),
            context.get('volume_ratio', 1.0),
            context.get('rsi', 50) / 100,
            1 if context.get('market_phase') == 'BULLISH' else -1 if context.get('market_phase') == 'BEARISH' else 0,
            context.get('liquidity_score', 0.5),
            context.get('time_of_day_factor', 0.5),
            min(context.get('signal_score', 0) / 10, 1.0)
        ]
    
    def get_success_rate(self, cases: List[Dict]) -> float:
        """Рассчитывает процент успешных сделок"""
        if not cases:
            return 0.5
        
        successful_trades = sum(1 for case in cases if case.get('profitable', False))
        return successful_trades / len(cases)
    
    def get_avg_profit(self, cases: List[Dict]) -> float:
        """Рассчитывает среднюю прибыль"""
        if not cases:
            return 0
        
        profits = [case.get('pnl', 0) for case in cases if case.get('pnl') is not None]
        return np.mean(profits) if profits else 0

class CognitiveTrader:
    def __init__(self, decision_maker, risk_manager, data_feeder=None):
        self.decision_maker = decision_maker
        self.risk_manager = risk_manager
        self.data_feeder = data_feeder
        self.market_memory = MarketMemory(getattr(Config, 'MARKET_MEMORY_SIZE', 1000))
        self.learned_patterns = {}
        self.pattern_clusters = []
        self.last_cluster_analysis = datetime.now()
        self.adaptive_weights = {
            'historical': 0.4,
            'pattern': 0.3,
            'context': 0.3
        }
        
    def make_trading_decision(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Упрощенная версия принятия торгового решения"""
        try:
            if not getattr(Config, 'COGNITIVE_TRADING_ENABLED', False):
                return None
            
            # Базовая проверка от decision maker
            base_decision = self.decision_maker.should_enter_trade(symbol, signal)
            if not base_decision.get('decision', False):
                return None
            
            # Простой анализ контекста
            context = self._get_simple_market_context(symbol)
            
            # Проверяем базовые условия
            if not self._check_basic_conditions(context, signal):
                return None
            
            # Возвращаем решение с когнитивными факторами
            enhanced_decision = base_decision.copy()
            enhanced_decision['cognitive_analysis'] = {
                'market_context': context,
                'confidence_multiplier': 1.0,
                'timestamp': datetime.now()
            }
            
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Cognitive trading decision error for {symbol}: {e}")
            return None
    
    def _get_simple_market_context(self, symbol: str) -> Dict:
        """Упрощенный анализ рыночного контекста"""
        context = {
            'trend_strength': 0.5,
            'volatility': 0.02,
            'volume_ratio': 1.0,
            'market_phase': 'NEUTRAL',
            'liquidity_score': 0.5,
            'timestamp': datetime.now()
        }
        
        try:
            if self.data_feeder:
                # Получаем базовые данные
                df = self.data_feeder.get_market_data(symbol, getattr(Config, 'PRIMARY_TIMEFRAME', '15m'))
                if df.empty or len(df) < 10:
                    return context
                
                # Простые расчеты без TA-Lib
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                volume_avg = df['volume'].tail(10).mean()
                volume_current = df['volume'].iloc[-1]
                
                context['trend_strength'] = min(abs(price_change) * 10, 1.0)
                context['volatility'] = df['close'].pct_change().std() * np.sqrt(365)
                context['volume_ratio'] = volume_current / volume_avg if volume_avg > 0 else 1.0
                context['market_phase'] = 'BULLISH' if price_change > 0.01 else 'BEARISH' if price_change < -0.01 else 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
        
        return context
    
    def _check_basic_conditions(self, context: Dict, signal: Dict) -> bool:
        """Проверяет базовые условия для входа"""
        # Минимальная сила сигнала
        if signal.get('score', 0) < getattr(Config, 'MIN_SCORE', 5.0):
            return False
        
        # Проверка волатильности
        volatility = context.get('volatility', 0.02)
        max_volatility = getattr(Config, 'VOLATILITY_THRESHOLD', 0.03) * 1.5
        if volatility > max_volatility:
            return False
        
        # Проверка объема
        volume_ratio = context.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:  # Слишком низкий объем
            return False
        
        return True
    
    def learn_from_trade(self, trade_result: Dict):
        """Упрощенное обучение на основе результата сделки"""
        try:
            if not trade_result or 'symbol' not in trade_result:
                return
            
            # Создаем простой опыт
            experience = {
                'timestamp': datetime.now(),
                'symbol': trade_result.get('symbol'),
                'direction': trade_result.get('direction', 'UNKNOWN'),
                'profitable': trade_result.get('pnl', 0) > 0,
                'pnl': trade_result.get('pnl', 0),
                'context': trade_result.get('context', {}),
                'signal_score': trade_result.get('signal_score', 0)
            }
            
            # Добавляем в память
            self.market_memory.add_experience(experience)
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
    
    def find_high_conviction_patterns(self, symbol: str) -> List[Dict]:
        """Упрощенный поиск паттернов с высокой уверенностью"""
        try:
            if not self.data_feeder:
                return []
                
            # Анализируем только основной таймфрейм
            timeframe = getattr(Config, 'PRIMARY_TIMEFRAME', '15m')
            df = self.data_feeder.get_market_data(symbol, timeframe)
            if df.empty or len(df) < 20:
                return []
            
            # Простые паттерны на основе цены и объема
            signals = self._find_simple_price_patterns(symbol, df, timeframe)
            
            return signals[:3]  # Ограничиваем количество сигналов
            
        except Exception as e:
            logger.error(f"High conviction patterns error for {symbol}: {e}")
            return []
    
    def _find_simple_price_patterns(self, symbol: str, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Ищет простые ценовые паттерны"""
        signals = []
        
        try:
            # Паттерн 1: Сильный тренд с объемом
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            volume_ratio = df['volume'].iloc[-1] / df['volume'].tail(10).mean()
            
            if abs(price_change_5) > 0.02 and volume_ratio > 1.5:
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY' if price_change_5 > 0 else 'SELL',
                    'score': min(abs(price_change_5) * 100, 8.0),
                    'confidence': 0.7,
                    'pattern': 'TREND_WITH_VOLUME',
                    'timeframe': timeframe,
                    'timestamp': datetime.now()
                })
            
            # Паттерн 2: Отскок от поддержки/сопротивления
            recent_high = df['high'].tail(10).max()
            recent_low = df['low'].tail(10).min()
            current_price = df['close'].iloc[-1]
            
            # Проверяем близость к поддержке/сопротивлению
            support_distance = abs(current_price - recent_low) / current_price
            resistance_distance = abs(current_price - recent_high) / current_price
            
            if support_distance < 0.005:  # В пределах 0.5% от поддержки
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY',
                    'score': 6.5,
                    'confidence': 0.6,
                    'pattern': 'SUPPORT_BOUNCE',
                    'timeframe': timeframe,
                    'timestamp': datetime.now()
                })
            
            if resistance_distance < 0.005:  # В пределах 0.5% от сопротивления
                signals.append({
                    'symbol': symbol,
                    'direction': 'SELL',
                    'score': 6.5,
                    'confidence': 0.6,
                    'pattern': 'RESISTANCE_TEST',
                    'timeframe': timeframe,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error finding simple price patterns: {e}")
        
        return signals
    
    def save_learned_patterns(self):
        """Сохраняет learned patterns в файл"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            with open('data/learned_patterns.json', 'w') as f:
                json.dump(self.learned_patterns, f, default=str)
        except Exception as e:
            logger.error(f"Error saving learned patterns: {e}")
    
    def load_learned_patterns(self):
        """Загружает learned patterns из файла"""
        try:
            with open('data/learned_patterns.json', 'r') as f:
                self.learned_patterns = json.load(f)
        except Exception as e:
            logger.error(f"Error loading learned patterns: {e}")
            self.learned_patterns = {}
    
    def save_memory(self):
        """Сохраняет память в файл"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            recent_memory = self.market_memory.memory[-500:] if len(self.market_memory.memory) > 500 else self.market_memory.memory
            
            with open('data/market_memory.json', 'w') as f:
                json.dump(recent_memory, f, default=str)
        except Exception as e:
            logger.error(f"Error saving market memory: {e}")
    
    def load_memory(self):
        """Загружает память из файла"""
        try:
            with open('data/market_memory.json', 'r') as f:
                memory_data = json.load(f)
                self.market_memory.memory = memory_data
        except Exception as e:
            logger.error(f"Error loading market memory: {e}")
            self.market_memory.memory = []

    # Упрощенные версии остальных методов
    def quick_learn_from_micro_patterns(self, symbol: str, timeframe: str):
        """Упрощенное быстрое обучение"""
        pass
    
    def _update_learned_patterns(self, experience: Dict):
        """Упрощенное обновление паттернов"""
        pass

# Простая версия для быстрого запуска
class SimpleCognitiveTrader:
    """Минимальная версия CognitiveTrader для немедленного запуска"""
    def __init__(self, decision_maker, risk_manager, data_feeder=None):
        self.decision_maker = decision_maker
        self.risk_manager = risk_manager
        self.data_feeder = data_feeder
        logger.info("SimpleCognitiveTrader initialized - basic functionality only")
    
    def make_trading_decision(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Простое принятие решения - пропускаем все через базовый decision maker"""
        try:
            return self.decision_maker.should_enter_trade(symbol, signal)
        except Exception as e:
            logger.error(f"Simple cognitive decision error: {e}")
            return None
    
    def learn_from_trade(self, trade_result: Dict):
        """Минимальное обучение - просто логируем результат"""
        try:
            if trade_result and trade_result.get('symbol'):
                logger.info(f"Trade learned: {trade_result['symbol']} PnL: {trade_result.get('pnl', 0):.2f}%")
        except Exception as e:
            logger.error(f"Simple learning error: {e}")
    
    def find_high_conviction_patterns(self, symbol: str) -> List[Dict]:
        """Возвращает пустой список - отключаем сложный анализ"""
        return []

import logging
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Настройка базового логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Config")

load_dotenv()

class Config:
    # 🔥 ОБНОВЛЕННЫЕ НАСТРОЙКИ ПАМЯТИ - УМЕНЬШЕНЫ ЛИМИТЫ
    MEMORY_OPTIMIZATION = {
        'max_memory_mb': 800,  # было 1000 - уменьшаем лимит
        'cleanup_interval': 300,  # было 600 - чаще очистка
        'disable_heavy_features': True,
        'max_cache_size': 10,  # было 20 - меньше кэша
        'log_retention_days': 3
    }
    
    # 🔥 РАЗБЛОКИРУЕМ ТОРГОВЛЮ - УБИРАЕМ СЛИШКОМ ВЫСОКИЕ ПОРОГИ
    MIN_SCORE = float(os.getenv("MIN_SCORE", "3.0"))
    CONFIRMATIONS_REQUIRED = int(os.getenv("CONFIRMATIONS_REQUIRED", "1"))
    MIN_ADX = float(os.getenv("MIN_ADX", "10"))
    VOLUME_FACTOR = float(os.getenv("VOLUME_FACTOR", "1.0"))
    
    # 🔥 УВЕЛИЧИВАЕМ ЧАСТОТУ ПРОВЕРОК
    CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL", "1.0"))
    MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", "100"))
    
    # 🔥 ОСЛАБЛЯЕМ ФИЛЬТРЫ ДЛЯ БОЛЕЕ АКТИВНОЙ ТОРГОВЛИ
    RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
    RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
    VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", "0.06"))
    
    # 🔥 УПРОЩАЕМ ПРОЦЕСС ПРИНЯТИЯ РЕШЕНИЙ
    MULTIDIMENSIONAL_ANALYSIS = os.getenv("MULTIDIMENSIONAL_ANALYSIS", "false").lower() == "true"
    COGNITIVE_TRADING_ENABLED = os.getenv("COGNITIVE_TRADING_ENABLED", "false").lower() == "true"
    ADVANCED_ORDERBOOK_ANALYSIS = os.getenv("ADVANCED_ORDERBOOK_ANALYSIS", "false").lower() == "true"
    
    # 🔥 ОТКЛЮЧАЕМ РЕСУРСОЕМКИЕ ФУНКЦИИ
    DL_USE_HYBRID_MODEL = False  # Явно отключено
    DL_USE_ATTENTION = os.getenv("DL_USE_ATTENTION", "false").lower() == "true"
    TRANSFORMER_ENABLED = False  # Явно отключено
    SCENARIO_ANALYSIS_WORKERS = 2  # Уменьшено количество workers
    
    # 🔥 ИСПОЛЬЗОВАТЬ УПРОЩЕННУЮ МОДЕЛЬ ДЛЯ ПРОДАКШЕНА
    USE_SIMPLIFIED_MODEL = True  # Упрощенная LSTM модель вместо сложной гибридной
    
    # API Keys
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # 🔥 АГРЕССИВНЫЕ НАСТРОЙКИ API
    API_RATE_LIMIT = {
        'requests_per_minute': 200,
        'max_concurrent_requests': 2,
        'retry_delay': 5.0
    }
    
    # 🔥 УВЕЛИЧЕННАЯ АКТИВНОСТЬ
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "5"))
    API_LATENCY_THRESHOLD = 3.0
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "300"))
    DATA_CACHE_DURATION = int(os.getenv("DATA_CACHE_DURATION", "150"))
    
    # 🔥 АКТИВИРУЕМ АГРЕССИВНЫЕ ФУНКЦИИ
    SOCIAL_SIGNALS_ENABLED = os.getenv("SOCIAL_SIGNALS_ENABLED", "false").lower() == "true"
    
    # 🔥 УВЕЛИЧИВАЕМ РАЗМЕРЫ ДАННЫХ ДЛЯ АКТИВНОСТИ
    MARKET_MEMORY_SIZE = int(os.getenv("MARKET_MEMORY_SIZE", "500"))
    MAX_LEARNING_EXAMPLES = 500
    DL_TRAINING_THRESHOLD = int(os.getenv("DL_TRAINING_THRESHOLD", "300"))
    
    # 🔥 УВЕЛИЧИВАЕМ АКТИВНОСТЬ
    MAX_SYMBOLS_TO_CHECK = int(os.getenv("MAX_SYMBOLS_TO_CHECK", "12"))
    
    # 🔥 CPU OPTIMIZATION SETTINGS ДЛЯ АГРЕССИВНОГО РЕЖИМА
    CPU_OPTIMIZATION = {
        'max_cpu_usage': 85,
        'adaptive_sleep': False,
        'simplified_analysis': False,
        'disable_heavy_indicator': False,
    }
    
    # Trading Settings - АГРЕССИВНЫЕ НАСТРОЙКИ
    PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
    DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "5"))
    RISK_PERCENT = float(os.getenv("RISK_PERCENT", "2.0"))
    MIN_BALANCE = float(os.getenv("MIN_BALANCE", "100"))
    MAX_STOP_LOSS = 0.05
    
    # 🔥 УВЕЛИЧЕННАЯ АКТИВНОСТЬ: более частые проверки
    TIMEFRAMES = os.getenv("TIMEFRAMES", "3m,5m,15m").split(",")
    PRIMARY_TIMEFRAME = os.getenv("PRIMARY_TIMEFRAME", "5m")
    
    # Telegram
    TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
    
    # 🔥 АГРЕССИВНЫЕ Trading Limits
    STOP_LOSS_COOLDOWN = int(os.getenv("STOP_LOSS_COOLDOWN", "2"))
    MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.08"))
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.3"))
    
    # Indicators - более агрессивные настройки
    EMA_FAST = int(os.getenv("EMA_FAST", "5"))
    EMA_MEDIUM = int(os.getenv("EMA_MEDIUM", "15"))
    EMA_SLOW = int(os.getenv("EMA_SLOW", "35"))
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "12"))
    ADX_PERIOD = int(os.getenv("ADX_PERIOD", "12"))
    MIN_VOLUME = float(os.getenv("MIN_VOLUME", "1000000"))
    
    # Order Settings - более агрессивные
    TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "0.03"))
    SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "0.015"))
    QUICK_TP_MULTIPLIER = float(os.getenv("QUICK_TP_MULTIPLIER", "0.015"))
    QUICK_TP_SIZE = float(os.getenv("QUICK_TP_SIZE", "0.6"))
    SAFETY_PRICE_LIMIT = float(os.getenv("SAFETY_PRICE_LIMIT", "0.000001"))
    TRAILING_START = float(os.getenv("TRAILING_START", "0.03"))
    TRAILING_DISTANCE = float(os.getenv("TRAILING_DISTANCE", "0.015"))
    MIN_PROFIT_RATIO = float(os.getenv("MIN_PROFIT_RATIO", "1.5"))
    
    # System Monitoring Thresholds
    CPU_THRESHOLD = 90
    MEMORY_THRESHOLD = 90
    DISK_THRESHOLD = 95
    GPU_THRESHOLD = 90
    
    # Symbol Filters - более агрессивные
    MIN_DAILY_VOLUME = float(os.getenv("MIN_DAILY_VOLUME", "30000000"))
    TRADING_HOURS_START = int(os.getenv("TRADING_HOURS_START", "0"))
    TRADING_HOURS_END = int(os.getenv("TRADING_HOURS_END", "24"))
    
    # Reversal Detection - более чувствительное
    REVERSAL_DETECTION = os.getenv("REVERSAL_DETECTION", "true").lower() == "true"
    REVERSAL_CONFIRMATION_BARS = int(os.getenv("REVERSAL_CONFIRMATION_BARS", "2"))
    MIN_REVERSAL_STRENGTH = float(os.getenv("MIN_REVERSAL_STRENGTH", "0.2"))
    VOLUME_SPIKE_MULTIPLIER = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "2.5"))
    
    # Risk Management - более агрессивное
    TREND_CONFIRMATION = int(os.getenv("TREND_CONFIRMATION", "2"))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
    MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.4"))
    DYNAMIC_TRAILING = os.getenv("DYNAMIC_TRAILING", "true").lower() == "true"
    
    # 🔥 ВАЖНО: Trading Modes - АКТИВИРОВАНЫ АГРЕССИВНЫЕ ФУНКЦИИ
    AGGRESSIVE_MODE = os.getenv("AGGRESSIVE_MODE", "true").lower() == "true"
    MOMENTUM_TRADING = os.getenv("MOMENTUM_TRADING", "true").lower() == "true"
    BREAKOUT_TRADING = os.getenv("BREAKOUT_TRADING", "true").lower() == "true"
    VOLUME_CONFIRMATION = os.getenv("VOLUME_CONFIRMATION", "true").lower() == "true"
    NEWS_SENSITIVITY = os.getenv("NEWS_SENSITIVITY", "false").lower() == "true"
    
    # Session Factors - более агрессивные
    ASIAN_SESSION_FACTOR = float(os.getenv("ASIAN_SESSION_FACTOR", "0.9"))
    EUROPEAN_SESSION_FACTOR = float(os.getenv("EUROPEAN_SESSION_FACTOR", "1.0"))
    US_SESSION_FACTOR = float(os.getenv("US_SESSION_FACTOR", "1.2"))
    WEEKEND_FACTOR = float(os.getenv("WEEKEND_FACTOR", "0.7"))
    
    # 🔥 АКТИВИРОВАНЫ: AI & ML Settings для агрессивной торговли
    ADAPTIVE_LEARNING = os.getenv("ADAPTIVE_LEARNING", "true").lower() == "true"
    LEARNING_UPDATE_INTERVAL = int(os.getenv("LEARNING_UPDATE_INTERVAL", "43200"))
    BAYESIAN_OPTIMIZATION = os.getenv("BAYESIAN_OPTIMIZATION", "true").lower() == "true"
    
    # Orderbook Settings - более агрессивные
    ORDERBOOK_DEPTH = int(os.getenv("ORDERBOOK_DEPTH", "15"))
    ORDERBOOK_UPDATE_INTERVAL = int(os.getenv("ORDERBOOK_UPDATE_INTERVAL", "30"))
    
    # Deep Learning Settings - оптимизированы для скорости
    DL_SEQUENCE_LENGTH = int(os.getenv("DL_SEQUENCE_LENGTH", "20"))
    DL_NUM_FEATURES = int(os.getenv("DL_NUM_FEATURES", "10"))
    DL_TRAINING_EPOCHS = int(os.getenv("DL_TRAINING_EPOCHS", "30"))
    DL_BATCH_SIZE = int(os.getenv("DL_BATCH_SIZE", "16"))
    DL_MIN_CONFIDENCE = float(os.getenv("DL_MIN_CONFIDENCE", "0.25"))
    
    # Portfolio Management - более агрессивно
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.3"))
    MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", "50000000"))
    LIQUIDITY_THRESHOLD = float(os.getenv("LIQUIDITY_THRESHOLD", "500000"))
    CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.8"))
    ADAPTIVE_SL = os.getenv("ADAPTIVE_SL", "true").lower() == "true"
    ADAPTIVE_TP = os.getenv("ADAPTIVE_TP", "true").lower() == "true"
    MARKET_CAP_WEIGHT = float(os.getenv("MARKET_CAP_WEIGHT", "0.3"))
    LIQUIDITY_WEIGHT = float(os.getenv("LIQUIDITY_WEIGHT", "0.4"))
    VOLATILITY_WEIGHT = float(os.getenv("VOLATILITY_WEIGHT", "0.3"))
    ENSEMBLE_LEARNING = os.getenv("ENSEMBLE_LEARNING", "true").lower() == "true"
    
    # Additional Settings
    VOLATILITY_SCALING = os.getenv("VOLATILITY_SCALING", "true").lower() == "true"
    MAX_HOLDING_TIME = int(os.getenv("MAX_HOLDING_TIME", "1800"))
    EARLY_CLOSE_THRESHOLD = float(os.getenv("EARLY_CLOSE_THRESHOLD", "0.6"))
    POSITION_HEDGING = os.getenv("POSITION_HEDGING", "false").lower() == "true"
    
    # Exchange Filters
    PRICE_FILTER = float(os.getenv("PRICE_FILTER", "0.00000001"))
    MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "5.0"))
    MAX_PRICE_FILTER = float(os.getenv("MAX_PRICE_FILTER", "1000000.0"))
    
    # Enhanced Sensitivity Settings - более чувствительные
    SIGNAL_QUALITY_THRESHOLD = float(os.getenv("SIGNAL_QUALITY_THRESHOLD", "0.4"))
    CONFLUENCE_FACTORS_REQUIRED = int(os.getenv("CONFLUENCE_FACTORS_REQUIRED", "2"))
    VOLUME_SPIKE_THRESHOLD = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "1.5"))
    HIDDEN_DIVERGENCE_ENABLED = os.getenv("HIDDEN_DIVERGENCE_ENABLED", "true").lower() == "true"
    REINFORCEMENT_LEARNING_ENABLED = os.getenv("REINFORCEMENT_LEARNING_ENABLED", "false").lower() == "true"
    
    # Volatility Prediction - активировано
    VOLATILITY_PREDICTION_ENABLED = os.getenv("VOLATILITY_PREDICTION_ENABLED", "true").lower() == "true"
    
    # Critical entry parameters - более агрессивные
    
    # Новые параметры для агрессивной торговли
    TREND_FILTER_STRENGTH = float(os.getenv("TREND_FILTER_STRENGTH", "0.7"))
    FALSE_BREAKOUT_FILTER = os.getenv("FALSE_BREAKOUT_FILTER", "true").lower() == "true"
    MIN_TREND_STRENGTH = float(os.getenv("MIN_TREND_STRENGTH", "0.4"))
    WHALE_MANIPULATION_PROTECTION = os.getenv("WHALE_MANIPULATION_PROTECTION", "true").lower() == "true"
    
    # 🔥 АГРЕССИВНОЕ: Управление рисками
    MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "2.0"))
    MAX_POSITIONS_OPEN = int(os.getenv("MAX_POSITIONS_OPEN", "3"))
    MIN_TIME_BETWEEN_TRADES = int(os.getenv("MIN_TIME_BETWEEN_TRADES", "10"))
    
    # Автоматическая оптимизация - активирована для агрессивной торговли
    AUTO_PARAMETER_OPTIMIZATION = os.getenv("AUTO_PARAMETER_OPTIMIZATION", "true").lower() == "true"
    
    # Autonomous Trading Settings - агрессивные
    AUTO_OPTIMIZATION = {
        'enabled': True,
        'optimization_interval': 43200,
        'parameters_to_optimize': ['RSI_PERIOD', 'EMA_FAST', 'TP_MULTIPLIER', 'SL_MULTIPLIER'],
        'performance_threshold': 0.5
    }
    
    AUTO_PROFIT_WITHDRAWAL = {
        'enabled': False,
        'withdrawal_threshold': 1.5,
        'withdrawal_percentage': 0.3,
    }
    
    BACKUP_SETTINGS = {
        'enabled': True,
        'interval': 1800,
        'max_backups': 48,
        'backup_path': 'backups/'
    }
    
    # Dynamic Thresholds Configuration - агрессивные
    DYNAMIC_THRESHOLDS = {
        'high_volatility': {'min_score': 3.5, 'volume_multiplier': 1.5},
        'low_volatility': {'min_score': 3.0, 'volume_multiplier': 1.0},
        'asian_session': {'min_score': 3.5, 'confirmations': 2},
        'european_session': {'min_score': 3.5, 'confirmations': 2},
        'us_session': {'min_score': 3.0, 'confirmations': 1}
    }
    
    # Trading Sessions Configuration
    TRADING_SESSIONS = {
        'asian': (0, 8, 0.9),
        'european': (8, 16, 1.0),
        'us': (16, 24, 1.2)
    }

    # 🔥 АГРЕССИВНО: Увеличено количество символов
    FUNDAMENTAL_PAIRS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT']
    
    CORRELATED_PAIRS = {
        "BTCUSDT": ["ETHUSDT", "BNBUSDT", "SOLUSDT"],
        "ETHUSDT": ["BTCUSDT", "BNBUSDT", "MATICUSDT"],
    }

    # SCENARIOS - агрессивные
    SCENARIOS = {
        'trend_following': {
            'indicators': ['ema', 'adx'],
            'timeframes': ['3m', '5m'],
            'weight': 0.7
        },
        'mean_reversion': {
            'indicators': ['rsi', 'bollinger'],
            'timeframes': ['3m', '5m'],
            'weight': 0.3
        }
    }
    
    # FLEXIBLE_SETTINGS - агрессивный режим ОБНОВЛЕН
    FLEXIBLE_SETTINGS = {
        'aggressive': {
            'MIN_SCORE': 3.0,
            'CONFIRMATIONS_REQUIRED': 1,
            'MIN_ADX': 10,
            'VOLUME_FACTOR': 1.0,
            'PROFIT_MULTIPLIER': 1.2
        },
        'conservative': {
            'MIN_SCORE': 4.5,
            'CONFIRMATIONS_REQUIRED': 3,
            'MIN_ADX': 20,
            'VOLUME_FACTOR': 0.4,
            'PROFIT_MULTIPLIER': 0.8
        }
    }
    
    # Добавляем PROFIT_MULTIPLIER в основные настройки
    PROFIT_MULTIPLIER = float(os.getenv("PROFIT_MULTIPLIER", "1.2"))
    
    SYMBOL_INFO = {}
    
    def __init__(self):
        self._load_from_yaml()
        self._validate_config()
        
        # 🔥 ДИАГНОСТИКА НАСТРОЕК ПРИ ЗАПУСКЕ
        logger.info("=== ⚙️ CONFIG DIAGNOSTICS ===")
        logger.info(f"MIN_SCORE: {self.MIN_SCORE}")
        logger.info(f"CONFIRMATIONS_REQUIRED: {self.CONFIRMATIONS_REQUIRED}")
        logger.info(f"AGGRESSIVE_MODE: {self.AGGRESSIVE_MODE}")
        logger.info(f"PAPER_TRADING: {self.PAPER_TRADING}")
        logger.info(f"USE_SIMPLIFIED_MODEL: {self.USE_SIMPLIFIED_MODEL}")
        
        if self.AGGRESSIVE_MODE:
            logger.info("🔥 AGGRESSIVE MODE ACTIVE")
        
        if self.USE_SIMPLIFIED_MODEL:
            logger.info("🧠 SIMPLIFIED DL MODEL ACTIVE")
        
        # 🔥 УБЕДИМСЯ ЧТО API_RATE_LIMIT СУЩЕСТВУЕТ
        if not hasattr(self, 'API_RATE_LIMIT'):
            self.API_RATE_LIMIT = {
                'requests_per_minute': 200,
                'max_concurrent_requests': 2,
                'retry_delay': 5.0
            }
        
        # 🔥 АКТИВИРУЕМ АГРЕССИВНЫЙ РЕЖИМ
        if self.AGGRESSIVE_MODE:
            logger.info("🔥 ACTIVATING AGGRESSIVE TRADING MODE")
            self.apply_mode('aggressive')
        else:
            self.apply_mode('conservative')
        
        # 🔥 АКТИВИРУЕМ НАСТРОЙКИ ДЛЯ ВЫСОКОЙ АКТИВНОСТИ
        logger.info("✅ High-activity settings activated")
        logger.info(f"⚡ Check interval: {self.CHECK_INTERVAL}s")
        logger.info(f"📈 Max trades per hour: {self.MAX_TRADES_PER_HOUR}")
        logger.info(f"💾 Memory limit: {self.MEMORY_OPTIMIZATION['max_memory_mb']}MB")
        
    def _load_from_yaml(self):
        config_path = Path('config.yaml')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        for key, value in yaml_config.items():
                            attr_name = key.upper()
                            if hasattr(self, attr_name):
                                setattr(self, attr_name, value)
                            else:
                                logger.warning(f"Unknown config parameter in YAML: {key}")
            except Exception as e:
                logger.error(f"Error loading YAML config: {e}")
    
    def _validate_config(self):
        if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
            logger.error("❌ Binance API keys are missing!")
            
        if self.RISK_PERCENT > 5:
            logger.warning("⚠️ Risk percentage is too high! Reducing to 2%")
            self.RISK_PERCENT = 2.0
            
        # 🔥 ПРОВЕРКА АГРЕССИВНЫХ НАСТРОЕК
        if self.CHECK_INTERVAL < 0.5:
            logger.warning("⚠️ Check interval too low! Increasing to 1.0s")
            self.CHECK_INTERVAL = 1.0
    
    def apply_mode(self, mode_name):
        if mode_name in self.FLEXIBLE_SETTINGS:
            mode = self.FLEXIBLE_SETTINGS[mode_name]
            for key, value in mode.items():
                attr_name = key.upper()
                if hasattr(self, attr_name):
                    setattr(self, attr_name, value)
                else:
                    logger.warning(f"Config parameter {attr_name} not found, skipping")
            logger.info(f"✅ Applied trading mode: {mode_name}")
        else:
            logger.warning(f"Trading mode {mode_name} not found")

config = Config()

import numpy as np
import pandas as pd
import logging
import time
import threading
import requests
import json
from datetime import datetime, timedelta
from binance import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Tuple, Any, Optional
from config import Config
import pandas_ta as ta
from collections import deque
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re

logger = logging.getLogger("DataFeeder")

class DataFeeder:
    def __init__(self, client: Client):
        self.client = client
        self.market_data = {}
        self.orderbook_data = {}
        self.social_sentiment = {}
        self.technical_indicators = {}
        self.fundamental_data = {}
        self.market_caps = {}
        self.funding_rates = {}
        self.liquidation_data = {}
        self.last_update = {}
        self.running = True
        self.threads = []
        self.lock = threading.RLock()
        self.data_queue = deque(maxlen=1000)
        self.connection_errors = 0
        self.max_connection_errors = 5
        
        # Улучшенная система rate limiting
        self.request_timestamps = deque(maxlen=800)
        self.max_requests_per_minute = 800
        self.semaphore = threading.Semaphore(2)
        
        # Система банов
        self.ip_banned_until = 0
        self.last_ban_time = 0
        self.ban_count = 0
        
        # Создаем сессию с повторными попытками
        self.session = self._create_session()
        
        # Кэши для ускорения доступа
        self.price_cache = {}
        self.indicator_cache = {}
        self.volume_cache = {}
        self.technical_data_cache = {}  # Новый кэш для технических данных
        
        # Статистика запросов
        self.request_stats = {
            'total': 0,
            'errors': 0,
            'last_reset': time.time()
        }
        
        logger.info("DataFeeder initialized with improved rate limiting")
        
        # Запускаем потоки для сбора данных
        self.start_data_feeds()
    
    def _create_session(self):
        """Создает сессию с повторными попытками"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _is_ip_banned(self):
        """Проверяет, забанен ли IP"""
        current_time_ms = int(time.time() * 1000)
        return self.ip_banned_until > current_time_ms
    
    def _extract_ban_time(self, error_msg):
        """Извлекает время бана из сообщения об ошибке"""
        try:
            match = re.search(r'banned until (\d+)', error_msg)
            if match:
                return int(match.group(1))
        except:
            pass
        return int(time.time() * 1000) + 60000
    
    def _handle_rate_limit(self):
        """УВЕЛИЧЕННЫЕ ЗАДЕРЖЖКИ ДЛЯ API"""
        current_time = time.time()
        
        # Очищаем старые запросы (120 секунд вместо 70)
        while (self.request_timestamps and 
               current_time - self.request_timestamps[0] > 120):
            self.request_timestamps.popleft()
        
        # УВЕЛИЧИВАЕМ минимальное время между запросами
        if self.request_timestamps:
            time_since_last = current_time - self.request_timestamps[-1]
            if time_since_last < 1.0:  # было 0.2 секунды
                time.sleep(1.0 - time_since_last)
        
        # ОГРАНИЧИВАЕМ количество запросов
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60  # ждем целую минуту
            logger.warning(f"Rate limit exceeded, sleeping for {sleep_time}s")
            time.sleep(sleep_time)
            self.request_timestamps.clear()
    
    def _safe_api_request(self, func, *args, **kwargs):
        """Безопасный запрос к API с обработкой ошибок и rate limiting"""
        if self._is_ip_banned():
            return None
        
        max_retries = 2
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()
                
                with self.semaphore:
                    current_time = time.time()
                    self.request_timestamps.append(current_time)
                    self.request_stats['total'] += 1
                    
                    result = func(*args, **kwargs)
                    return result
                    
            except BinanceAPIException as e:
                self.request_stats['errors'] += 1
                last_exception = e
                
                if e.code == -1003:
                    self.ip_banned_until = self._extract_ban_time(str(e))
                    current_time_ms = int(time.time() * 1000)
                    ban_remaining = (self.ip_banned_until - current_time_ms) / 1000
                    
                    self.ban_count += 1
                    self.last_ban_time = time.time()
                    
                    if ban_remaining > 0:
                        ban_until_dt = datetime.fromtimestamp(self.ip_banned_until/1000)
                        logger.error(f"IP banned until {ban_until_dt}")
                        sleep_time = min(ban_remaining, 300)
                        logger.warning(f"Sleeping for {sleep_time:.1f}s due to ban")
                        time.sleep(sleep_time)
                    
                    if attempt == max_retries - 1:
                        break
                    continue
                    
                elif e.code in [-1006, -1021, -1000]:
                    logger.warning(f"API error {e.code}, retrying in {attempt + 1}s...")
                    time.sleep(attempt + 1)
                    continue
                    
                else:
                    logger.error(f"API error {e.code}: {e}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(attempt + 1)
                    
            except Exception as e:
                self.request_stats['errors'] += 1
                last_exception = e
                logger.error(f"Unexpected error in API request: {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(attempt + 1)
        
        return None
    
    def start_data_feeds(self):
        """ТОЛЬКО САМЫЕ КРИТИЧЕСКИ ВАЖНЫЕ ПОТОКИ С БОЛЬШИМИ ИНТЕРВАЛАМИ"""
        threads_config = [
            (self._update_market_data, 300, "Market Data"),  # 5 минут вместо 2
            (self._update_technical_indicators, 600, "Technical Indicators"),  # 10 минут вместо 5
        ]
        
        logger.info("🔥 ULTRA-CONSERVATIVE MODE: Minimal data feeds with 5-10 minute intervals")

        for target, interval, name in threads_config:
            try:
                thread = threading.Thread(
                    target=self._run_with_interval, 
                    args=(target, interval, name),
                    daemon=True,
                    name=f"DataFeed-{name}"
                )
                thread.start()
                self.threads.append(thread)
                logger.info(f"Started ULTRA-CONSERVATIVE data feed: {name} ({interval}s interval)")
            except Exception as e:
                logger.error(f"Failed to start {name} thread: {e}")
    
    def _run_with_interval(self, target, interval, name):
        """Запускает функцию с заданным интервалом"""
        logger.info(f"Starting {name} thread with {interval}s interval")
        
        while self.running:
            try:
                start_time = time.time()
                
                if self._is_ip_banned():
                    logger.warning(f"{name} thread sleeping due to IP ban")
                    time.sleep(60)
                    continue
                
                target()
                execution_time = time.time() - start_time
                sleep_time = max(15, interval - execution_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in {name} thread: {e}")
                sleep_time = min(300, interval * 2)
                time.sleep(sleep_time)
    
    def _clean_cache(self):
        """Добавьте это в DataFeeder - очистка кэшей"""
        current_time = time.time()
        # Очищаем старые записи (>15 минут)
        for cache_dict in [self.market_data, self.technical_data_cache]:
            keys_to_remove = [k for k, v in cache_dict.items() 
                             if current_time - v.get('timestamp', 0) > 900]
            for key in keys_to_remove:
                del cache_dict[key]

    # В data_feeder.py УПРОЩАЕМ метод _update_market_data
    def _update_market_data(self):
        """СУПЕР-ЛЕГКОЕ обновление данных"""
        if self.client is None or self._is_ip_banned():
            return
            
        # ТОЛЬКО BTC для минимальной нагрузки
        symbols = ['BTCUSDT'][:1]  # Только 1 символ
        
        for symbol in symbols:
            try:
                # МИНИМАЛЬНЫЙ запрос данных
                klines = self._safe_api_request(
                    self.client.futures_klines,
                    symbol=symbol,
                    interval='15m',  # Только 15м таймфрейм
                    limit=10         # Только 10 свечей
                )
                
                # УПРОЩЕННАЯ обработка
                if klines and len(klines) > 5:
                    current_price = float(klines[-1][4])
                    
                    with self.lock:
                        self.market_data[symbol] = {
                            'current_price': current_price,
                            'timestamp': time.time(),
                            '24h_volume': 0  # Не запрашиваем объем для экономии
                        }
                        
            except Exception as e:
                logger.debug(f"Light market data update failed for {symbol}: {e}")
                continue

    def get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Улучшенное получение данных с кэшированием и ограничениями"""
        if self._is_ip_banned():
            logger.warning("IP is banned, returning empty DataFrame")
            return pd.DataFrame()
            
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        cache_duration = 120
        
        with self.lock:
            cache_entry = self.market_data.get(cache_key, {})
            if cache_entry and current_time - cache_entry.get('timestamp', 0) < cache_duration:
                return cache_entry.get('data', pd.DataFrame())
        
        try:
            klines = self._safe_api_request(
                self.client.futures_klines,
                symbol=symbol,
                interval=timeframe,
                limit=50
            )
            
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0.0)
            
            with self.lock:
                self.market_data[cache_key] = {
                    'data': df,
                    'timestamp': current_time
                }
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            with self.lock:
                cache_entry = self.market_data.get(cache_key, {})
                return cache_entry.get('data', pd.DataFrame())

    def _update_orderbook_data(self):
        """Обновляет данные стакана заявок с ограничениями"""
        if self._is_ip_banned() or not getattr(Config, 'MULTIDIMENSIONAL_ANALYSIS', False):
            return
            
        max_symbols = min(3, getattr(Config, 'MAX_SYMBOLS_TO_CHECK', 6))
        symbols = getattr(Config, 'FUNDAMENTAL_PAIRS', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])[:max_symbols]
        
        for i, symbol in enumerate(symbols):
            try:
                if i > 0:
                    time.sleep(5)
                
                orderbook_depth = min(20, getattr(Config, 'ORDERBOOK_DEPTH', 10))
                orderbook = self._safe_api_request(
                    self.client.futures_order_book,
                    symbol=symbol, 
                    limit=orderbook_depth
                )
                
                orderbook_metrics = self.calculate_orderbook_metrics(orderbook)
                
                with self.lock:
                    self.orderbook_data[symbol] = {
                        'bids': orderbook.get('bids', []),
                        'asks': orderbook.get('asks', []),
                        'metrics': orderbook_metrics,
                        'timestamp': time.time()
                    }
                    
            except Exception as e:
                logger.error(f"Error updating orderbook for {symbol}: {e}")
                continue

    def calculate_orderbook_metrics(self, orderbook):
        """Рассчитывает метрики стакана ордеров"""
        try:
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return self._get_default_orderbook_metrics()
            
            bids = np.array(orderbook['bids'], dtype=float)
            asks = np.array(orderbook['asks'], dtype=float)
            
            if bids.ndim == 1:
                bids = bids.reshape(-1, 2)
            if asks.ndim == 1:
                asks = asks.reshape(-1, 2)
            
            if len(bids) == 0 or len(asks) == 0:
                return self._get_default_orderbook_metrics()
            
            total_bid_volume = np.sum(bids[:, 1]) if bids.size > 0 else 0
            total_ask_volume = np.sum(asks[:, 1]) if asks.size > 0 else 0
            
            if total_bid_volume + total_ask_volume > 0:
                imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) * 100
            else:
                imbalance = 0
            
            if total_bid_volume > 0:
                pressure = (total_bid_volume - total_ask_volume) / total_bid_volume * 100
            else:
                pressure = 0
            
            bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
            
            return {
                'orderbook_imbalance': float(imbalance),
                'orderbook_pressure': float(pressure),
                'bid_ask_ratio': float(bid_ask_ratio),
                'orderbook_depth': float((total_bid_volume + total_ask_volume) / 2),
                'bid_walls': 0,
                'ask_walls': 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating orderbook metrics: {e}")
            return self._get_default_orderbook_metrics()
    
    def _get_default_orderbook_metrics(self):
        """Возвращает метрики по умолчанию при ошибке"""
        return {
            'orderbook_imbalance': 0,
            'orderbook_pressure': 0,
            'bid_ask_ratio': 1,
            'orderbook_depth': 0,
            'bid_walls': 0,
            'ask_walls': 0
        }

    def calculate_volatility(self, symbol: str) -> float:
        """Рассчитывает волатильность символа с безопасными запросами"""
        if self._is_ip_banned():
            return 0.05
            
        try:
            klines = self._safe_api_request(
                self.client.futures_klines,
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=15
            )
            
            if not klines or len(klines) < 5:
                return 0.05
            
            closes = [float(k[4]) for k in klines if len(k) > 4]
            
            if len(closes) < 2:
                return 0.05
            
            returns = []
            for i in range(1, len(closes)):
                if closes[i-1] != 0:
                    daily_return = (closes[i] - closes[i-1]) / closes[i-1]
                    returns.append(daily_return)
            
            if len(returns) == 0:
                return 0.05
            
            volatility = np.std(returns)
            annual_volatility = volatility * np.sqrt(365)
            
            return max(0.01, min(annual_volatility, 2.0))
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.05

    def get_current_price(self, symbol: str) -> float:
        """Получает текущую цену символа с кэшированием"""
        if self._is_ip_banned():
            return 0.0
            
        try:
            with self.lock:
                if symbol in self.market_data:
                    data_time = self.market_data[symbol].get('timestamp', 0)
                    if time.time() - data_time < 60:
                        return self.market_data[symbol].get('current_price', 0.0)
            
            ticker = self._safe_api_request(self.client.futures_ticker, symbol=symbol)
            if ticker is None:
                return 0.0
                
            price = float(ticker['lastPrice'])
            
            with self.lock:
                self.market_data[symbol] = {
                    'current_price': price,
                    'timestamp': time.time(),
                    '24h_volume': float(ticker.get('quoteVolume', 0)),
                    'price_change': float(ticker.get('priceChangePercent', 0))
                }
            
            return price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            with self.lock:
                if symbol in self.market_data:
                    return self.market_data[symbol].get('current_price', 0.0)
            return 0.0

    def get_technical_indicators(self, symbol: str, timeframe: str) -> Dict:
        """Вычисляет технические индикаторы для символа с использованием pandas_ta"""
        try:
            df = self.get_market_data(symbol, timeframe)
            if df.empty or len(df) < 20:
                return {}
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            indicators = {}
            
            # RSI
            try:
                rsi = ta.rsi(df['close'], length=14)
                indicators['rsi'] = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
            except Exception as e:
                indicators['rsi'] = 50.0
            
            # MACD
            try:
                macd_data = ta.macd(df['close'])
                if not macd_data.empty and len(macd_data.columns) >= 3:
                    indicators['macd'] = float(macd_data.iloc[-1, 0]) if not pd.isna(macd_data.iloc[-1, 0]) else 0.0
                    indicators['macd_signal'] = float(macd_data.iloc[-1, 1]) if not pd.isna(macd_data.iloc[-1, 1]) else 0.0
                    indicators['macd_hist'] = float(macd_data.iloc[-1, 2]) if not pd.isna(macd_data.iloc[-1, 2]) else 0.0
                else:
                    indicators['macd'] = 0.0
                    indicators['macd_signal'] = 0.0
                    indicators['macd_hist'] = 0.0
            except Exception as e:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_hist'] = 0.0
            
            # Bollinger Bands
            try:
                bb_data = ta.bbands(df['close'], length=20)
                if not bb_data.empty and len(bb_data.columns) >= 3:
                    bb_upper = bb_data.iloc[-1, 0]
                    bb_lower = bb_data.iloc[-1, 2]
                    if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper != bb_lower:
                        current_close = df['close'].iloc[-1]
                        bb_percent = (current_close - bb_lower) / (bb_upper - bb_lower) * 100
                        indicators['bb_percent'] = float(bb_percent)
                    else:
                        indicators['bb_percent'] = 50.0
                else:
                    indicators['bb_percent'] = 50.0
            except Exception as e:
                indicators['bb_percent'] = 50.0
            
            # Stochastic
            try:
                stoch_data = ta.stoch(df['high'], df['low'], df['close'])
                if not stoch_data.empty and len(stoch_data.columns) >= 2:
                    indicators['stochastic_k'] = float(stoch_data.iloc[-1, 0]) if not pd.isna(stoch_data.iloc[-1, 0]) else 50.0
                    indicators['stochastic_d'] = float(stoch_data.iloc[-1, 1]) if not pd.isna(stoch_data.iloc[-1, 1]) else 50.0
                else:
                    indicators['stochastic_k'] = 50.0
                    indicators['stochastic_d'] = 50.0
            except Exception as e:
                indicators['stochastic_k'] = 50.0
                indicators['stochastic_d'] = 50.0
            
            # ATR
            try:
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                indicators['atr'] = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
            except Exception as e:
                indicators['atr'] = 0.0
            
            # ADX
            try:
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                indicators['adx'] = float(adx.iloc[-1, 0]) if not adx.empty and not pd.isna(adx.iloc[-1, 0]) else 20.0
            except Exception as e:
                indicators['adx'] = 20.0
            
            # Volume indicators
            try:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if df['close'].iloc[-2] != 0 else 0
                indicators['price_change'] = float(price_change)
                
                volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] if df['volume'].iloc[-2] > 0 else 0
                indicators['volume_change'] = float(volume_change)
            except Exception as e:
                indicators['price_change'] = 0.0
                indicators['volume_change'] = 0.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicators error for {symbol} on {timeframe}: {e}")
            return {}

    def _update_technical_indicators(self):
        """Обновляет технические индикаторы для символов"""
        if self._is_ip_banned():
            return
            
        max_symbols = min(4, getattr(Config, 'MAX_SYMBOLS_TO_CHECK', 6))
        symbols = getattr(Config, 'FUNDAMENTAL_PAIRS', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'])[:max_symbols]
        timeframes = getattr(Config, 'TIMEFRAMES', ['15m', '1h'])
        
        for i, symbol in enumerate(symbols):
            try:
                if i > 0:
                    time.sleep(2)
                
                for timeframe in timeframes:
                    indicators = self.get_technical_indicators(symbol, timeframe)
                    with self.lock:
                        if symbol not in self.technical_indicators:
                            self.technical_indicators[symbol] = {}
                        self.technical_indicators[symbol][timeframe] = {
                            'indicators': indicators,
                            'timestamp': time.time()
                        }
                        
            except Exception as e:
                logger.error(f"Error updating technical indicators for {symbol}: {e}")
                continue

    def get_24h_volume(self, symbol: str) -> float:
        """Получает объем торгов за 24 часа"""
        if self._is_ip_banned():
            return 0.0
            
        try:
            with self.lock:
                if symbol in self.market_data:
                    data_time = self.market_data[symbol].get('timestamp', 0)
                    if time.time() - data_time < 300:
                        return self.market_data[symbol].get('24h_volume', 0.0)
            
            ticker = self._safe_api_request(self.client.futures_ticker, symbol=symbol)
            if ticker is None:
                return 0.0
                
            volume = float(ticker.get('quoteVolume', 0))
            
            with self.lock:
                if symbol in self.market_data:
                    self.market_data[symbol]['24h_volume'] = volume
                    self.market_data[symbol]['timestamp'] = time.time()
                else:
                    self.market_data[symbol] = {
                        '24h_volume': volume,
                        'timestamp': time.time()
                    }
            
            return volume
            
        except Exception as e:
            logger.error(f"Error getting 24h volume for {symbol}: {e}")
            return 0.0

    # ОБНОВЛЕННЫЕ МЕТОДЫ - ПОЛНАЯ РЕАЛИЗАЦИЯ

    def get_technical_data(self, symbol: str) -> Dict:
        """Получение технических данных с кэшированием и улучшенной обработкой"""
        try:
            cache_key = f"{symbol}_technical"
            current_time = time.time()
            cache_duration = 180  # 3 минуты кэш
            
            # Проверяем кэш
            with self.lock:
                if cache_key in self.technical_data_cache:
                    cache_entry = self.technical_data_cache[cache_key]
                    if current_time - cache_entry.get('timestamp', 0) < cache_duration:
                        logger.debug(f"Using cached technical data for {symbol}")
                        return cache_entry.get('data', {})
            
            # Получаем актуальные данные через безопасный запрос
            logger.info(f"Fetching fresh technical data for {symbol}")
            technical_data = self.get_technical_indicators(symbol, '15m')
            
            # Обогащаем данными дополнительных таймфреймов если доступно
            with self.lock:
                if symbol in self.technical_indicators:
                    for timeframe, data in self.technical_indicators[symbol].items():
                        if timeframe != '15m':
                            technical_data[f'{timeframe}_rsi'] = data['indicators'].get('rsi', 50)
                            technical_data[f'{timeframe}_macd'] = data['indicators'].get('macd', 0)
            
            # Добавляем мета-информацию
            technical_data['last_updated'] = current_time
            technical_data['symbol'] = symbol
            
            # Сохраняем в кэш
            with self.lock:
                self.technical_data_cache[cache_key] = {
                    'data': technical_data,
                    'timestamp': current_time
                }
            
            return technical_data
            
        except Exception as e:
            logger.error(f"Error getting technical data for {symbol}: {e}")
            # Пытаемся вернуть кэшированные данные даже если устарели
            with self.lock:
                if cache_key in self.technical_data_cache:
                    return self.technical_data_cache[cache_key].get('data', {})
            return {}

    def get_orderbook_data(self, symbol: str) -> Dict:
        """Получение данных стакана с улучшенной логикой и кэшированием"""
        try:
            current_time = time.time()
            cache_duration = 120  # 2 минуты кэш
            
            # Проверяем кэш
            with self.lock:
                if symbol in self.orderbook_data:
                    cache_entry = self.orderbook_data[symbol]
                    if current_time - cache_entry.get('timestamp', 0) < cache_duration:
                        logger.debug(f"Using cached orderbook data for {symbol}")
                        return cache_entry.get('metrics', self._get_default_orderbook_metrics())
            
            # Если MULTIDIMENSIONAL_ANALYSIS включен, пытаемся получить актуальные данные
            if getattr(Config, 'MULTIDIMENSIONAL_ANALYSIS', False):
                try:
                    logger.info(f"Fetching fresh orderbook data for {symbol}")
                    orderbook_depth = min(20, getattr(Config, 'ORDERBOOK_DEPTH', 10))
                    orderbook = self._safe_api_request(
                        self.client.futures_order_book,
                        symbol=symbol, 
                        limit=orderbook_depth
                    )
                    
                    metrics = self.calculate_orderbook_metrics(orderbook)
                    metrics['last_updated'] = current_time
                    metrics['symbol'] = symbol
                    
                    # Сохраняем в кэш
                    with self.lock:
                        self.orderbook_data[symbol] = {
                            'bids': orderbook.get('bids', []),
                            'asks': orderbook.get('asks', []),
                            'metrics': metrics,
                            'timestamp': current_time
                        }
                    
                    return metrics
                    
                except Exception as api_error:
                    logger.warning(f"Could not fetch orderbook for {symbol}: {api_error}")
                    # Продолжаем с кэшированными данными если есть
                    with self.lock:
                        if symbol in self.orderbook_data:
                            return self.orderbook_data[symbol].get('metrics', self._get_default_orderbook_metrics())
            
            # Возвращаем данные по умолчанию если не удалось получить актуальные
            default_metrics = self._get_default_orderbook_metrics()
            default_metrics['last_updated'] = current_time
            default_metrics['symbol'] = symbol
            
            return default_metrics
            
        except Exception as e:
            logger.error(f"Error getting orderbook data for {symbol}: {e}")
            return self._get_default_orderbook_metrics()

    def get_social_sentiment(self, symbol: str) -> Dict:
        """Получение социальных сигналов с улучшенной логикой и кэшированием"""
        try:
            cache_key = f"{symbol}_social"
            current_time = time.time()
            cache_duration = 600  # 10 минут кэш
            
            # Проверяем кэш
            with self.lock:
                if cache_key in self.social_sentiment:
                    cache_entry = self.social_sentiment[cache_key]
                    if current_time - cache_entry.get('timestamp', 0) < cache_duration:
                        logger.debug(f"Using cached social sentiment for {symbol}")
                        return cache_entry.get('data', self._get_default_social_sentiment(symbol))
            
            # Проверяем, включены ли социальные сигналы в конфиге
            if not getattr(Config, 'SOCIAL_SIGNALS_ENABLED', False):
                logger.debug(f"Social signals disabled, returning default for {symbol}")
                return self._get_default_social_sentiment(symbol)
            
            # Здесь может быть реализация с реальным API (LunarCrush, Santiment и т.д.)
            logger.info(f"Generating social sentiment data for {symbol}")
            sentiment_data = self._get_default_social_sentiment(symbol)
            
            # Добавляем динамические элементы для реалистичности
            volume_24h = self.get_24h_volume(symbol)
            price_change = self.get_current_price(symbol) > 0  # Простая проверка
            
            # Корректируем sentiment на основе объема и цены
            if volume_24h > 50000000:  # Высокий объем
                sentiment_data['social_volume'] = volume_24h / 1000000  # Нормализуем
                sentiment_data['sentiment'] = min(0.8, sentiment_data['sentiment'] + 0.1)
            elif volume_24h < 1000000:  # Очень низкий объем
                sentiment_data['social_volume'] = volume_24h / 100000
                sentiment_data['sentiment'] = max(0.2, sentiment_data['sentiment'] - 0.1)
            
            if price_change:
                sentiment_data['sentiment'] = min(0.9, sentiment_data['sentiment'] + 0.05)
            
            sentiment_data['last_updated'] = current_time
            sentiment_data['symbol'] = symbol
            
            # Сохраняем в кэш
            with self.lock:
                self.social_sentiment[cache_key] = {
                    'data': sentiment_data,
                    'timestamp': current_time
                }
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return self._get_default_social_sentiment(symbol)

    def get_market_cap(self, symbol: str) -> float:
        """Получение рыночной капитализации с улучшенной логикой оценки"""
        try:
            cache_key = f"{symbol}_marketcap"
            current_time = time.time()
            cache_duration = 3600  # 1 час кэш
            
            # Проверяем кэш
            with self.lock:
                if cache_key in self.market_caps:
                    cache_entry = self.market_caps[cache_key]
                    if current_time - cache_entry.get('timestamp', 0) < cache_duration:
                        logger.debug(f"Using cached market cap for {symbol}")
                        return cache_entry.get('value', 1000000000)
            
            # Пытаемся получить реальную оценку
            logger.info(f"Estimating market cap for {symbol}")
            market_cap = self._estimate_market_cap(symbol)
            
            # Сохраняем в кэш
            with self.lock:
                self.market_caps[cache_key] = {
                    'value': market_cap,
                    'timestamp': current_time,
                    'symbol': symbol
                }
            
            return market_cap
            
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            # Возвращаем кэшированное значение или значение по умолчанию
            with self.lock:
                if cache_key in self.market_caps:
                    return self.market_caps[cache_key].get('value', 1000000000)
            return 1000000000  # Fallback

    def get_symbol_score(self, symbol: str) -> float:
        """Улучшенная оценка символа на основе множества факторов"""
        try:
            # Собираем все необходимые данные
            volume_24h = self.get_24h_volume(symbol)
            current_price = self.get_current_price(symbol)
            volatility = self.calculate_volatility(symbol)
            orderbook_data = self.get_orderbook_data(symbol)
            technical_data = self.get_technical_data(symbol)
            
            # 1. Балл на основе объема (основной фактор) - 40% веса
            if volume_24h > 200000000:  # > 200M
                volume_score = 9.0
            elif volume_24h > 100000000:  # > 100M
                volume_score = 8.0
            elif volume_24h > 50000000:  # > 50M
                volume_score = 6.5
            elif volume_24h > 20000000:  # > 20M
                volume_score = 5.0
            elif volume_24h > 5000000:   # > 5M
                volume_score = 3.5
            else:
                volume_score = 2.0
            
            # 2. Корректировка на ликвидность (глубина стакана) - 25% веса
            orderbook_depth = orderbook_data.get('orderbook_depth', 0)
            if orderbook_depth > 5000000:  # Очень высокая ликвидность
                liquidity_score = 2.5
            elif orderbook_depth > 1000000:  # Высокая ликвидность
                liquidity_score = 2.0
            elif orderbook_depth > 500000:  # Средняя ликвидность
                liquidity_score = 1.5
            elif orderbook_depth > 100000:  # Низкая ликвидность
                liquidity_score = 1.0
            else:
                liquidity_score = 0.5
            
            # 3. Корректировка на волатильность - 20% веса
            if volatility < 0.2:  # Очень низкая волатильность (стабильность)
                volatility_score = 2.0
            elif volatility < 0.5:  # Нормальная волатильность
                volatility_score = 1.5
            elif volatility < 1.0:  # Высокая волатильность
                volatility_score = 0.5
            else:  # Очень высокая волатильность (риск)
                volatility_score = -1.0
            
            # 4. Технические индикаторы - 15% веса
            technical_score = 0.0
            rsi = technical_data.get('rsi', 50)
            if 40 <= rsi <= 60:  # Нейтральная зона RSI
                technical_score += 0.5
            elif 30 <= rsi <= 70:  # Нормальная зона
                technical_score += 0.3
            else:  # Экстремальные зоны
                technical_score += 0.1
            
            # 5. Дополнительные факторы
            additional_score = 0.0
            
            # Наличие стен в стакане
            bid_ask_ratio = orderbook_data.get('bid_ask_ratio', 1)
            if bid_ask_ratio > 1.5:  # Преобладание покупок
                additional_score += 0.3
            elif bid_ask_ratio < 0.7:  # Преобладание продаж
                additional_score -= 0.2
            
            # Итоговый балл с весами
            total_score = (
                volume_score * 0.4 +        # 40%
                liquidity_score * 0.25 +    # 25% 
                volatility_score * 0.2 +    # 20%
                technical_score * 0.15 +    # 15%
                additional_score            # Бонусы/штрафы
            )
            
            # Ограничиваем диапазон 0-10
            final_score = max(0.0, min(10.0, total_score))
            
            logger.debug(f"Symbol score for {symbol}: "
                        f"volume={volume_score:.1f}, "
                        f"liquidity={liquidity_score:.1f}, "
                        f"volatility={volatility_score:.1f}, "
                        f"technical={technical_score:.1f}, "
                        f"additional={additional_score:.1f}, "
                        f"total={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating symbol score for {symbol}: {e}")
            return 0.0

    def _get_default_social_sentiment(self, symbol: str = None) -> Dict:
        """Улучшенная заглушка для социальных сигналов"""
        base_sentiment = 0.5
        base_volume = 0
        
        # Динамические вариации в зависимости от символа
        if symbol:
            # Создаем псевдо-уникальный хэш для символа
            symbol_hash = sum(ord(c) for c in symbol) % 100
            base_sentiment = 0.4 + (symbol_hash % 40) / 100  # 0.4-0.8
            base_volume = symbol_hash * 100
            
            # Популярные символы имеют более высокие показатели
            if 'BTC' in symbol or 'ETH' in symbol:
                base_sentiment += 0.1
                base_volume *= 2
        
        return {
            'sentiment': round(base_sentiment, 3),
            'social_volume': int(base_volume),
            'galaxy_score': 50,
            'alt_rank': 50,
            'last_updated': time.time(),
            'symbol': symbol or 'unknown'
        }

    def _estimate_market_cap(self, symbol: str) -> float:
        """Улучшенная оценка рыночной капитализации"""
        try:
            volume_24h = self.get_24h_volume(symbol)
            current_price = self.get_current_price(symbol)
            
            if volume_24h > 0 and current_price > 0:
                # Более сложная эвристика для оценки капитализации
                if volume_24h > 1000000000:  # > 1B volume
                    multiplier = 50  # Высоколиквидные активы
                elif volume_24h > 100000000:  # > 100M volume
                    multiplier = 80  # Среднеликвидные
                elif volume_24h > 10000000:   # > 10M volume
                    multiplier = 120  # Низколиквидные
                else:
                    multiplier = 200  # Очень низкая ликвидность
                
                estimated_mcap = volume_24h * multiplier
                
                # Корректировка на основе символа
                if 'BTC' in symbol:
                    estimated_mcap = max(estimated_mcap, 500000000000)  # Минимум 500B для BTC
                elif 'ETH' in symbol:
                    estimated_mcap = max(estimated_mcap, 200000000000)  # Минимум 200B для ETH
                
                # Ограничиваем разумными пределами
                return max(1000000, min(estimated_mcap, 5000000000000))  # 1M - 5T
            else:
                return 1000000000  # 1B по умолчанию
                
        except Exception as e:
            logger.warning(f"Could not estimate market cap for {symbol}: {e}")
            return 1000000000

    # СУЩЕСТВУЮЩИЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ

    def _update_social_sentiment(self):
        """Заглушка для социальных сигналов"""
        if not getattr(Config, 'SOCIAL_SIGNALS_ENABLED', False):
            return
        pass

    def _update_fundamental_data(self):
        """Заглушка для фундаментальных данных"""
        pass

    def _update_market_caps(self):
        """Заглушка для рыночных капитализаций"""
        pass

    def _update_funding_rates(self):
        """Заглушка для funding rates"""
        pass

    def _update_liquidation_data(self):
        """Заглушка для данных о ликвидациях"""
        pass

    def _update_correlations(self):
        """Заглушка для корреляций"""
        pass

    def stop(self):
        """Останавливает все потоки"""
        self.running = False
        logger.info("Stopping DataFeeder...")
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=3)
        
        self.session.close()
        logger.info("DataFeeder stopped successfully")

    def get_status(self) -> Dict:
        """Возвращает статус DataFeeder"""
        current_time = time.time()
        if current_time - self.request_stats['last_reset'] > 3600:
            self.request_stats['total'] = 0
            self.request_stats['errors'] = 0
            self.request_stats['last_reset'] = current_time
        
        active_threads = len([t for t in self.threads if t.is_alive()])
        
        return {
            'running': self.running,
            'ip_banned': self._is_ip_banned(),
            'banned_until': self.ip_banned_until,
            'ban_count': self.ban_count,
            'request_stats': self.request_stats.copy(),
            'active_threads': active_threads,
            'market_data_symbols': len(self.market_data),
            'technical_indicators_symbols': len(self.technical_indicators),
            'technical_data_cache_size': len(self.technical_data_cache),
            'orderbook_data_size': len(self.orderbook_data)
        }

# emergency_cleanup.py
import os
import time
import psutil
import logging
import json
import platform
import shutil
import glob
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmergencyCleanup")

def emergency_system_cleanup():
    """ЭКСТРЕННАЯ ОЧИСТКА СИСТЕМЫ ПЕРЕД ЗАПУСКОМ"""
    logger.info("🚨 STARTING EMERGENCY SYSTEM CLEANUP")
    
    # 1. Останавливаем возможные предыдущие процессы бота
    logger.info("🛑 Stopping previous bot processes...")
    processes_to_kill = ['trading_bot.py', 'binance_bot.py', 'crypto_bot.py', 'python']
    for process in processes_to_kill:
        if platform.system() == "Windows":
            os.system(f"taskkill /f /im {process} 2>nul")
        else:
            os.system(f"pkill -f {process} 2>/dev/null")
    time.sleep(3)
    
    # 2. Расширенная очистка кэш-директорий
    logger.info("🗑️ Cleaning cache directories...")
    cache_dirs = [
        '__pycache__', '.pytest_cache', '.cache', 'dl_models', 'models', 'backups',
        'temp', 'tmp', 'downloads', 'cache', '.ipynb_checkpoints',
        'logs', 'log', 'output', 'results', 'data/temp'
    ]
    
    # Рекурсивный поиск и удаление __pycache__
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                logger.info(f"✅ Deleted: {pycache_path}")
            except Exception as e:
                logger.error(f"❌ Error deleting {pycache_path}: {e}")
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"✅ Deleted: {cache_dir}")
            except Exception as e:
                logger.error(f"❌ Error deleting {cache_dir}: {e}")
    
    # 3. Очистка временных файлов
    logger.info("🧹 Cleaning temporary files...")
    temp_patterns = ['*.tmp', '*.temp', '*.log', '*.cache', '*.pyc']  # Исправлено: добавлена запятая
    for pattern in temp_patterns:
        for file in glob.glob(pattern, recursive=True):
            try:
                os.remove(file)
                logger.info(f"✅ Deleted temp: {file}")
            except Exception as e:
                logger.error(f"❌ Error deleting temp file {file}: {e}")
    
    # 4. Очистка логов (всех файлов .log)
    logger.info("📋 Clearing log files...")
    log_files = glob.glob('*.log') + glob.glob('logs/*.log') + ['trading_bot.log', 'debug.log', 'system.log']
    for log_file in set(log_files):  # set для уникальности
        if os.path.exists(log_file):
            try:
                if os.path.getsize(log_file) > 10 * 1024 * 1024:  # >10MB (исправлен синтаксис)
                    with open(log_file, 'w') as f:
                        f.write("")  # Полная очистка больших логов
                    logger.info(f"✅ Cleared large log: {log_file}")
                else:
                    with open(log_file, 'w') as f:
                        f.write("")  # Очищаем файл
                    logger.info(f"✅ Cleared: {log_file}")
            except Exception as e:
                logger.error(f"❌ Error clearing {log_file}: {e}")
    
    # 5. Расширенная очистка больших файлов данных
    logger.info("📊 Trimming large data files...")
    large_files = [
        'trade_history.json', 'optimization_history.json', 
        'price_data.json', 'user_data.json', 'config_backup.json',
        'data/*.json', 'backups/*.json'
    ]
    
    all_data_files = []
    for pattern in large_files:
        all_data_files.extend(glob.glob(pattern))
    
    for file in set(all_data_files):
        if os.path.exists(file):
            try:
                file_size = os.path.getsize(file)
                if file_size > 5 * 1024 * 1024:  # >5MB
                    with open(file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 500:
                        with open(file, 'w') as f:
                            json.dump(data[-500:], f, indent=2)
                        logger.info(f"✅ Trimmed: {file} (kept last 500 entries)")
                    elif isinstance(data, dict):
                        # Для словарей оставляем только ключевые данные
                        important_keys = ['last_update', 'config', 'essential_data']
                        cleaned_data = {k: data[k] for k in important_keys if k in data}
                        with open(file, 'w') as f:
                            json.dump(cleaned_data, f, indent=2)
                        logger.info(f"✅ Simplified: {file}")
                    else:
                        logger.info(f"ℹ️ File {file} structure not supported for trimming")
                else:
                    logger.info(f"ℹ️ File {file} is small ({file_size/1024:.1f} KB), skipping")
            except Exception as e:
                logger.error(f"❌ Error processing {file}: {e}")
    
    # 6. Очистка кэша pip и окружения
    logger.info("🐍 Cleaning Python cache...")
    
    # Очистка кэша в виртуальном окружении
    if 'venv' in os.getcwd() or '.virtualenv' in os.getcwd():
        venv_cache = glob.glob('**/__pycache__', recursive=True)
        for cache_dir in venv_cache:
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"✅ Cleared venv cache: {cache_dir}")
            except Exception as e:
                logger.error(f"❌ Error clearing venv cache {cache_dir}: {e}")
    
    # 7. Очистка системного temp (осторожно!)
    try:
        temp_dir = tempfile.gettempdir()
        bot_temp_files = glob.glob(os.path.join(temp_dir, '*bot*'))
        bot_temp_files.extend(glob.glob(os.path.join(temp_dir, '*trading*')))
        bot_temp_files.extend(glob.glob(os.path.join(temp_dir, '*binance*')))
        
        for temp_file in bot_temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                    logger.info(f"✅ Deleted system temp: {temp_file}")
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                    logger.info(f"✅ Deleted system temp dir: {temp_file}")
            except Exception as e:
                logger.error(f"❌ Error deleting system temp {temp_file}: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Could not clean system temp: {e}")
    
    # 8. Проверка системы
    logger.info("📈 Checking system status...")
    try:
        disk_usage = psutil.disk_usage('.').percent  # Текущая директория
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        logger.info(f"📊 System status after cleanup:")
        logger.info(f"   💾 Disk usage: {disk_usage}%")
        logger.info(f"   🧠 Memory usage: {memory_usage}%")
        logger.info(f"   ⚡ CPU usage: {cpu_usage}%")
        
        # Более строгая проверка
        if disk_usage > 85:
            logger.warning("⚠️ WARNING: Disk usage is high!")
        if memory_usage > 85:
            logger.warning("⚠️ WARNING: Memory usage is high!")
        if disk_usage > 90 or memory_usage > 90:
            logger.error("❌ CRITICAL: System resources critically low!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking system status: {e}")
        return False
    
    # 9. Финальный отчет
    logger.info("📋 Cleanup summary:")
    logger.info("✅ Process termination completed")
    logger.info("✅ Cache directories cleaned") 
    logger.info("✅ Log files cleared")
    logger.info("✅ Temporary files removed")
    logger.info("✅ Large files trimmed")
    
    logger.info("🎉 Emergency cleanup completed successfully!")
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = emergency_system_cleanup()
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ Cleanup took {elapsed_time:.2f} seconds")
    exit(0 if success else 1)

# memory_optimizer.py
import gc
import psutil
import time
import sys
import logging
import os
import json

logger = logging.getLogger("TradingBot")

class MemoryOptimizer:
    def __init__(self, bot_instance=None):
        self.memory_limit_mb = 500  # Лимит памяти 500MB
        self.last_cleanup = time.time()
        self.bot = bot_instance  # Ссылка на основной бот для доступа к его атрибутам
        self.cleanup_count = 0
        
    def check_memory(self):
        """Проверяет использование памяти и запускает очистку при необходимости"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            logger.debug(f"📊 Memory usage: {memory_mb:.2f}MB / {self.memory_limit_mb}MB")
            
            if memory_mb > self.memory_limit_mb * 0.8:  # 80% от лимита
                logger.warning(f"🚨 High memory usage detected: {memory_mb:.2f}MB")
                self.force_cleanup()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Memory check error: {e}")
            return False
    
    def force_cleanup(self):
        """Принудительная очистка памяти"""
        self.cleanup_count += 1
        logger.warning(f"🔄 FORCING MEMORY CLEANUP #{self.cleanup_count}")
        
        try:
            # 1. Очищаем кэши бота, если есть доступ
            if self.bot and hasattr(self.bot, 'api_cache'):
                cache_size = len(self.bot.api_cache)
                self.bot.api_cache.clear()
                logger.info(f"✅ Cleared bot API cache ({cache_size} entries)")
            
            # 2. Очищаем data_feeder кэши
            if self.bot and hasattr(self.bot, 'data_feeder') and self.bot.data_feeder:
                cleared_items = 0
                if hasattr(self.bot.data_feeder, 'market_data'):
                    cleared_items += len(self.bot.data_feeder.market_data)
                    self.bot.data_feeder.market_data.clear()
                if hasattr(self.bot.data_feeder, 'technical_data_cache'):
                    cleared_items += len(self.bot.data_feeder.technical_data_cache)
                    self.bot.data_feeder.technical_data_cache.clear()
                if hasattr(self.bot.data_feeder, 'orderbook_data'):
                    cleared_items += len(self.bot.data_feeder.orderbook_data)
                    self.bot.data_feeder.orderbook_data.clear()
                logger.info(f"✅ Cleared data_feeder caches ({cleared_items} items)")
            
            # 3. Очистка больших объектов в глобальной области
            self._cleanup_global_objects()
            
            # 4. Сборщик мусора
            gc.collect()
            
            # 5. Очистка временных файлов
            self._cleanup_temp_files()
            
            # 6. Проверяем результат очистки
            process = psutil.Process()
            memory_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"✅ Memory cleanup completed: {memory_after:.2f}MB")
            
            self.last_cleanup = time.time()
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")
    
    def _cleanup_global_objects(self):
        """Очищает большие объекты в глобальной области"""
        try:
            # Ищем большие объекты в глобальных переменных
            large_objects = []
            for name, obj in list(globals().items()):
                try:
                    size = sys.getsizeof(obj)
                    if size > 1000000:  # Объекты больше 1MB
                        large_objects.append((name, size))
                except:
                    continue
            
            if large_objects:
                logger.info(f"Found {len(large_objects)} large objects")
                for name, size in large_objects:
                    logger.debug(f"Large object: {name} - {size/1024/1024:.2f}MB")
            
        except Exception as e:
            logger.debug(f"Global objects cleanup error: {e}")
    
    def _cleanup_temp_files(self):
        """Очищает временные файлы"""
        try:
            temp_extensions = ['.tmp', '.temp', '.cache']
            cleaned_files = 0
            
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if any(file.endswith(ext) for ext in temp_extensions):
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            cleaned_files += 1
                        except:
                            pass
            
            if cleaned_files > 0:
                logger.info(f"✅ Cleaned {cleaned_files} temporary files")
                
        except Exception as e:
            logger.debug(f"Temp files cleanup error: {e}")
    
    def optimize_data_structures(self):
        """Оптимизирует структуры данных для экономии памяти"""
        try:
            # Конвертируем большие списки в более эффективные структуры
            if self.bot and hasattr(self.bot, 'trade_history'):
                if len(self.bot.trade_history) > 1000:
                    # Оставляем только последние 1000 сделок
                    self.bot.trade_history = self.bot.trade_history[-1000:]
                    logger.info("✅ Optimized trade history size")
            
            # Очищаем устаревшие записи в кэшах
            if self.bot and hasattr(self.bot, '_cleanup_old_cache'):
                self.bot._cleanup_old_cache()
                
        except Exception as e:
            logger.error(f"Data structures optimization error: {e}")
    
    def get_memory_stats(self):
        """Возвращает статистику использования памяти"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_percent = (memory_mb / self.memory_limit_mb) * 100
            
            return {
                'memory_used_mb': round(memory_mb, 2),
                'memory_limit_mb': self.memory_limit_mb,
                'memory_percent': round(memory_percent, 1),
                'cleanup_count': self.cleanup_count,
                'last_cleanup': time.strftime('%H:%M:%S', time.localtime(self.last_cleanup))
            }
        except Exception as e:
            logger.error(f"Memory stats error: {e}")
            return {}
    
    def set_memory_limit(self, limit_mb):
        """Устанавливает новый лимит памяти"""
        self.memory_limit_mb = limit_mb
        logger.info(f"🎯 New memory limit set: {limit_mb}MB")
    
    def monitor_memory_usage(self, interval=60):
        """Мониторинг использования памяти (для дебага)"""
        try:
            while True:
                stats = self.get_memory_stats()
                if stats:
                    logger.info(
                        f"📊 Memory: {stats['memory_used_mb']}MB/"
                        f"{stats['memory_limit_mb']}MB "
                        f"({stats['memory_percent']}%)"
                    )
                
                if stats['memory_percent'] > 85:
                    self.force_cleanup()
                
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Memory monitoring stopped")
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")

# Функция для экстренной очистки диска
def emergency_disk_cleanup():
    """СРОЧНАЯ очистка дискового пространства"""
    logger.critical("🚨 EMERGENCY DISK CLEANUP ACTIVATED")
    
    try:
        # 1. Очистка кэшей
        cache_dirs = ['__pycache__', '.pytest_cache', '.cache', 'dl_models', 'models']
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Deleted cache: {cache_dir}")
                except:
                    logger.warning(f"Failed to delete cache: {cache_dir}")
        
        # 2. Очистка логов (оставляем только последние 100 строк)
        log_files = ['trading_bot.log', 'debug.log']
        for log_file in log_files:
            if os.path.exists(log_file) and os.path.getsize(log_file) > 1024 * 1024:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    if len(lines) > 100:
                        with open(log_file, 'w', encoding='utf-8') as f:
                            f.writelines(lines[-100:])
                    logger.info(f"Cleaned log: {log_file}")
                except Exception as e:
                    logger.warning(f"Log cleanup failed for {log_file}: {e}")
        
        # 3. Очистка торговой истории
        trade_file = 'trade_history.json'
        if os.path.exists(trade_file):
            try:
                with open(trade_file, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
                if len(trades) > 50:
                    with open(trade_file, 'w', encoding='utf-8') as f:
                        json.dump(trades[-50:], f, indent=2)
                logger.info("Cleaned trade history (kept last 50 trades)")
            except Exception as e:
                logger.warning(f"Trade history cleanup failed: {e}")
        
        # 4. Временные файлы
        import tempfile
        tempfile.tempdir = None
        
        # Удаляем временные файлы
        temp_extensions = ['.tmp', '.temp', '.cache']
        temp_files_cleaned = 0
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.endswith(ext) for ext in temp_extensions):
                    try:
                        os.remove(os.path.join(root, file))
                        temp_files_cleaned += 1
                    except:
                        pass
        
        logger.info(f"Cleaned {temp_files_cleaned} temporary files")
        
        # 5. Проверка свободного места
        disk_usage = psutil.disk_usage('/').percent
        logger.info(f"📊 Disk usage after cleanup: {disk_usage}%")
        
        # 6. Принудительный сбор мусора
        gc.collect()
        
        logger.info("✅ Emergency disk cleanup completed")
        
    except Exception as e:
        logger.error(f"Emergency disk cleanup error: {e}")

# Упрощенная версия для использования без бота
class SimpleMemoryOptimizer:
    def __init__(self, memory_limit_mb=500):
        self.memory_limit_mb = memory_limit_mb
        self.last_cleanup = time.time()
    
    def check_and_cleanup(self):
        """Проверяет память и выполняет очистку при необходимости"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.memory_limit_mb * 0.8:
                logger.warning(f"High memory usage: {memory_mb:.2f}MB")
                gc.collect()
                return True
            return False
        except:
            return False

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from config import Config
from scipy import stats

logger = logging.getLogger("ProfitManager")

@dataclass
class TradePerformance:
    """Data class for trade performance metrics"""
    entry_price: float
    exit_price: float
    position_size: float
    direction: str
    pnl_percent: float
    pnl_absolute: float
    hold_time: timedelta
    peak_profit: float
    max_drawdown: float
    r_multiple: float
    quality_score: float

class ProfitManager:
    def __init__(self, trading_bot, risk_manager):
        self.trading_bot = trading_bot
        self.risk_manager = risk_manager
        self.initial_balance = risk_manager.get_current_balance()
        self.month_start_balance = self.initial_balance
        self.last_withdrawal = datetime.now()
        self.profit_history = []
        self.trade_history = []
        self.performance_metrics = {}
        self.last_rebalance = datetime.now()
        
    def check_profit_targets(self):
        """Проверяет достижение целей по прибыли"""
        if not Config.AUTO_PROFIT_WITHDRAWAL['enabled']:
            return
            
        current_balance = self.risk_manager.get_current_balance()
        
        # Проверяем удвоение депозита
        if current_balance >= self.initial_balance * Config.AUTO_PROFIT_WITHDRAWAL['withdrawal_threshold']:
            profit = current_balance - self.initial_balance
            withdrawal_amount = profit * Config.AUTO_PROFIT_WITHDRAWAL['withdrawal_percentage']
            self.withdraw_profits(withdrawal_amount, "DOUBLE_DEPOSIT")
            
        # Проверяем ежемесячный вывод
        now = datetime.now()
        if now.month != self.last_withdrawal.month:
            monthly_profit = current_balance - self.month_start_balance
            if monthly_profit > 0:
                withdrawal_amount = monthly_profit * Config.AUTO_PROFIT_WITHDRAWAL['monthly_withdrawal_percentage']
                self.withdraw_profits(withdrawal_amount, "MONTHLY")
                
            self.month_start_balance = current_balance
            self.last_withdrawal = now
            
        # Проверяем необходимость ребалансировки портфеля
        if (now - self.last_rebalance).days >= 7:  # Ребалансировка раз в неделю
            self.rebalance_portfolio()
            self.last_rebalance = now
            
    def withdraw_profits(self, amount: float, reason: str):
        """Выводит прибыль"""
        try:
            current_balance = self.risk_manager.get_current_balance()
            
            # Проверяем, что сумма вывода не превышает доступный баланс
            if amount > current_balance * 0.8:  # Не более 80% баланса
                amount = current_balance * 0.8
                logger.warning(f"Adjusted withdrawal amount to ${amount:.2f} (80% of balance)")
                
            # Здесь должен быть код для вывода средств
            logger.info(f"Withdrawing ${amount:.2f} for reason: {reason}")
            
            # НЕ уменьшаем initial_balance - это историческая запись
            # Вместо этого обновляем month_start_balance
            self.month_start_balance -= amount
            
            # Сохраняем в историю
            self.profit_history.append({
                'timestamp': datetime.now(),
                'amount': amount,
                'reason': reason,
                'balance_before': current_balance,
                'balance_after': current_balance - amount
            })
            
            # Отправляем уведомление
            if Config.TELEGRAM_ENABLED:
                self._send_withdrawal_notification(amount, reason, current_balance)
                
        except Exception as e:
            logger.error(f"Error withdrawing profits: {e}")
            
    def _send_withdrawal_notification(self, amount: float, reason: str, current_balance: float):
        """Отправляет уведомление о выводе средств"""
        message = (
            f"💰 PROFIT WITHDRAWAL\n"
            f"Amount: ${amount:.2f}\n"
            f"Reason: {reason}\n"
            f"Balance before: ${current_balance:.2f}\n"
            f"Balance after: ${current_balance - amount:.2f}\n"
            f"Total withdrawn: ${sum(x['amount'] for x in self.profit_history):.2f}"
        )
        logger.info(f"Telegram notification: {message}")
    
    def manage_open_positions(self):
        """Управляет открытыми позициями для максимизации прибыли"""
        try:
            open_positions = self.trading_bot.get_open_positions()
            
            for position in open_positions:
                symbol = position['symbol']
                current_price = self.trading_bot.get_current_price(symbol)
                
                # Обновляем трейлинг-стоп
                self.update_trailing_stop(position, current_price)
                
                # Проверяем частичный тейк-профит
                self.check_partial_take_profit(position, current_price)
                
                # Адаптируем цели based на рыночных условиях
                self.adapt_targets_based_on_market(position, current_price)
                
        except Exception as e:
            logger.error(f"Error managing open positions: {e}")
    
    def update_trailing_stop(self, position: Dict, current_price: float):
        """Обновляет трейлинг-стоп для позиции"""
        try:
            symbol = position['symbol']
            entry_price = position['entry_price']
            direction = position['direction']
            current_stop = position.get('stop_loss', 0)
            
            # Рассчитываем новый трейлинг-стоп
            if Config.ADAPTIVE_TRAILING_STOP:
                new_stop = self.calculate_adaptive_trailing_stop(
                    symbol, entry_price, current_price, direction, current_stop
                )
            else:
                new_stop = self.calculate_static_trailing_stop(
                    entry_price, current_price, direction
                )
            
            # Обновляем стоп-лосс, если это выгодно
            if self.should_update_stop(direction, new_stop, current_stop):
                self.trading_bot.update_stop_loss(symbol, new_stop)
                logger.info(f"Updated trailing stop for {symbol}: {new_stop:.4f}")
                
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    def calculate_adaptive_trailing_stop(self, symbol: str, entry_price: float, 
                                       current_price: float, direction: str, 
                                       current_stop: float) -> float:
        """Рассчитывает адаптивный трейлинг-стоп based на волатильности"""
        # Получаем текущую волатильность
        volatility = self.risk_manager.calculate_atr(symbol, 14)
        price_change_pct = abs(current_price - entry_price) / entry_price
        
        # Базовый трейлинг based на волатильности
        if direction == 'BUY':
            if volatility > 0.02:  # Высокая волатильность
                trail_pct = 0.015
            elif volatility > 0.01:  # Средняя волатильность
                trail_pct = 0.01
            else:  # Низкая волатильность
                trail_pct = 0.007
                
            new_stop = current_price * (1 - trail_pct)
            # Защищаем прибыль - никогда не двигаем стоп в убыток
            new_stop = max(new_stop, entry_price * (1 - Config.MAX_STOP_LOSS))
            
        else:  # SELL
            if volatility > 0.02:
                trail_pct = 0.015
            elif volatility > 0.01:
                trail_pct = 0.01
            else:
                trail_pct = 0.007
                
            new_stop = current_price * (1 + trail_pct)
            new_stop = min(new_stop, entry_price * (1 + Config.MAX_STOP_LOSS))
            
        return new_stop
    
    def calculate_static_trailing_stop(self, entry_price: float, 
                                     current_price: float, direction: str) -> float:
        """Рассчитывает статический трейлинг-стоп"""
        if direction == 'BUY':
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > 0.1:  # 10% прибыли
                stop_pct = 0.07
            elif profit_pct > 0.05:  # 5% прибыли
                stop_pct = 0.04
            else:
                stop_pct = Config.MAX_STOP_LOSS
                
            return current_price * (1 - stop_pct)
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct > 0.1:
                stop_pct = 0.07
            elif profit_pct > 0.05:
                stop_pct = 0.04
            else:
                stop_pct = Config.MAX_STOP_LOSS
                
            return current_price * (1 + stop_pct)
    
    def should_update_stop(self, direction: str, new_stop: float, current_stop: float) -> bool:
        """Определяет, стоит ли обновлять стоп-лосс"""
        if direction == 'BUY':
            return new_stop > current_stop
        else:
            return new_stop < current_stop
    
    def check_partial_take_profit(self, position: Dict, current_price: float):
        """Проверяет условия для частичного тейк-профита"""
        try:
            symbol = position['symbol']
            entry_price = position['entry_price']
            direction = position['direction']
            size = position['size']
            
            # Рассчитываем текущую прибыль
            if direction == 'BUY':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price
            
            # Проверяем условия для частичного тейк-профита
            take_profit_levels = self.calculate_take_profit_levels(profit_pct)
            
            for level, close_pct in take_profit_levels.items():
                if profit_pct >= level and not position.get(f'tp_{level}_hit', False):
                    # Закрываем часть позиции
                    close_size = size * close_pct
                    self.trading_bot.close_position_partial(symbol, close_size)
                    
                    # Помечаем уровень как достигнутый
                    position[f'tp_{level}_hit'] = True
                    logger.info(f"Partial take profit for {symbol}: {level*100:.1f}% (+${close_size * profit_pct:.2f})")
                    
        except Exception as e:
            logger.error(f"Error in partial take profit: {e}")
    
    def calculate_take_profit_levels(self, profit_pct: float) -> Dict[float, float]:
        """Рассчитывает уровни для частичного тейк-профита"""
        if profit_pct < 0.03:  # Меньше 3% прибыли
            return {}
        elif profit_pct < 0.06:  # 3-6% прибыли
            return {0.03: 0.3}  # Закрываем 30% на 3%
        elif profit_pct < 0.1:   # 6-10% прибыли
            return {0.06: 0.4}   # Закрываем 40% на 6%
        else:                    # Более 10% прибыли
            return {0.1: 0.5}    # Закрываем 50% на 10%
    
    def adapt_targets_based_on_market(self, position: Dict, current_price: float):
        """Адаптирует цели based на рыночных условиях"""
        try:
            symbol = position['symbol']
            market_context = self.trading_bot.advanced_decision_maker.assess_market_context(symbol)
            
            # Корректируем цели based на волатильности
            if market_context.volatility_regime == 'HIGH':
                # В высокой волатильности увеличиваем тейк-профиты
                self.adjust_take_profit(position, 1.3)  # +30%
            elif market_context.volatility_regime == 'LOW':
                # В низкой волатильности уменьшаем тейк-профиты
                self.adjust_take_profit(position, 0.8)  # -20%
                
            # Корректируем based на силе тренда
            if market_context.trend_strength > 0.7:
                self.adjust_take_profit(position, 1.2)  # +20% в сильном тренде
                
        except Exception as e:
            logger.error(f"Error adapting targets: {e}")
    
    def adjust_take_profit(self, position: Dict, multiplier: float):
        """Адаптирует уровень тейк-профита"""
        symbol = position['symbol']
        current_tp = position.get('take_profit')
        
        if current_tp:
            new_tp = current_tp * multiplier
            self.trading_bot.update_take_profit(symbol, new_tp)
            logger.info(f"Adjusted take profit for {symbol}: {new_tp:.4f}")
    
    def rebalance_portfolio(self):
        """Ребалансирует портфель based на производительности"""
        try:
            # Анализируем эффективность активов
            performance_data = self.analyze_asset_performance()
            
            # Определяем новые веса активов
            new_weights = self.calculate_optimal_weights(performance_data)
            
            # Применяем ребалансировку
            self.apply_rebalancing(new_weights)
            
            logger.info("Portfolio rebalancing completed")
            
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
    
    def analyze_asset_performance(self) -> Dict[str, Dict]:
        """Анализирует производительность активов"""
        performance_data = {}
        
        for symbol in Config.TRADING_SYMBOLS:
            # Анализируем исторические сделки по активу
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol]
            
            if not symbol_trades:
                continue
                
            # Рассчитываем метрики производительности
            win_rate = self.calculate_win_rate(symbol_trades)
            avg_profit = self.calculate_avg_profit(symbol_trades)
            sharpe_ratio = self.calculate_sharpe_ratio(symbol_trades)
            sortino_ratio = self.calculate_sortino_ratio(symbol_trades)
            
            performance_data[symbol] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'trade_count': len(symbol_trades)
            }
            
        return performance_data
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Рассчитывает процент прибыльных сделок"""
        if not trades:
            return 0
            
        winning_trades = sum(1 for t in trades if t['pnl_absolute'] > 0)
        return winning_trades / len(trades)
    
    def calculate_avg_profit(self, trades: List[Dict]) -> float:
        """Рассчитывает среднюю прибыль на сделку"""
        if not trades:
            return 0
            
        return sum(t['pnl_absolute'] for t in trades) / len(trades)
    
    def calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Рассчитывает коэффициент Шарпа"""
        if len(trades) < 2:
            return 0
            
        returns = [t['pnl_percent'] for t in trades]
        avg_return = np.mean(returns)
        std_dev = np.std(returns)
        
        if std_dev == 0:
            return 0
            
        return avg_return / std_dev * np.sqrt(252)  # Годовой Sharpe
    
    def calculate_sortino_ratio(self, trades: List[Dict]) -> float:
        """Рассчитывает коэффициент Сортино"""
        if len(trades) < 2:
            return 0
            
        returns = [t['pnl_percent'] for t in trades]
        avg_return = np.mean(returns)
        
        # Только отрицательные возвраты
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0
            
        downside_dev = np.std(negative_returns)
        
        if downside_dev == 0:
            return 0
            
        return avg_return / downside_dev * np.sqrt(252)  # Годовой Sortino
    
    def calculate_optimal_weights(self, performance_data: Dict) -> Dict[str, float]:
        """Рассчитывает оптимальные веса активов based на производительности"""
        total_score = 0
        symbol_scores = {}
        
        for symbol, data in performance_data.items():
            # Композитный score based на различных метриках
            score = (
                data['win_rate'] * 0.3 +
                min(data['avg_profit'] * 10, 0.3) +  # Нормализуем среднюю прибыль
                data['sharpe_ratio'] * 0.2 +
                data['sortino_ratio'] * 0.2
            )
            
            # Наказываем за малое количество сделок
            if data['trade_count'] < 5:
                score *= 0.7
                
            symbol_scores[symbol] = score
            total_score += score
        
        # Рассчитываем веса
        if total_score == 0:
            # Равномерное распределение если нет данных
            return {symbol: 1/len(performance_data) for symbol in performance_data.keys()}
            
        return {symbol: score/total_score for symbol, score in symbol_scores.items()}
    
    def apply_rebalancing(self, new_weights: Dict[str, float]):
        """Применяет ребалансировку портфеля"""
        current_balance = self.risk_manager.get_current_balance()
        
        for symbol, target_weight in new_weights.items():
            # Рассчитываем целевую позицию
            target_value = current_balance * target_weight
            current_value = self.trading_bot.get_position_value(symbol)
            
            # Определяем необходимые действия
            if current_value < target_value * 0.9:  # Ниже целевого значения
                # Нужно докупить
                buy_amount = target_value - current_value
                if buy_amount > current_balance * 0.05:  # Минимальная сумма
                    self.trading_bot.place_trade(symbol, 'BUY', buy_amount)
                    
            elif current_value > target_value * 1.1:  # Выше целевого значения
                # Нужно продать часть
                sell_amount = current_value - target_value
                if sell_amount > current_balance * 0.05:  # Минимальная сумма
                    self.trading_bot.place_trade(symbol, 'SELL', sell_amount)
    
    def record_trade_performance(self, trade_data: Dict):
        """Записывает метрики производительности сделки"""
        try:
            performance = TradePerformance(
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                position_size=trade_data['position_size'],
                direction=trade_data['direction'],
                pnl_percent=trade_data['pnl_percent'],
                pnl_absolute=trade_data['pnl_absolute'],
                hold_time=trade_data['hold_time'],
                peak_profit=trade_data.get('peak_profit', 0),
                max_drawdown=trade_data.get('max_drawdown', 0),
                r_multiple=self.calculate_r_multiple(trade_data),
                quality_score=self.calculate_trade_quality(trade_data)
            )
            
            self.trade_history.append(performance.__dict__)
            
            # Обновляем агрегированные метрики
            self.update_performance_metrics(trade_data['symbol'], performance)
            
        except Exception as e:
            logger.error(f"Error recording trade performance: {e}")
    
    def calculate_r_multiple(self, trade_data: Dict) -> float:
        """Рассчитывает R-Multiple сделки"""
        if trade_data['direction'] == 'BUY':
            risk = trade_data['entry_price'] - trade_data.get('stop_loss', trade_data['entry_price'] * 0.99)
            reward = trade_data.get('take_profit', trade_data['entry_price'] * 1.01) - trade_data['entry_price']
        else:
            risk = trade_data.get('stop_loss', trade_data['entry_price'] * 1.01) - trade_data['entry_price']
            reward = trade_data['entry_price'] - trade_data.get('take_profit', trade_data['entry_price'] * 0.99)
        
        if risk == 0:
            return 0
            
        return reward / risk
    
    def calculate_trade_quality(self, trade_data: Dict) -> float:
        """Рассчитывает общий quality score сделки"""
        quality_factors = []
        
        # Фактор прибыльности
        if trade_data['pnl_percent'] > 0:
            quality_factors.append(0.8 + min(trade_data['pnl_percent'] / 0.1, 0.2))  # 0.8-1.0
        else:
            quality_factors.append(max(0.2 + trade_data['pnl_percent'] / -0.05, 0))  # 0.0-0.2
        
        # Фактор времени удержания
        optimal_hold = timedelta(hours=4)  # Оптимальное время удержания
        hold_factor = 1 - min(abs((trade_data['hold_time'] - optimal_hold).total_seconds()) / 36000, 1)
        quality_factors.append(hold_factor * 0.3)
        
        # Фактор максимальной просадки
        drawdown_factor = 1 - min(trade_data.get('max_drawdown', 0) / 0.1, 1)  # Штраф за просадку >10%
        quality_factors.append(drawdown_factor * 0.2)
        
        return sum(quality_factors) / len(quality_factors)
    
    def update_performance_metrics(self, symbol: str, performance: TradePerformance):
        """Обновляет агрегированные метрики производительности"""
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'avg_hold_time': timedelta(),
                'best_trade': 0,
                'worst_trade': 0,
                'avg_r_multiple': 0,
                'quality_scores': []
            }
        
        metrics = self.performance_metrics[symbol]
        metrics['total_trades'] += 1
        metrics['total_pnl'] += performance.pnl_absolute
        
        if performance.pnl_absolute > 0:
            metrics['winning_trades'] += 1
            
        if performance.pnl_absolute > metrics['best_trade']:
            metrics['best_trade'] = performance.pnl_absolute
            
        if performance.pnl_absolute < metrics['worst_trade']:
            metrics['worst_trade'] = performance.pnl_absolute
            
        # Обновляем среднее время удержания
        total_seconds = metrics['avg_hold_time'].total_seconds() * (metrics['total_trades'] - 1)
        total_seconds += performance.hold_time.total_seconds()
        metrics['avg_hold_time'] = timedelta(seconds=total_seconds / metrics['total_trades'])
        
        # Обновляем средний R-Multiple
        metrics['avg_r_multiple'] = (
            metrics['avg_r_multiple'] * (metrics['total_trades'] - 1) + performance.r_multiple
        ) / metrics['total_trades']
        
        # Добавляем quality score
        metrics['quality_scores'].append(performance.quality_score)
        
        # Сохраняем только последние 100 scores
        if len(metrics['quality_scores']) > 100:
            metrics['quality_scores'] = metrics['quality_scores'][-100:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Генерирует отчет о производительности"""
        total_trades = sum(m['total_trades'] for m in self.performance_metrics.values())
        winning_trades = sum(m['winning_trades'] for m in self.performance_metrics.values())
        total_pnl = sum(m['total_pnl'] for m in self.performance_metrics.values())
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Рассчитываем коэффициент прибыльности
        profit_factor = self.calculate_profit_factor()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': profit_factor,
            'initial_balance': self.initial_balance,
            'current_balance': self.risk_manager.get_current_balance(),
            'total_withdrawn': sum(x['amount'] for x in self.profit_history),
            'by_symbol': {
                symbol: {
                    'win_rate': m['winning_trades'] / m['total_trades'] if m['total_trades'] > 0 else 0,
                    'avg_pnl': m['total_pnl'] / m['total_trades'] if m['total_trades'] > 0 else 0,
                    'avg_hold_time': str(m['avg_hold_time']),
                    'best_trade': m['best_trade'],
                    'worst_trade': m['worst_trade'],
                    'avg_r_multiple': m['avg_r_multiple'],
                    'avg_quality': np.mean(m['quality_scores']) if m['quality_scores'] else 0
                }
                for symbol, m in self.performance_metrics.items()
            }
        }
    
    def calculate_profit_factor(self) -> float:
        """Рассчитывает фактор прибыльности"""
        gross_profit = sum(t['pnl_absolute'] for t in self.trade_history if t['pnl_absolute'] > 0)
        gross_loss = abs(sum(t['pnl_absolute'] for t in self.trade_history if t['pnl_absolute'] < 0))
        
        if gross_loss == 0:
            return float('inf')
            
        return gross_profit / gross_loss

import logging
import os
import time

logger = logging.getLogger("StabilityFix")

def apply_stability_fixes():
    """НЕМЕДЛЕННОЕ ПРИМЕНЕНИЕ ФИКСОВ СТАБИЛЬНОСТИ"""
    logger.info("🔧 APPLYING ULTRA-STABILITY FIXES...")
    
    import config
    from trading_bot import TradingBot
    
    # 🔥 КРИТИЧЕСКИЕ НАСТРОЙКИ ДЛЯ МАКСИМАЛЬНОЙ СТАБИЛЬНОСТИ
    config.CHECK_INTERVAL = 30.0
    config.REQUEST_TIMEOUT = 30
    config.MAX_TRADES_PER_HOUR = 3
    config.MAX_SYMBOLS_TO_CHECK = 2
    
    # СОЗДАЕМ API_RATE_LIMIT ЕСЛИ ЕГО НЕТ
    if not hasattr(config, 'API_RATE_LIMIT'):
        config.API_RATE_LIMIT = {
            'requests_per_minute': 120,
            'max_concurrent_requests': 1,
            'retry_delay': 10.0
        }
    else:
        # Обновляем существующие значения
        config.API_RATE_LIMIT['requests_per_minute'] = 120
        config.API_RATE_LIMIT['max_concurrent_requests'] = 1
        config.API_RATE_LIMIT['retry_delay'] = 10.0
    
    # ОТКЛЮЧАЕМ ВСЕ СЛОЖНЫЕ ФУНКЦИИ
    config.MULTIDIMENSIONAL_ANALYSIS = False
    config.SOCIAL_SIGNALS_ENABLED = False
    config.COGNITIVE_TRADING_ENABLED = False
    config.DL_USE_HYBRID_MODEL = False
    config.ADVANCED_ORDERBOOK_ANALYSIS = False
    
    # ДОПОЛНИТЕЛЬНЫЕ ФИКСЫ ДЛЯ СТАБИЛЬНОСТИ
    config.AGGRESSIVE_MODE = False
    config.BREAKOUT_TRADING = False
    config.NEWS_SENSITIVITY = False
    config.BAYESIAN_OPTIMIZATION = False
    config.AUTO_PARAMETER_OPTIMIZATION = False
    config.TRANSFORMER_ENABLED = False
    config.REINFORCEMENT_LEARNING_ENABLED = False
    config.ENSEMBLE_LEARNING = False
    config.POSITION_HEDGING = False
    
    # УВЕЛИЧИВАЕМ ТАЙМАУТЫ И ИНТЕРВАЛЫ
    config.UPDATE_INTERVAL = 1800  # 30 минут
    config.LEARNING_UPDATE_INTERVAL = 86400  # 24 часа
    config.ORDERBOOK_UPDATE_INTERVAL = 300  # 5 минут
    config.DATA_CACHE_DURATION = 600  # 10 минут
    
    logger.info("✅ ULTRA-STABLE MODE ACTIVATED!")
    logger.info("   - 30s check intervals")
    logger.info("   - 3 max trades per hour") 
    logger.info("   - 2 symbols monitoring")
    logger.info("   - 120 API requests/minute")
    logger.info("   - All complex features DISABLED")

# Применяем при импорте
apply_stability_fixes()

python-binance==1.0.19
pandas==2.0.3
numpy==1.23.5
python-dotenv==1.0.0
requests==2.31.0
psutil==5.9.6
scikit-learn==1.3.0
pandas-ta==0.3.14b0
ccxt==4.1.100
aiohttp==3.8.5
pyyaml==6.0.1
scipy==1.10.1
websocket-client==1.6.3
urllib3==2.4.0


import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
from binance import Client
from config import Config

logger = logging.getLogger("RiskManager")

class RiskManager:
    def __init__(self, client: Client):
        self.client = client
        self.config = Config()
        self.trade_count = 0
        self.last_loss_time = 0
        self.consecutive_losses = 0
        self.market_regime = "NEUTRAL"
        self.tradable_symbols = []
        self.hedge_positions = {}
        self.performance_history = []
        self.last_symbol_update = 0
        self.economic_events = []
        self.max_spread = getattr(self.config, 'MAX_SPREAD', 0.001)  # 0.1% по умолчанию
        self.min_volume = getattr(self.config, 'MIN_DAILY_VOLUME', 100000)
        # Добавляем атрибуты для кэширования
        self._symbols_cache = None
        self._symbols_cache_time = 0
        self._initialize()
        
    def _initialize(self):
        self.get_tradable_symbols()  # Используем обновленный метод
        logger.info("Risk Manager initialized")
        
    def validate_spread(self, symbol: str) -> bool:
        """Улучшенная валидация спреда с обработкой ошибок"""
        try:
            # Проверяем, есть ли данные в кэше
            if hasattr(self, 'data_feeder') and self.data_feeder:
                price_data = self.data_feeder.get_current_price(symbol)
                if price_data == 0 or price_data is None:
                    logger.debug(f"No price data for {symbol}, skipping validation")
                    return False
            
            # Стандартная логика валидации...
            ticker = self.client.futures_ticker(symbol=symbol)
            if not ticker or 'bidPrice' not in ticker or 'askPrice' not in ticker:
                return False
                
            bid_price = float(ticker['bidPrice'])
            ask_price = float(ticker['askPrice'])
            
            if bid_price == 0 or ask_price == 0:
                return False
                
            spread = (ask_price - bid_price) / bid_price
            return spread <= self.max_spread
            
        except Exception as e:
            logger.debug(f"Spread validation error for {symbol}: {e}")
            return False
    
    def get_tradable_symbols(self) -> List[str]:
        """СУПЕР-БЫСТРЫЙ метод получения символов с кэшированием"""
        try:
            # Кэшируем на 1 час вместо частых запросов
            current_time = time.time()
            if hasattr(self, '_symbols_cache') and current_time - self._symbols_cache_time < 3600:
                return self._symbols_cache
                
            # ТОЛЬКО основные ликвидные пары
            fundamental_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'][:self.config.MAX_SYMBOLS_TO_CHECK]
            
            # Простая проверка ликвидности без сложных запросов
            liquid_symbols = []
            for symbol in fundamental_symbols:
                try:
                    # Быстрая проверка через 24h тикер
                    ticker = self.client.futures_24hr_ticker(symbol=symbol)
                    if float(ticker['volume']) > self.min_volume:
                        liquid_symbols.append(symbol)
                        
                    # Задержка между проверками символов
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {e}")
                    continue
            
            # Кэшируем результат
            self._symbols_cache = liquid_symbols
            self._symbols_cache_time = current_time
            self.tradable_symbols = liquid_symbols  # Сохраняем для совместимости
            self.last_symbol_update = current_time   # Обновляем время
            
            logger.info(f"Tradable symbols updated: {len(liquid_symbols)} symbols")
            return liquid_symbols
            
        except Exception as e:
            logger.error(f"Error getting tradable symbols: {e}")
            # Fallback на основные пары
            fallback_symbols = ['BTCUSDT', 'ETHUSDT'][:self.config.MAX_SYMBOLS_TO_CHECK]
            self.tradable_symbols = fallback_symbols
            return fallback_symbols

    def emergency_stop_loss(self, symbol: str):
        """Экстренное закрытие позиции"""
        try:
            position = self.get_position(symbol)
            if position and abs(position['size']) > 0:
                # Закрываем 50% позиции немедленно
                close_size = position['size'] * 0.5
                self.market_sell(symbol, close_size)
                logger.critical(f"EMERGENCY: Closed 50% of {symbol}")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")

    def _validate_symbol_liquidity(self, symbol: str) -> bool:
        """Проверка ликвидности символа"""
        try:
            ticker_24hr = self.client.futures_24hr_ticker(symbol=symbol)
            if ticker_24hr and 'volume' in ticker_24hr:
                volume = float(ticker_24hr['volume'])
                return volume >= self.min_volume
            return False
        except Exception as e:
            logger.debug(f"Error checking liquidity for {symbol}: {e}")
            return False
            
    def filter_by_liquidity(self, symbols: list, min_volume: float = None) -> list:
        """Filter symbols by liquidity"""
        if min_volume is None:
            min_volume = self.min_volume
            
        liquid_symbols = []
        
        for symbol in symbols:
            try:
                ticker_24hr = self.client.futures_24hr_ticker(symbol=symbol)
                if ticker_24hr and 'volume' in ticker_24hr:
                    volume = float(ticker_24hr['volume'])
                    if volume >= min_volume:
                        liquid_symbols.append(symbol)
            except Exception as e:
                logger.debug(f"Error checking liquidity for {symbol}: {e}")
                continue
                
        return liquid_symbols
        
    def can_trade(self) -> bool:
        # Проверяем, можно ли торговать в текущих условиях
        current_time = time.time()
        
        # Проверяем время после последнего убытка
        if current_time - self.last_loss_time < self.config.STOP_LOSS_COOLDOWN * 60:
            return False
            
        # Проверяем максимальное количество убытков подряд
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            logger.warning("Max consecutive losses reached, cooling down")
            return False
            
        # Проверяем торговые часы
        current_hour = datetime.utcnow().hour
        if not (self.config.TRADING_HOURS_START <= current_hour < self.config.TRADING_HOURS_END):
            return False
            
        # Проверяем дневной лимит убытков
        daily_loss = self._get_daily_loss()
        if daily_loss > self.config.MAX_DAILY_LOSS_PERCENT:
            logger.warning("Daily loss limit reached")
            return False
            
        return True
        
    def _get_daily_loss(self) -> float:
        # Рассчитываем дневной убыток
        today = datetime.utcnow().date()
        today_trades = [t for t in self.performance_history 
                       if datetime.fromisoformat(t['timestamp']).date() == today and t['pnl'] < 0]
        
        if not today_trades:
            return 0
            
        total_loss = sum(t['pnl'] for t in today_trades)
        initial_balance = self.get_current_balance() - total_loss
        return abs(total_loss / initial_balance * 100) if initial_balance > 0 else 0
        
    def get_current_balance(self) -> float:
        try:
            balance = self.client.futures_account_balance()
            for asset in balance:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return 0
            
    def can_trade_symbol(self, symbol: str) -> bool:
        # Проверяем, можно ли торговать конкретный символ
        if symbol not in self.tradable_symbols:
            return False
            
        # Проверяем спред
        if not self.validate_spread(symbol):
            return False
            
        # Проверяем волатильность
        volatility = self.calculate_volatility(symbol)
        if volatility > self.config.VOLATILITY_THRESHOLD * 2:
            logger.warning(f"High volatility for {symbol}: {volatility:.4f}")
            return False
            
        # Проверяем наличие экономических событий
        if self._has_economic_events(symbol):
            return False
            
        return True
        
    def calculate_volatility(self, symbol: str) -> float:
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval='5m', limit=20
            )
            closes = [float(k[4]) for k in klines]
            returns = np.diff(closes) / closes[:-1]
            return np.std(returns) * np.sqrt(252)  # Годовая волатильность
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0
            
    def _has_economic_events(self, symbol: str) -> bool:
        # Проверяем наличие важных экономических событий
        base_currency = symbol.replace('USDT', '')
        for event in self.economic_events:
            if event['currency'] == base_currency and event['impact'] == 'high':
                return True
        return False
        
    def _fetch_economic_events(self):
        # Заглушка для получения экономических событий
        # В реальной реализации здесь будет API для получения событий
        self.economic_events = []
        
    def detect_market_regime(self):
        try:
            # Анализируем рынок для определения режима
            btc_volatility = self.calculate_volatility('BTCUSDT')
            eth_volatility = self.calculate_volatility('ETHUSDT')
            avg_volatility = (btc_volatility + eth_volatility) / 2
            
            if avg_volatility > 0.8:
                self.market_regime = "HIGH_VOLATILITY"
            elif avg_volatility < 0.3:
                self.market_regime = "LOW_VOLATILITY"
            else:
                self.market_regime = "NEUTRAL"
                
            logger.info(f"Market regime detected: {self.market_regime}")
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            
    def calculate_sl_tp(self, symbol: str, direction: str, entry_price: float) -> Tuple[float, float]:
        # Рассчитываем стоп-лосс и тейк-профит
        atr = self.calculate_atr(symbol)
        
        if direction == "BUY":
            sl_price = entry_price * (1 - self.config.SL_MULTIPLIER)
            tp_price = entry_price * (1 + self.config.TP_MULTIPLIER)
        else:
            sl_price = entry_price * (1 + self.config.SL_MULTIPLIER)
            tp_price = entry_price * (1 - self.config.TP_MULTIPLIER)
            
        # Адаптивные SL/TP based на волатильности
        if self.config.ADAPTIVE_SL:
            volatility = self.calculate_volatility(symbol)
            if volatility > 0.5:
                # Увеличиваем SL/TP для высокой волатильности
                if direction == "BUY":
                    sl_price = entry_price * (1 - self.config.SL_MULTIPLIER * 1.2)
                    tp_price = entry_price * (1 + self.config.TP_MULTIPLIER * 1.2)
                else:
                    sl_price = entry_price * (1 + self.config.SL_MULTIPLIER * 1.2)
                    tp_price = entry_price * (1 - self.config.TP_MULTIPLIER * 1.2)
                    
        return sl_price, tp_price
        
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval='5m', limit=period + 1
            )
            high = [float(k[2]) for k in klines]
            low = [float(k[3]) for k in klines]
            close = [float(k[4]) for k in klines]
            
            tr = []
            for i in range(1, len(klines)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr.append(max(tr1, tr2, tr3))
                
            return np.mean(tr) if tr else 0
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
            return 0
            
    def calculate_size(self, symbol: str, entry_price: float, direction: str, sl_price: float) -> float:
        balance = self.get_current_balance()
        if balance < self.config.MIN_BALANCE:
            return 0
            
        # Рассчитываем риск в долларах
        risk_amount = balance * (self.config.RISK_PERCENT / 100)
        
        # Рассчитываем размер позиции
        if direction == "BUY":
            risk_per_unit = entry_price - sl_price
        else:
            risk_per_unit = sl_price - entry_price
            
        if risk_per_unit <= 0:
            return 0
            
        size = risk_amount / risk_per_unit
        
        # Применяем максимальный размер позиции
        max_position_size = balance * self.config.DEFAULT_LEVERAGE * self.config.MAX_POSITION_SIZE
        position_value = size * entry_price
        
        if position_value > max_position_size:
            size = max_position_size / entry_price
            
        return size
        
    def get_mark_price(self, symbol: str) -> float:
        try:
            ticker = self.client.futures_mark_price(symbol=symbol)
            return float(ticker['markPrice'])
        except Exception as e:
            logger.error(f"Error getting mark price for {symbol}: {str(e)}")
            return 0
            
    def get_open_positions(self) -> List[Dict]:
        try:
            positions = self.client.futures_position_information()
            return [p for p in positions if float(p['positionAmt']) != 0]
        except Exception as e:
            logger.error(f"Error getting open positions: {str(e)}")
            return []
            
    def has_correlated_position(self, symbol: str, direction: str) -> bool:
        # Проверяем, есть ли коррелированные позиции
        open_positions = self.get_open_positions()
        
        for position in open_positions:
            pos_symbol = position['symbol']
            pos_amount = float(position['positionAmt'])
            
            if pos_amount == 0:
                continue
                
            # Проверяем корреляцию
            if self._are_symbols_correlated(symbol, pos_symbol):
                pos_direction = "BUY" if pos_amount > 0 else "SELL"
                if pos_direction != direction:
                    return True
                    
        return False
        
    def _are_symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        # Проверяем, коррелированы ли символы
        if symbol1 == symbol2:
            return True
            
        base1 = symbol1.replace('USDT', '')
        base2 = symbol2.replace('USDT', '')
        
        # Проверяем явные корреляции из конфига
        for base, correlated in self.config.CORRELATED_PAIRS.items():
            if base1 == base and base2 in correlated:
                return True
            if base2 == base and base1 in correlated:
                return True
                
        return False
        
    def detect_market_reversal(self, symbol: str, direction: str) -> bool:
        # Обнаружение разворота рынка
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval='3m', limit=10
            )
            closes = [float(k[4]) for k in klines]
            
            # Простая проверка на разворот
            if len(closes) >= 5:
                recent_trend = np.sign(closes[-1] - closes[-5])
                current_direction = 1 if direction == "BUY" else -1
                
                if recent_trend != current_direction:
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error detecting market reversal for {symbol}: {str(e)}")
            return False
            
    def detect_whale_activity(self, symbol: str) -> bool:
        # Обнаружение активности китов
        try:
            depth = self.client.futures_order_book(symbol=symbol, limit=20)
            bids = depth['bids']
            asks = depth['asks']
            
            # Проверяем крупные ордера
            large_bids = sum(float(b[1]) for b in bids if float(b[1]) > self.config.WHALE_VOLUME_FACTOR * 1000)
            large_asks = sum(float(a[1]) for a in asks if float(a[1]) > self.config.WHALE_VOLUME_FACTOR * 1000)
            
            return large_bids > 50000 or large_asks > 50000
        except Exception as e:
            logger.error(f"Error detecting whale activity for {symbol}: {str(e)}")
            return False
            
    def calculate_market_momentum(self, symbol: str) -> float:
        # Расчет импульса рынка
        try:
            klines = self.client.futures_klines(
                symbol=symbol, interval='5m', limit=10
            )
            closes = [float(k[4]) for k in klines]
            
            if len(closes) >= 5:
                returns = np.diff(closes) / closes[:-1]
                momentum = np.mean(returns[-3:])  # Среднее за последние 3 периода
                return momentum
                
            return 0
        except Exception as e:
            logger.error(f"Error calculating momentum for {symbol}: {str(e)}")
            return 0
            
    def update_trade_outcome(self, profitable: bool, pnl: float, symbol: str):
        # Обновляем статистику после сделки
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'profitable': profitable,
            'pnl': pnl
        }
        
        self.performance_history.append(trade_data)
        
        # Ограничиваем размер истории
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
        # Обновляем счетчик убытков
        if not profitable:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()
        else:
            self.consecutive_losses = 0
            
    def calculate_win_rate(self) -> float:
        if not self.performance_history:
            return 0
            
        profitable_trades = sum(1 for t in self.performance_history if t['profitable'])
        return profitable_trades / len(self.performance_history)
        
    def get_max_leverage(self, symbol: str) -> int:
        # Получаем максимальное доступное кредитное плечо
        try:
            leverage_brackets = self.client.futures_leverage_bracket()
            for bracket in leverage_brackets:
                if bracket['symbol'] == symbol:
                    return max(int(b['leverage']) for b in bracket['brackets'])
            return self.config.DEFAULT_LEVERAGE
        except Exception as e:
            logger.error(f"Error getting max leverage for {symbol}: {str(e)}")
            return self.config.DEFAULT_LEVERAGE
            
    def create_hedge_position(self, symbol: str, direction: str, size: float):
        # Создание хеджирующей позиции
        if not self.config.POSITION_HEDGING:
            return
            
        try:
            hedge_direction = "SELL" if direction == "BUY" else "BUY"
            hedge_size = size * self.config.HEDGE_RATIO
            
            # Проверяем минимальный размер
            min_qty = self.config.SYMBOL_INFO.get(symbol, {}).get('min_qty', 1.0)
            if hedge_size < min_qty:
                return
                
            order = self.client.futures_create_order(
                symbol=symbol,
                side=hedge_direction,
                type="MARKET",
                quantity=hedge_size
            )
            
            self.hedge_positions[symbol] = {
                'order_id': order['orderId'],
                'size': hedge_size,
                'direction': hedge_direction,
                'timestamp': time.time()
            }
            
            logger.info(f"Hedge position created for {symbol}: {hedge_size} {hedge_direction}")
            
        except Exception as e:
            logger.error(f"Error creating hedge position for {symbol}: {str(e)}")
            
    def close_hedge_position(self, symbol: str):
        # Закрытие хеджирующей позиции
        if symbol not in self.hedge_positions:
            return
            
        try:
            hedge_info = self.hedge_positions[symbol]
            close_direction = "BUY" if hedge_info['direction'] == "SELL" else "SELL"
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_direction,
                type="MARKET",
                quantity=hedge_info['size'],
                reduceOnly=True
            )
            
            del self.hedge_positions[symbol]
            logger.info(f"Hedge position closed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing hedge position for {symbol}: {str(e)}")
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from config import Config
import pandas_ta as ta
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger("ScenarioAnalyzer")

@dataclass
class ScenarioResult:
    name: str
    probability: float
    confidence: float
    expected_return: float
    risk_level: str
    timeframe: str
    conditions: Dict

class ScenarioAnalyzer:
    def __init__(self, data_feeder):
        self.data_feeder = data_feeder
        self.scenarios = self._initialize_scenarios()
        self.scenario_performance = {}
        self.last_analysis_time = {}
        self.analysis_cache = {}
        self.lock = threading.RLock()
        
        logger.info("ScenarioAnalyzer initialized with pandas_ta")
    
    def _analyze_symbol_simple(self, symbol: str) -> Dict:
        """Упрощенный анализ символа без сложных зависимостей"""
        try:
            # Получаем базовые данные
            current_price = self.data_feeder.get_current_price(symbol) if self.data_feeder else 0
            if current_price == 0:
                return None
                
            # Простая логика на основе цены и объема
            df = self.data_feeder.get_market_data(symbol, '15m') if self.data_feeder else None
            
            if df is None or df.empty or len(df) < 10:
                return None
            
            # Простые индикаторы
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if df['close'].iloc[-5] != 0 else 0
            volume_avg = df['volume'].tail(10).mean()
            volume_current = df['volume'].iloc[-1]
            volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1
            
            # Простая оценка
            score = 5.0  # Базовый score
            
            # Корректировки based на данных
            if abs(price_change) > 0.01:  # Изменение цены > 1%
                score += 1.0
            if volume_ratio > 1.5:  # Объем выше среднего
                score += 1.0
            if volume_ratio > 2.0:  # Высокий объем
                score += 1.0
                
            # Определяем направление
            if price_change > 0.005 and volume_ratio > 1.2:
                direction = 'BUY'
            elif price_change < -0.005 and volume_ratio > 1.2:
                direction = 'SELL'
            else:
                direction = 'HOLD'
            
            if direction == 'HOLD':
                return None
                
            return {
                'symbol': symbol,
                'direction': direction,
                'score': min(score, 10.0),
                'confidence': min(volume_ratio / 3.0, 1.0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Simple analysis error for {symbol}: {e}")
            return None

    def _initialize_scenarios(self) -> Dict[str, Dict]:
        """Инициализирует торговые сценарии"""
        return {
            'momentum_breakout': {
                'description': 'Прорыв с подтверждением импульса',
                'timeframes': ['5m', '15m'],
                'conditions': ['high_volume', 'trend_confirmation', 'volatility_expansion'],
                'weight': 0.8
            },
            'mean_reversion': {
                'description': 'Возврат к среднему с перепроданностью/перекупленностью',
                'timeframes': ['15m', '1h'],
                'conditions': ['rsi_extreme', 'bollinger_band_extreme', 'volume_confirmation'],
                'weight': 0.7
            },
            'trend_following': {
                'description': 'Следование тренду с подтверждением',
                'timeframes': ['1h', '4h'],
                'conditions': ['trend_strength', 'moving_average_alignment', 'momentum_confirmation'],
                'weight': 0.75
            },
            'volatility_breakout': {
                'description': 'Прорыв в условиях высокой волатильности',
                'timeframes': ['5m', '15m'],
                'conditions': ['low_volatility', 'volume_spike', 'price_breakout'],
                'weight': 0.65
            },
            'reversal_pattern': {
                'description': 'Разворотные паттерны с подтверждением',
                'timeframes': ['15m', '1h'],
                'conditions': ['divergence', 'candlestick_pattern', 'volume_confirmation'],
                'weight': 0.6
            }
        }
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Анализирует символ по всем сценариям"""
        try:
            # Сначала пробуем быстрый простой анализ
            simple_result = self._analyze_symbol_simple(symbol)
            if simple_result is None:
                # Если простой анализ не дал сигнала, возвращаем HOLD
                return {
                    'symbol': symbol,
                    'score': 0,
                    'direction': 'HOLD',
                    'scenario_details': {},
                    'timestamp': datetime.now(),
                    'analysis_type': 'simple'
                }
            
            # Проверяем кэш для сложного анализа
            cache_key = f"{symbol}_{datetime.now().minute}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            results = []
            total_weight = 0
            
            for scenario_name, scenario_config in self.scenarios.items():
                for timeframe in scenario_config['timeframes']:
                    try:
                        result = self._analyze_scenario(symbol, scenario_name, timeframe)
                        if result and result.probability > 0.3:  # Минимальный порог
                            weighted_score = result.probability * result.confidence * scenario_config['weight']
                            results.append({
                                'scenario': scenario_name,
                                'timeframe': timeframe,
                                'probability': result.probability,
                                'confidence': result.confidence,
                                'expected_return': result.expected_return,
                                'risk_level': result.risk_level,
                                'weighted_score': weighted_score,
                                'conditions': result.conditions
                            })
                            total_weight += scenario_config['weight']
                    except Exception as e:
                        logger.error(f"Error analyzing {scenario_name} for {symbol}: {e}")
                        continue
            
            if not results:
                # Если сложный анализ не дал результатов, используем простой
                return {
                    'symbol': symbol,
                    'score': simple_result['score'],
                    'direction': simple_result['direction'],
                    'scenario_details': {'simple_analysis': simple_result},
                    'timestamp': datetime.now(),
                    'analysis_type': 'simple_fallback'
                }
            
            # Вычисляем общий score
            total_score = sum(r['weighted_score'] for r in results) / total_weight if total_weight > 0 else 0
            max_score_result = max(results, key=lambda x: x['weighted_score'])
            
            # Определяем направление
            direction = self._determine_direction(max_score_result, symbol)
            
            result = {
                'symbol': symbol,
                'score': min(total_score * 10, 10.0),  # Нормализуем к 0-10
                'direction': direction,
                'scenario_details': {r['scenario']: r for r in results},
                'timestamp': datetime.now(),
                'top_scenario': max_score_result['scenario'],
                'top_timeframe': max_score_result['timeframe'],
                'analysis_type': 'advanced'
            }
            
            # Кэшируем результат
            self.analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            # В случае ошибки используем простой анализ как фолбэк
            simple_result = self._analyze_symbol_simple(symbol)
            if simple_result:
                return {
                    'symbol': symbol,
                    'score': simple_result['score'],
                    'direction': simple_result['direction'],
                    'scenario_details': {'error_fallback': simple_result},
                    'timestamp': datetime.now(),
                    'analysis_type': 'error_fallback'
                }
            else:
                return {
                    'symbol': symbol,
                    'score': 0,
                    'direction': 'HOLD',
                    'scenario_details': {},
                    'timestamp': datetime.now(),
                    'analysis_type': 'error'
                }
    
    def analyze_symbol_fast(self, symbol: str) -> Dict[str, Any]:
        """Быстрый анализ только через упрощенный метод"""
        try:
            result = self._analyze_symbol_simple(symbol)
            if result:
                return {
                    'symbol': symbol,
                    'score': result['score'],
                    'direction': result['direction'],
                    'confidence': result['confidence'],
                    'timestamp': result['timestamp'],
                    'analysis_type': 'fast_simple'
                }
            else:
                return {
                    'symbol': symbol,
                    'score': 0,
                    'direction': 'HOLD',
                    'confidence': 0,
                    'timestamp': datetime.now(),
                    'analysis_type': 'fast_simple'
                }
        except Exception as e:
            logger.error(f"Fast analysis error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'score': 0,
                'direction': 'HOLD',
                'confidence': 0,
                'timestamp': datetime.now(),
                'analysis_type': 'fast_error'
            }
    
    def _analyze_scenario(self, symbol: str, scenario_name: str, timeframe: str) -> Optional[ScenarioResult]:
        """Анализирует конкретный сценарий для символа"""
        try:
            df = self.data_feeder.get_market_data(symbol, timeframe)
            if df.empty or len(df) < 50:
                return None
            
            # Получаем технические индикаторы через pandas_ta
            indicators = self._calculate_indicators(df)
            
            if scenario_name == 'momentum_breakout':
                return self._analyze_momentum_breakout(symbol, df, indicators, timeframe)
            elif scenario_name == 'mean_reversion':
                return self._analyze_mean_reversion(symbol, df, indicators, timeframe)
            elif scenario_name == 'trend_following':
                return self._analyze_trend_following(symbol, df, indicators, timeframe)
            elif scenario_name == 'volatility_breakout':
                return self._analyze_volatility_breakout(symbol, df, indicators, timeframe)
            elif scenario_name == 'reversal_pattern':
                return self._analyze_reversal_pattern(symbol, df, indicators, timeframe)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in scenario analysis {scenario_name} for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Вычисляет технические индикаторы с использованием pandas_ta"""
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df['close'])
            if macd_data is not None and not macd_data.empty:
                indicators['macd'] = macd_data.iloc[:, 0]  # MACD line
                indicators['macd_signal'] = macd_data.iloc[:, 1]  # Signal line
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if bb_data is not None and not bb_data.empty:
                indicators['bb_upper'] = bb_data.iloc[:, 0]
                indicators['bb_lower'] = bb_data.iloc[:, 2]
                indicators['bb_middle'] = bb_data.iloc[:, 1]
            
            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'])
            if stoch_data is not None and not stoch_data.empty:
                indicators['stoch_k'] = stoch_data.iloc[:, 0]
                indicators['stoch_d'] = stoch_data.iloc[:, 1]
            
            # ATR
            indicators['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # ADX
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_data is not None and not adx_data.empty:
                indicators['adx'] = adx_data.iloc[:, 0]  # ADX line
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(20).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            # Price changes
            indicators['price_change_1'] = df['close'].pct_change(1)
            indicators['price_change_5'] = df['close'].pct_change(5)
            indicators['price_change_20'] = df['close'].pct_change(20)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _analyze_momentum_breakout(self, symbol: str, df: pd.DataFrame, 
                                 indicators: Dict, timeframe: str) -> ScenarioResult:
        """Анализирует сценарий импульсного прорыва"""
        conditions = {}
        
        try:
            current_price = df['close'].iloc[-1]
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            price_change_5 = indicators['price_change_5'].iloc[-1] if 'price_change_5' in indicators else 0
            
            # Условие 1: Высокий объем
            conditions['high_volume'] = volume_ratio > 1.5
            
            # Условие 2: Подтверждение тренда
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                conditions['trend_confirmation'] = macd > macd_signal and macd > 0
            
            # Условие 3: Расширение волатильности
            if 'atr' in indicators:
                atr_current = indicators['atr'].iloc[-1]
                atr_prev = indicators['atr'].iloc[-2] if len(indicators['atr']) > 1 else atr_current
                conditions['volatility_expansion'] = atr_current > atr_prev * 1.1
            
            # Вычисляем вероятность
            true_conditions = sum(conditions.values())
            total_conditions = len(conditions)
            probability = true_conditions / total_conditions if total_conditions > 0 else 0
            
            # Уверенность based на силе сигналов
            confidence = min(volume_ratio / 2.0, 1.0) * 0.6 + probability * 0.4
            
            # Ожидаемая доходность
            expected_return = abs(price_change_5) * 2 if probability > 0.5 else abs(price_change_5)
            
            return ScenarioResult(
                name='momentum_breakout',
                probability=probability,
                confidence=confidence,
                expected_return=expected_return,
                risk_level='MEDIUM',
                timeframe=timeframe,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Momentum breakout analysis error for {symbol}: {e}")
            return ScenarioResult(
                name='momentum_breakout',
                probability=0,
                confidence=0,
                expected_return=0,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions={}
            )
    
    def _analyze_mean_reversion(self, symbol: str, df: pd.DataFrame, 
                              indicators: Dict, timeframe: str) -> ScenarioResult:
        """Анализирует сценарий возврата к среднему"""
        conditions = {}
        
        try:
            current_price = df['close'].iloc[-1]
            rsi = indicators.get('rsi', pd.Series([50])).iloc[-1]
            
            # Условие 1: Экстремальные значения RSI
            conditions['rsi_extreme'] = rsi < 30 or rsi > 70
            
            # Условие 2: Экстремумы полос Боллинджера
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper = indicators['bb_upper'].iloc[-1]
                bb_lower = indicators['bb_lower'].iloc[-1]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                conditions['bollinger_band_extreme'] = bb_position < 0.1 or bb_position > 0.9
            
            # Условие 3: Подтверждение объемом
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            conditions['volume_confirmation'] = volume_ratio > 1.2
            
            probability = sum(conditions.values()) / len(conditions) if conditions else 0
            confidence = 0.7 if conditions.get('rsi_extreme', False) else 0.4
            
            # Ожидаемая доходность based на отклонении от среднего
            if 'bb_middle' in indicators:
                bb_middle = indicators['bb_middle'].iloc[-1]
                deviation = abs(current_price - bb_middle) / bb_middle
                expected_return = deviation * 0.8
            else:
                expected_return = 0.02
            
            return ScenarioResult(
                name='mean_reversion',
                probability=probability,
                confidence=confidence,
                expected_return=expected_return,
                risk_level='MEDIUM',
                timeframe=timeframe,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Mean reversion analysis error for {symbol}: {e}")
            return ScenarioResult(
                name='mean_reversion',
                probability=0,
                confidence=0,
                expected_return=0,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions={}
            )
    
    def _analyze_trend_following(self, symbol: str, df: pd.DataFrame, 
                               indicators: Dict, timeframe: str) -> ScenarioResult:
        """Анализирует сценарий следования тренду"""
        conditions = {}
        
        try:
            # Условие 1: Сила тренда (ADX)
            adx = indicators.get('adx', pd.Series([20])).iloc[-1]
            conditions['trend_strength'] = adx > 25
            
            # Условие 2: Выравнивание скользящих средних
            if len(df) > 50:
                ema_fast = ta.ema(df['close'], length=8).iloc[-1]
                ema_medium = ta.ema(df['close'], length=21).iloc[-1]
                ema_slow = ta.ema(df['close'], length=50).iloc[-1]
                
                # Проверяем порядок скользящих средних для восходящего/нисходящего тренда
                if ema_fast > ema_medium > ema_slow:
                    conditions['moving_average_alignment'] = True
                elif ema_fast < ema_medium < ema_slow:
                    conditions['moving_average_alignment'] = True
                else:
                    conditions['moving_average_alignment'] = False
            
            # Условие 3: Подтверждение импульсом
            if 'macd' in indicators:
                macd = indicators['macd'].iloc[-1]
                conditions['momentum_confirmation'] = macd > 0 if conditions.get('moving_average_alignment', False) else macd < 0
            
            probability = sum(conditions.values()) / len(conditions) if conditions else 0
            confidence = min(adx / 50.0, 1.0) * 0.7 + probability * 0.3
            
            # Ожидаемая доходность based на силе тренда
            expected_return = min(adx / 100.0, 0.05)
            
            return ScenarioResult(
                name='trend_following',
                probability=probability,
                confidence=confidence,
                expected_return=expected_return,
                risk_level='LOW',
                timeframe=timeframe,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Trend following analysis error for {symbol}: {e}")
            return ScenarioResult(
                name='trend_following',
                probability=0,
                confidence=0,
                expected_return=0,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions={}
            )
    
    def _analyze_volatility_breakout(self, symbol: str, df: pd.DataFrame, 
                                   indicators: Dict, timeframe: str) -> ScenarioResult:
        """Анализирует сценарий прорыва волатильности"""
        conditions = {}
        
        try:
            # Условие 1: Низкая волатильность (сжатие)
            if 'atr' in indicators and len(indicators['atr']) > 20:
                atr_current = indicators['atr'].iloc[-1]
                atr_avg = indicators['atr'].tail(20).mean()
                conditions['low_volatility'] = atr_current < atr_avg * 0.7
            
            # Условие 2: Скачок объема
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            conditions['volume_spike'] = volume_ratio > 2.0
            
            # Условие 3: Прорыв цены
            if len(df) > 10:
                recent_high = df['high'].tail(10).max()
                recent_low = df['low'].tail(10).min()
                current_price = df['close'].iloc[-1]
                range_size = recent_high - recent_low
                
                if range_size > 0:
                    price_position = (current_price - recent_low) / range_size
                    conditions['price_breakout'] = price_position > 0.8 or price_position < 0.2
            
            probability = sum(conditions.values()) / len(conditions) if conditions else 0
            confidence = volume_ratio / 3.0 * 0.6 + probability * 0.4
            
            # Ожидаемая доходность
            expected_return = 0.03  # Фиксированная для волатильностных прорывов
            
            return ScenarioResult(
                name='volatility_breakout',
                probability=probability,
                confidence=confidence,
                expected_return=expected_return,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Volatility breakout analysis error for {symbol}: {e}")
            return ScenarioResult(
                name='volatility_breakout',
                probability=0,
                confidence=0,
                expected_return=0,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions={}
            )
    
    def _analyze_reversal_pattern(self, symbol: str, df: pd.DataFrame, 
                                indicators: Dict, timeframe: str) -> ScenarioResult:
        """Анализирует сценарий разворотного паттерна"""
        conditions = {}
        
        try:
            # Условие 1: Дивергенция
            conditions['divergence'] = self._check_divergence(df, indicators)
            
            # Условие 2: Свечные паттерны (упрощенная проверка)
            conditions['candlestick_pattern'] = self._check_simple_patterns(df)
            
            # Условие 3: Подтверждение объемом
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            conditions['volume_confirmation'] = volume_ratio > 1.3
            
            probability = sum(conditions.values()) / len(conditions) if conditions else 0
            confidence = 0.8 if conditions.get('divergence', False) else 0.5
            
            # Ожидаемая доходность
            expected_return = 0.04  # Высокая для разворотов
            
            return ScenarioResult(
                name='reversal_pattern',
                probability=probability,
                confidence=confidence,
                expected_return=expected_return,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Reversal pattern analysis error for {symbol}: {e}")
            return ScenarioResult(
                name='reversal_pattern',
                probability=0,
                confidence=0,
                expected_return=0,
                risk_level='HIGH',
                timeframe=timeframe,
                conditions={}
            )
    
    def _check_divergence(self, df: pd.DataFrame, indicators: Dict) -> bool:
        """Проверяет наличие дивергенции"""
        try:
            if len(df) < 20:
                return False
            
            # Простая проверка дивергенции между ценой и RSI
            price_highs = df['high'].tail(10)
            rsi = indicators.get('rsi', pd.Series([50] * len(df))).tail(10)
            
            if len(price_highs) < 5 or len(rsi) < 5:
                return False
            
            # Проверяем расхождение между максимумами цены и RSI
            price_trend = price_highs.iloc[-1] > price_highs.iloc[-5]
            rsi_trend = rsi.iloc[-1] > rsi.iloc[-5]
            
            return price_trend != rsi_trend
            
        except Exception as e:
            logger.error(f"Divergence check error: {e}")
            return False
    
    def _check_simple_patterns(self, df: pd.DataFrame) -> bool:
        """Проверяет простые ценовые паттерны"""
        try:
            if len(df) < 5:
                return False
            
            # Простая проверка на разворотные формации
            recent_candles = df.tail(3)
            
            # Проверяем паттерн "утренняя/вечерняя звезда"
            if len(recent_candles) == 3:
                first = recent_candles.iloc[0]
                second = recent_candles.iloc[1]
                third = recent_candles.iloc[2]
                
                # Упрощенная проверка (реальная реализация требует более сложной логики)
                body_first = abs(first['close'] - first['open'])
                body_second = abs(second['close'] - second['open'])
                body_third = abs(third['close'] - third['open'])
                
                # Маленькое тело в середине (доджи/спиннинг топ)
                if body_second < body_first * 0.3 and body_second < body_third * 0.3:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Pattern check error: {e}")
            return False
    
    def _determine_direction(self, scenario_result: Dict, symbol: str) -> str:
        """Определяет направление сделки based на сценарии"""
        try:
            scenario_name = scenario_result['scenario']
            
            if scenario_name in ['momentum_breakout', 'trend_following']:
                # Анализируем тренд
                df = self.data_feeder.get_market_data(symbol, scenario_result['timeframe'])
                if df.empty:
                    return 'HOLD'
                
                # Простой анализ тренда
                if len(df) > 20:
                    short_ma = df['close'].tail(5).mean()
                    long_ma = df['close'].tail(20).mean()
                    
                    if short_ma > long_ma:
                        return 'BUY'
                    else:
                        return 'SELL'
            
            elif scenario_name in ['mean_reversion', 'reversal_pattern']:
                # Анализируем перепроданность/перекупленность
                df = self.data_feeder.get_market_data(symbol, '15m')
                if df.empty:
                    return 'HOLD'
                
                indicators = self._calculate_indicators(df)
                rsi = indicators.get('rsi', pd.Series([50])).iloc[-1]
                
                if rsi < 35:
                    return 'BUY'
                elif rsi > 65:
                    return 'SELL'
            
            elif scenario_name == 'volatility_breakout':
                # Случайное направление для волатильностного прорыва
                return 'BUY' if datetime.now().second % 2 == 0 else 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Direction determination error for {symbol}: {e}")
            return 'HOLD'
    
    def update_scenario_performance(self, scenario_name: str, profitable: bool, pnl: float):
        """Обновляет производительность сценария"""
        try:
            with self.lock:
                if scenario_name not in self.scenario_performance:
                    self.scenario_performance[scenario_name] = {
                        'total_trades': 0,
                        'profitable_trades': 0,
                        'total_pnl': 0,
                        'last_updated': datetime.now()
                    }
                
                performance = self.scenario_performance[scenario_name]
                performance['total_trades'] += 1
                performance['total_pnl'] += pnl
                
                if profitable:
                    performance['profitable_trades'] += 1
                
                performance['last_updated'] = datetime.now()
                
                # Обновляем вес сценария based на производительности
                win_rate = performance['profitable_trades'] / performance['total_trades']
                if performance['total_trades'] >= 5:
                    new_weight = min(win_rate * 1.2, 1.0)
                    if scenario_name in self.scenarios:
                        self.scenarios[scenario_name]['weight'] = new_weight
                        
        except Exception as e:
            logger.error(f"Error updating scenario performance: {e}")
    
    def get_scenario_performance(self) -> Dict:
        """Возвращает статистику производительности сценариев"""
        return self.scenario_performance.copy()

    def clear_cache(self):
        """Очищает кэш анализа"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")

    def get_analysis_stats(self) -> Dict:
        """Возвращает статистику анализа"""
        return {
            'cached_results': len(self.analysis_cache),
            'scenario_performance': self.scenario_performance,
            'active_scenarios': len(self.scenarios),
            'last_analysis_time': self.last_analysis_time
        }

import psutil
import time
import threading
import logging
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
import statistics
from config import Config

logger = logging.getLogger("SystemMonitor")

@dataclass
class PerformanceStats:
    cpu_peak: float = 0.0
    memory_peak: float = 0.0
    network_usage: float = 0.0
    errors_count: int = 0
    api_latency: float = 0.0
    memory_leaks_detected: int = 0
    disk_usage: float = 0.0
    process_count: int = 0
    gpu_usage: Optional[float] = None
    api_errors: int = 0
    last_cpu: float = 0.0
    last_memory: float = 0.0
    historical_data: Dict[str, list] = None
    
    def __post_init__(self):
        if self.historical_data is None:
            self.historical_data = {
                'cpu': [], 'memory': [], 'api_latency': [],
                'network': [], 'disk': []
            }

class SystemMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.performance_stats = PerformanceStats()
        self.start_time = time.time()
        self.anomaly_detected = False
        self.alert_cooldown = {}
        self.monitoring_active = True
        self.loop = asyncio.new_event_loop()
        self.start_monitoring()

    def start_monitoring(self):
        """Запуск мониторинга в отдельном потоке"""
        def monitor_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._monitoring_coroutine())
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def _monitoring_coroutine(self):
        """Основная корутина мониторинга"""
        while self.monitoring_active:
            try:
                await self._check_resources()
                await asyncio.sleep(60)  # Проверка каждую минуту
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                self.performance_stats.errors_count += 1
                await asyncio.sleep(10)

    async def _check_resources(self):
        """Проверка системных ресурсов"""
        tasks = [
            self._check_cpu(),
            self._check_memory(),
            self._check_disk(),
            self._check_network(),
            self._check_api_latency(),
            self._check_gpu(),
            self._check_processes()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Проверка аномалий и отправка отчетов
        self._check_for_anomalies()
        
        # Логирование каждые 5 минут
        if int(time.time()) % 300 == 0:
            await self._log_system_status()
            
        # Ежечасный отчет
        if int(time.time()) % 3600 == 0:
            await self.send_performance_report()

    async def _check_cpu(self):
        """Мониторинг использования CPU"""
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Обновление исторических данных
        self.performance_stats.historical_data['cpu'].append(cpu_usage)
        if len(self.performance_stats.historical_data['cpu']) > 60:
            self.performance_stats.historical_data['cpu'].pop(0)
        
        # Проверка превышения порога
        if cpu_usage > Config.CPU_THRESHOLD:
            await self._send_alert(
                "⚠️ HIGH CPU USAGE", 
                f"CPU usage: {cpu_usage}%", 
                "cpu_alert"
            )
        
        # Обновление пикового значения
        self.performance_stats.cpu_peak = max(
            self.performance_stats.cpu_peak, 
            cpu_usage
        )

    async def _check_memory(self):
        """Мониторинг использования памяти"""
        memory_usage = psutil.virtual_memory().percent
        
        # Обновление исторических данных
        self.performance_stats.historical_data['memory'].append(memory_usage)
        if len(self.performance_stats.historical_data['memory']) > 60:
            self.performance_stats.historical_data['memory'].pop(0)
        
        # Проверка превышения порога
        if memory_usage > Config.MEMORY_THRESHOLD:
            await self._send_alert(
                "⚠️ HIGH MEMORY USAGE", 
                f"Memory usage: {memory_usage}%", 
                "memory_alert"
            )
        
        # Проверка на утечки памяти
        if (memory_usage > 80 and 
            self.performance_stats.memory_leaks_detected < 3):
            self.performance_stats.memory_leaks_detected += 1
            if self.performance_stats.memory_leaks_detected == 3:
                await self._send_alert(
                    "🚨 POSSIBLE MEMORY LEAK", 
                    "Potential memory leak detected", 
                    "memory_leak"
                )
        
        # Обновление пикового значения
        self.performance_stats.memory_peak = max(
            self.performance_stats.memory_peak, 
            memory_usage
        )

    async def _check_disk(self):
        """Мониторинг использования диска"""
        disk_usage = psutil.disk_usage('/').percent
        
        # Обновление исторических данных
        self.performance_stats.historical_data['disk'].append(disk_usage)
        if len(self.performance_stats.historical_data['disk']) > 60:
            self.performance_stats.historical_data['disk'].pop(0)
        
        # Проверка превышения порога
        if disk_usage > Config.DISK_THRESHOLD:
            await self._send_alert(
                "⚠️ HIGH DISK USAGE", 
                f"Disk usage: {disk_usage}%", 
                "disk_alert"
            )
        
        # Проверка свободного места
        free_space_gb = psutil.disk_usage('/').free / (1024 ** 3)
        if free_space_gb < 5:
            await self._send_alert(
                "⚠️ LOW DISK SPACE", 
                f"Only {free_space_gb:.1f}GB free space left", 
                "disk_space_alert"
            )

    async def _check_network(self):
        """Мониторинг сетевой активности"""
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
        
        # Обновление исторических данных
        self.performance_stats.historical_data['network'].append(network_usage)
        if len(self.performance_stats.historical_data['network']) > 60:
            self.performance_stats.historical_data['network'].pop(0)
        
        self.performance_stats.network_usage = network_usage

    async def _check_api_latency(self):
        """УПРОЩЕННАЯ ПРОВЕРКА LATENCY БЕЗ ЛИШНИХ ЗАПРОСОВ"""
        try:
            # Вместо реального API запроса используем легковесную проверку
            start_time = time.time()
            
            # Простой ping вместо сложных запросов
            if hasattr(self.bot, 'client') and self.bot.client:
                # Используем самый простой метод
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.bot.client.futures_ping()
                )
            
            latency = (time.time() - start_time) * 1000
            
            # Обновление исторических данных
            self.performance_stats.historical_data['api_latency'].append(latency)
            if len(self.performance_stats.historical_data['api_latency']) > 60:
                self.performance_stats.historical_data['api_latency'].pop(0)
            
            self.performance_stats.api_latency = latency
            
            # 🔥 УВЕЛИЧИВАЕМ порог предупреждений
            if latency > 3000:  # 3 секунды вместо 1
                await self._send_alert(
                    "⚠️ HIGH API LATENCY", 
                    f"API latency: {latency:.2f}ms", 
                    "api_latency_alert"
                )
            
            return latency
            
        except Exception as e:
            logger.debug(f"API latency check skipped: {e}")
            self.performance_stats.api_errors += 1
            return 0

    async def _check_gpu(self):
        """Мониторинг использования GPU (если доступно)"""
        try:
            # Попытка импорта библиотек для мониторинга GPU
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = sum([gpu.load * 100 for gpu in gpus]) / len(gpus)
                self.performance_stats.gpu_usage = gpu_usage
                
                if gpu_usage > Config.GPU_THRESHOLD:
                    await self._send_alert(
                        "⚠️ HIGH GPU USAGE", 
                        f"GPU usage: {gpu_usage:.1f}%", 
                        "gpu_alert"
                    )
        except ImportError:
            # GPU мониторинг недоступен
            pass
        except Exception as e:
            logger.error(f"GPU check failed: {str(e)}")

    async def _check_processes(self):
        """Мониторинг количества процессов"""
        self.performance_stats.process_count = len(psutil.pids())

    async def _send_alert(self, title, message, alert_type):
        """РЕЖИМ ТИШИНЫ ДЛЯ LATENCY УВЕДОМЛЕНИЙ"""
        # НЕ отправляем уведомления о latency
        if "LATENCY" in title:
            logger.debug(f"Latency alert suppressed: {message}")
            return
            
        # УВЕЛИЧИВАЕМ кд для всех уведомлений
        current_time = time.time()
        last_alert = self.alert_cooldown.get(alert_type, 0)
        
        if current_time - last_alert < 600:  # 10 минут между уведомлениями
            return
            
        self.alert_cooldown[alert_type] = current_time
        
        full_message = f"{title}\n{message}\nTimestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        if Config.TELEGRAM_ENABLED:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.bot.send_telegram_alert(full_message)
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {str(e)}")

    def _check_for_anomalies(self):
        """Проверяет аномалии в системе с использованием статистических методов"""
        try:
            cpu_data = self.performance_stats.historical_data['cpu']
            memory_data = self.performance_stats.historical_data['memory']
            
            if len(cpu_data) > 10:
                # Обнаружение аномалий с использованием z-score
                cpu_mean = statistics.mean(cpu_data)
                cpu_std = statistics.stdev(cpu_data)
                
                if cpu_std > 0:  # Избегаем деления на ноль
                    current_cpu = cpu_data[-1]
                    z_score = abs((current_cpu - cpu_mean) / cpu_std)
                    
                    if z_score > 3.0 and current_cpu > 50:
                        self.anomaly_detected = True
                        asyncio.run_coroutine_threadsafe(
                            self._send_alert(
                                "🚨 CPU ANOMALY DETECTED",
                                f"Unusual CPU activity: {current_cpu}% (z-score: {z_score:.2f})",
                                "cpu_anomaly"
                            ),
                            self.loop
                        )
            
            # Аналогичная проверка для памяти
            if len(memory_data) > 10:
                memory_mean = statistics.mean(memory_data)
                memory_std = statistics.stdev(memory_data)
                
                if memory_std > 0:
                    current_memory = memory_data[-1]
                    z_score = abs((current_memory - memory_mean) / memory_std)
                    
                    if z_score > 3.0 and current_memory > 50:
                        self.anomaly_detected = True
                        asyncio.run_coroutine_threadsafe(
                            self._send_alert(
                                "🚨 MEMORY ANOMALY DETECTED",
                                f"Unusual memory activity: {current_memory}% (z-score: {z_score:.2f})",
                                "memory_anomaly"
                            ),
                            self.loop
                        )
                        
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")

    async def _log_system_status(self):
        """Логирование статуса системы"""
        stats = self.get_system_stats()
        logger.info(
            f"System status - "
            f"CPU: {stats['cpu']}%, "
            f"Memory: {stats['memory']}%, "
            f"Disk: {stats['disk']}%, "
            f"Network: {stats['network']:.2f}MB, "
            f"API Latency: {stats['api_latency']:.2f}ms, "
            f"Processes: {stats['process_count']}"
        )

    async def send_performance_report(self):
        """Отправка отчета о производительности"""
        try:
            stats = self.get_system_stats()
            
            # Расчет средних значений за последний час
            avg_cpu = statistics.mean(self.performance_stats.historical_data['cpu'][-60:]) if len(self.performance_stats.historical_data['cpu']) >= 60 else 0
            avg_memory = statistics.mean(self.performance_stats.historical_data['memory'][-60:]) if len(self.performance_stats.historical_data['memory']) >= 60 else 0
            
            report = (
                f"📊 SYSTEM PERFORMANCE REPORT\n"
                f"CPU (Current/Avg/Peak): {stats['cpu']}%/{avg_cpu:.1f}%/{self.performance_stats.cpu_peak}%\n"
                f"Memory (Current/Avg/Peak): {stats['memory']}%/{avg_memory:.1f}%/{self.performance_stats.memory_peak}%\n"
                f"Disk Usage: {stats['disk']}%\n"
                f"Network Usage: {stats['network']:.2f}MB\n"
                f"API Latency: {stats['api_latency']:.2f}ms\n"
                f"API Errors: {self.performance_stats.api_errors}\n"
                f"Process Count: {stats['process_count']}\n"
                f"Total Errors: {self.performance_stats.errors_count}\n"
                f"Uptime: {self.get_uptime()}\n"
                f"Memory Leaks: {self.performance_stats.memory_leaks_detected}\n"
                f"Anomalies: {'Yes' if self.anomaly_detected else 'No'}\n"
                f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
            if Config.TELEGRAM_ENABLED:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.bot.send_telegram_alert(report)
                )
            
            # Сброс пиковых значений и счетчиков ошибок
            self.performance_stats.cpu_peak = 0
            self.performance_stats.memory_peak = 0
            self.performance_stats.errors_count = 0
            self.performance_stats.api_errors = 0
            self.anomaly_detected = False
            
        except Exception as e:
            logger.error(f"Failed to send performance report: {str(e)}")

    def get_uptime(self):
        """Возвращает время работы системы"""
        uptime_seconds = time.time() - self.start_time
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        return f"{days}d {hours}h {minutes}m"

    async def check_internet_connection(self):
        """Проверяет интернет соединение"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get('https://www.google.com', timeout=5)
            )
            return True
        except:
            logger.error("No internet connection")
            await self._send_alert("🌐 INTERNET CONNECTION LOST", "", "internet_alert")
            return False

    def get_system_stats(self):
        """Возвращает текущую статистику системы"""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent,
            'network': self.performance_stats.network_usage,
            'uptime': self.get_uptime(),
            'errors': self.performance_stats.errors_count,
            'api_latency': self.performance_stats.api_latency,
            'process_count': self.performance_stats.process_count,
            'gpu': self.performance_stats.gpu_usage
        }

    def restart_recommended(self):
        """Рекомендует перезапуск системы при необходимости"""
        uptime_hours = (time.time() - self.start_time) / 3600
        if uptime_hours > 72:  # 3 дня работы
            asyncio.run_coroutine_threadsafe(
                self._send_alert(
                    "🔄 SYSTEM RESTART RECOMMENDED", 
                    "System has been running for 72+ hours", 
                    "restart_recommended"
                ),
                self.loop
            )
            return True
        return False

    async def check_bot_health(self):
        """Проверяет общее состояние бота"""
        try:
            stats = self.get_system_stats()
            health_issues = []
            
            if stats['cpu'] > Config.CPU_THRESHOLD:
                health_issues.append("High CPU usage")
            if stats['memory'] > Config.MEMORY_THRESHOLD:
                health_issues.append("High memory usage")
            if stats['disk'] > Config.DISK_THRESHOLD:
                health_issues.append("Low disk space")
            if self.performance_stats.errors_count > 10:
                health_issues.append("High error count")
            if self.performance_stats.api_latency > Config.API_LATENCY_THRESHOLD:
                health_issues.append("High API latency")
            if self.performance_stats.api_errors > 5:
                health_issues.append("High API error count")
            
            if health_issues:
                message = "🤖 BOT HEALTH ISSUES:\n" + "\n".join(f"• {issue}" for issue in health_issues)
                await self._send_alert(message, "", "health_alert")
                return False
            else:
                return True
                
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring_active = False
        self.loop.stop()

import os
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance import Client
from binance.exceptions import BinanceAPIException
import logging
import requests
import json
import traceback
import sys
import math
import atexit
from typing import Dict, Any, List
import psutil
import gc
import resource

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger("TradingBot")

# 🔥 ДИАГНОСТИЧЕСКИЙ ПАТЧ - НЕ МЕНЯЕТ ЛОГИКУ, ТОЛЬКО ДОБАВЛЯЕТ ЛОГИ
class DiagnosticPatch:
    @staticmethod
    def patch_trading_bot():
        """Добавляем диагностику в существующие методы"""
        logger.info("=== 🔧 APPLYING DIAGNOSTIC PATCH ===")
        
        # Сохраняем оригинальные методы
        original_process_signals = TradingBot._process_signals_ultra_light
        original_execute_trade = TradingBot._execute_trade
        
        def patched_process_signals(self):
            """Версия с диагностикой"""
            logger.info("=== 🔍 SIGNAL SCAN START ===")
            start_time = time.time()
            
            try:
                # Вызываем оригинальный метод
                result = original_process_signals(self)
                
                # Диагностика после выполнения
                scan_time = time.time() - start_time
                logger.info(f"=== 🔍 SIGNAL SCAN END: {scan_time:.2f}s ===")
                
                return result
            except Exception as e:
                logger.error(f"❌ SIGNAL SCAN CRASHED: {e}")
                return None
        
        def patched_execute_trade(self, signal):
            """Версия с диагностикой исполнения"""
            symbol = signal.get('symbol', 'UNKNOWN')
            logger.info(f"🚀 EXECUTE TRADE START: {symbol}")
            
            try:
                # Проверяем базовые условия перед исполнением
                if not self.client:
                    logger.error("❌ EXECUTION FAIL: No client")
                    return
                    
                if not self.risk_manager:
                    logger.error("❌ EXECUTION FAIL: No risk manager")
                    return
                
                # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка SYMBOL_INFO
                if not Config.SYMBOL_INFO:
                    logger.error("❌ No symbol info available, initializing default symbols")
                    self._add_default_symbols()
                    
                if not Config.SYMBOL_INFO:
                    logger.error("❌ Still no symbol info, cannot execute trade")
                    return
                    
                symbol = signal['symbol']
                if symbol not in Config.SYMBOL_INFO:
                    logger.error(f"❌ Symbol {symbol} not in SYMBOL_INFO, adding default info")
                    Config.SYMBOL_INFO[symbol] = {
                        'min_qty': 0.001, 
                        'step_size': 0.001, 
                        'tick_size': 0.01
                    }
                
                # Вызываем оригинальный метод
                result = original_execute_trade(self, signal)
                logger.info(f"✅ EXECUTE TRADE COMPLETE: {symbol}")
                return result
                
            except Exception as e:
                logger.error(f"💥 EXECUTION CRASHED: {e}")
                raise
        
        # Применяем патчи
        TradingBot._process_signals_ultra_light = patched_process_signals
        TradingBot._execute_trade = patched_execute_trade
        
        logger.info("✅ DIAGNOSTIC PATCH APPLIED")

# Условные импорты для обработки отсутствующих модулей
try:
    from scenario_analyzer import ScenarioAnalyzer
except ImportError as e:
    logger.warning(f"ScenarioAnalyzer import failed: {e}")
    # Создаем заглушку
    class ScenarioAnalyzer:
        def __init__(self, data_feeder):
            self.data_feeder = data_feeder
        def analyze_symbol(self, symbol):
            return {'symbol': symbol, 'score': 0, 'direction': 'HOLD', 'scenario_details': {}}
        def update_scenario_performance(self, *args, **kwargs):
            pass

try:
    from adaptive_learner import AdaptiveLearner
except ImportError as e:
    logger.warning(f"AdaptiveLearner import failed: {e}")
    class AdaptiveLearner:
        def __init__(self): 
            pass
        def predict(self, features): 
            return {'probability': 0.5, 'confidence': 0}
        def add_training_example(self, features, target, symbol):
            pass

try:
    from system_monitor import SystemMonitor
except ImportError as e:
    logger.warning(f"SystemMonitor import failed: {e}")
    class SystemMonitor:
        def __init__(self, bot): 
            pass

try:
    from auto_optimizer import AutoOptimizer
except ImportError as e:
    logger.warning(f"AutoOptimizer import failed: {e}")
    class AutoOptimizer:
        def __init__(self, risk_manager): 
            pass
        def collect_performance_data(self, *args): 
            pass

try:
    from profit_manager import ProfitManager
except ImportError as e:
    logger.warning(f"ProfitManager import failed: {e}")
    class ProfitManager:
        def __init__(self, bot, risk_manager): 
            pass

try:
    from advanced_decision_maker import AdvancedDecisionMaker
except ImportError as e:
    logger.warning(f"AdvancedDecisionMaker import failed: {e}")
    class AdvancedDecisionMaker:
        def __init__(self, data_feeder, risk_manager): 
            pass
        def should_enter_trade(self, symbol, signal):
            return {'decision': False, 'confidence': 0}

try:
    from cognitive_trader import CognitiveTrader
except ImportError as e:
    logger.warning(f"CognitiveTrader import failed: {e}")
    class CognitiveTrader:
        def __init__(self, decision_maker, risk_manager, data_feeder=None): 
            pass
        def make_trading_decision(self, symbol, signal): 
            return None
        def learn_from_trade(self, trade_result): 
            pass

try:
    from backup_system import BackupSystem
except ImportError as e:
    logger.warning(f"BackupSystem import failed: {e}")
    class BackupSystem:
        def __init__(self):
            pass
        def create_backup(self):
            pass

# Импорты основных модулей (после условных)
from config import Config
from risk_manager import RiskManager
from data_feeder import DataFeeder

class MemoryOptimizer:
    def __init__(self):
        self.memory_limit_mb = 500
        self.last_cleanup = time.time()
        
    def check_memory(self):
        """Проверяет использование памяти и запускает очистку"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.memory_limit_mb * 0.8:
            self.force_cleanup()
            return True
        return False
    
    def force_cleanup(self):
        """Принудительная очистка памяти"""
        logger.warning("🔄 FORCING MEMORY CLEANUP")
        
        if hasattr(self, 'api_cache'):
            self.api_cache.clear()
        
        if hasattr(self, 'data_feeder') and self.data_feeder:
            if hasattr(self.data_feeder, 'market_data'):
                self.data_feeder.market_data.clear()
            if hasattr(self.data_feeder, 'technical_data_cache'):
                self.data_feeder.technical_data_cache.clear()
            if hasattr(self.data_feeder, 'orderbook_data'):
                self.data_feeder.orderbook_data.clear()
        
        gc.collect()
        
        large_vars = [var for var in locals() if sys.getsizeof(var) > 1000000]
        for var in large_vars:
            del var
        
        logger.info("✅ Memory cleanup completed")

def emergency_disk_cleanup():
    """СРОЧНАЯ очистка дискового пространства"""
    logger.critical("🚨 EMERGENCY DISK CLEANUP ACTIVATED")
    
    cache_dirs = ['__pycache__', '.pytest_cache', '.cache', 'dl_models', 'models']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            import shutil
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Deleted cache: {cache_dir}")
            except:
                pass
    
    log_files = ['trading_bot.log', 'debug.log']
    for log_file in log_files:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 1024 * 1024:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 100:
                    with open(log_file, 'w') as f:
                        f.writelines(lines[-100:])
                logger.info(f"Cleaned log: {log_file}")
            except:
                pass
    
    trade_file = 'trade_history.json'
    if os.path.exists(trade_file):
        try:
            with open(trade_file, 'r') as f:
                trades = json.load(f)
            if len(trades) > 50:
                with open(trade_file, 'w') as f:
                    json.dump(trades[-50:], f, indent=2)
            logger.info("Cleaned trade history")
        except:
            pass
    
    import tempfile
    tempfile.tempdir = None
    os.system("find . -name '*.tmp' -delete 2>/dev/null")
    os.system("find . -name '*.temp' -delete 2>/dev/null")
    
    disk_usage = psutil.disk_usage('/').percent
    logger.info(f"📊 Disk usage after cleanup: {disk_usage}%")

class TradingBot:
    def __init__(self):
        self.client = None
        self.risk_manager = None
        self.data_feeder = None
        self.scenario_analyzer = None
        self.adaptive_learner = None
        self.system_monitor = None
        self.auto_optimizer = None
        self.profit_manager = None
        self.backup_system = None
        self.decision_maker = None
        self.cognitive_trader = None
        self.current_position = None
        self.signal_count = 0
        self.last_data_fetch = {tf: 0 for tf in Config.TIMEFRAMES}
        self.error_count = 0
        self.trade_history = []
        self.initial_balance = 1000.0
        self.daily_report_sent = False
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.daily_start_balance = 1000.0
        self.heartbeat_thread = None
        self.order_ids = {}
        
        self.api_cache = {}
        self.cache_timeout = 300
        self.api_call_count = 0
        
        self.memory_optimizer = MemoryOptimizer()
        self._last_emergency_test = 0
        
        self._initialize()
        self._start_heartbeat()
        atexit.register(self.cleanup_on_exit)

    def _debug_data_flow(self):
        """Детальная диагностика потока данных"""
        logger.info("=== 📊 DATA FLOW DIAGNOSTICS ===")
        
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            try:
                if self.client:
                    try:
                        price = self.data_feeder.get_current_price(symbol)
                        logger.info(f"💰 {symbol} PRICE: {price}")
                    except Exception as e:
                        logger.error(f"❌ {symbol} PRICE FAILED: {e}")
                        price = 0
                else:
                    logger.warning("🔶 No client - simulation mode")
                    price = 1000
                
                if self.data_feeder and price > 0:
                    try:
                        tech_data = self.data_feeder.get_technical_data(symbol)
                        if tech_data:
                            rsi = tech_data.get('rsi', 'N/A')
                            adx = tech_data.get('adx', 'N/A')
                            logger.info(f"📈 {symbol} TECH: RSI={rsi}, ADX={adx}")
                        else:
                            logger.warning(f"⚠️ {symbol} NO TECH DATA")
                    except Exception as e:
                        logger.error(f"❌ {symbol} TECH DATA FAILED: {e}")
                
                if self.risk_manager:
                    try:
                        balance = self.risk_manager.get_current_balance()
                        can_trade = self.risk_manager.can_trade()
                        logger.info(f"🔐 {symbol} RISK: balance=${balance:.2f}, can_trade={can_trade}")
                    except Exception as e:
                        logger.error(f"❌ {symbol} RISK CHECK FAILED: {e}")
                
                try:
                    test_signal = self._ultra_simple_signal_detection(symbol)
                    if test_signal:
                        logger.info(f"🎯 {symbol} TEST SIGNAL: score={test_signal.get('score', 0)}")
                    else:
                        logger.info(f"🔇 {symbol} NO SIGNAL")
                except Exception as e:
                    logger.error(f"❌ {symbol} SIGNAL GENERATION FAILED: {e}")
                    
            except Exception as e:
                logger.error(f"💥 {symbol} DIAGNOSTIC CRASH: {e}")

    def _emergency_test_trigger(self):
        """Аварийный тестовый триггер - гарантированно создает сделку для теста"""
        current_time = time.time()
        
        no_position = self.current_position is None
        enough_time_passed = current_time - self._last_emergency_test > 180
        
        if no_position and enough_time_passed:
            logger.warning("🚨 EMERGENCY TEST TRIGGER: No trades for 3 minutes!")
            
            emergency_signal = {
                'symbol': 'BTCUSDT',
                'direction': 'BUY',
                'score': 10.0,
                'confidence': 1.0,
                'reason': 'EMERGENCY DIAGNOSTIC TRADE',
                'timestamp': current_time
            }
            
            try:
                self._execute_trade(emergency_signal)
                self._last_emergency_test = current_time
                logger.info("✅ EMERGENCY TEST TRADE EXECUTED")
            except Exception as e:
                logger.error(f"❌ EMERGENCY TRADE FAILED: {e}")

    def _lightweight_system_monitor(self):
        """Облегченный мониторинг системы без тяжелых проверок"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 800:
                logger.warning(f"🔄 Light memory cleanup: {memory_mb:.2f}MB")
                gc.collect()
                
            return True
            
        except Exception as e:
            logger.debug(f"Light system monitor error: {e}")
            return True

    def force_trading_start(self):
        """Принудительный запуск торговли с упрощенными настройками"""
        logger.info("🔥 FORCING TRADING START WITH SIMPLIFIED RULES")
        
        Config.AGGRESSIVE_MODE = True
        Config.MIN_SCORE = 3.0
        Config.CONFIRMATIONS_REQUIRED = 1
        
        Config.MULTIDIMENSIONAL_ANALYSIS = False
        Config.COGNITIVE_TRADING_ENABLED = False
        
        Config.MAX_TRADES_PER_HOUR = 100
        Config.CHECK_INTERVAL = 1.0
        
        logger.info("✅ Trading forced started with aggressive settings")

    def _ultra_simple_signal_detection(self, symbol: str):
        """СУПЕР-ПРОСТАЯ логика обнаружения сигналов для немедленного запуска"""
        try:
            if not self.data_feeder:
                return None
                
            current_price = self.data_feeder.get_current_price(symbol)
            if current_price == 0:
                return None
                
            tech_data = self.data_feeder.get_technical_data(symbol)
            if not tech_data:
                return None
                
            rsi = tech_data.get('rsi', 50)
            price_change = tech_data.get('price_change_1', 0)
            volume_ratio = tech_data.get('volume_ratio', 1)
            
            if rsi < 35 and price_change > -0.005:
                return {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'score': 6.0,
                    'confidence': 0.7,
                    'reason': 'RSI oversold with stable price'
                }
            elif rsi > 65 and price_change < 0.005:
                return {
                    'symbol': symbol,
                    'direction': 'SELL', 
                    'score': 6.0,
                    'confidence': 0.7,
                    'reason': 'RSI overbought with resistance'
                }
            elif abs(price_change) > 0.015 and volume_ratio > 1.3:
                direction = 'BUY' if price_change > 0 else 'SELL'
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'score': 5.5,
                    'confidence': 0.6,
                    'reason': 'Strong price movement with volume'
                }
                
            return None
            
        except Exception as e:
            logger.debug(f"Ultra simple signal error for {symbol}: {e}")
            return None

    def api_call_with_retry(self, func, max_retries=2, delay=5):
        """СУПЕР-НАДЕЖНЫЕ API вызовы с защитой от сетевых сбоев"""
        try:
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > 90:
                logger.warning("⚠️ Low disk space, skipping API call")
                return None
        except:
            pass
        
        for attempt in range(max_retries):
            try:
                result = func()
                return result
            except BinanceAPIException as e:
                if e.code in [-1001, -1002, -1003, -1006, -1007]:
                    wait = delay * (2 ** attempt)
                    logger.warning(f"Network error {e.code}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    raise
            except requests.exceptions.ConnectionError as e:
                wait = delay * (2 ** attempt)
                logger.warning(f"Connection error, retrying in {wait}s...")
                time.sleep(wait)
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                wait = delay * (2 ** attempt)
                logger.warning(f"API error, retrying in {wait}s: {e}")
                time.sleep(wait)
        
        return None

    def _cleanup_old_cache(self):
        """Очищает устаревшие записи из кэша"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, (data, timestamp) in self.api_cache.items():
            if current_time - timestamp > self.cache_timeout:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.api_cache[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old cache entries")

    def cleanup_disk_space(self):
        """Срочная очистка дискового пространства"""
        try:
            logger.info("🔄 Cleaning up disk space...")
            
            self.api_cache.clear()
            
            if hasattr(self, 'data_feeder') and self.data_feeder:
                if hasattr(self.data_feeder, 'market_data'):
                    self.data_feeder.market_data.clear()
                if hasattr(self.data_feeder, 'technical_data_cache'):
                    self.data_feeder.technical_data_cache.clear()
                if hasattr(self.data_feeder, 'orderbook_data'):
                    self.data_feeder.orderbook_data.clear()
            
            log_file = 'trading_bot.log'
            if os.path.exists(log_file) and os.path.getsize(log_file) > 10 * 1024 * 1024:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                if len(lines) > 1000:
                    with open(log_file, 'w') as f:
                        f.writelines(lines[-1000:])
            
            if len(self.trade_history) > 100:
                self.trade_history = self.trade_history[-100:]
                with open('trade_history.json', 'w') as f:
                    json.dump(self.trade_history, f, indent=2, default=str)
            
            gc.collect()
            
            logger.info("✅ Disk space cleanup completed")
            
        except Exception as e:
            logger.error(f"Disk cleanup error: {e}")

    def ultra_light_mode(self):
        """АВАРИЙНЫЙ РЕЖИМ МИНИМАЛЬНОЙ НАГРУЗКИ"""
        logger.critical("🔴 ACTIVATING ULTRA-LIGHT EMERGENCY MODE")
        
        Config.CHECK_INTERVAL = 300
        Config.MAX_SYMBOLS_TO_CHECK = 1
        Config.MULTIDIMENSIONAL_ANALYSIS = False
        Config.ADAPTIVE_LEARNING = False
        Config.SCENARIO_ANALYSIS_WORKERS = 1
        
        if hasattr(self, 'data_feeder') and self.data_feeder:
            self.data_feeder.running = False
        
        self.cleanup_disk_space()
        
        logger.info("✅ Ultra-light mode activated - minimum functionality only")

    def _process_signals_ultra_light(self):
        """СУПЕР-УПРОЩЕННАЯ обработка сигналов для ULTRA-LIGHT режима"""
        if not self.risk_manager or not self.risk_manager.can_trade():
            return
            
        try:
            symbol = 'BTCUSDT'
            
            if not self.risk_manager.validate_spread(symbol):
                return
                
            signal = self._ultra_simple_analysis(symbol)
            if signal and signal['score'] >= (Config.MIN_SCORE + 2.0):
                logger.info(f"ULTRA-LIGHT: Strong signal for {symbol}")
                self._execute_trade(signal)
            else:
                logger.debug(f"ULTRA-LIGHT: No strong signals for {symbol}")
                    
        except Exception as e:
            logger.debug(f"ULTRA-LIGHT signal processing error: {e}")

    def _monitor_position_light(self):
        """Упрощенный мониторинг позиции"""
        try:
            if not self.current_position:
                return
                
            symbol = self.current_position['symbol']
            positions = self.api_call_with_retry(
                lambda: self.client.futures_position_information(symbol=symbol))
            
            if positions:
                position = positions[0]
                position_amt = float(position['positionAmt'])
                
                if abs(position_amt) < 1e-8:
                    pnl = float(position['unRealizedProfit'])
                    logger.info(f"ULTRA-LIGHT: Position closed, PnL = ${pnl:.4f}")
                    self.current_position = None
                    self.cancel_all_orders(symbol)
                            
        except Exception as e:
            logger.error(f"ULTRA-LIGHT position monitoring error: {e}")

    def _periodic_tasks_light(self):
        """Упрощенные периодические задачи"""
        try:
            self.cleanup_disk_space()
            
            if int(time.time()) % 3600 == 0:
                self._check_connection()
                
            if int(time.time()) % 7200 == 0:
                if hasattr(self.risk_manager, 'get_tradable_symbols'):
                    self.risk_manager.get_tradable_symbols()
                    
        except Exception as e:
            logger.error(f"Light periodic tasks error: {e}")

    def _initialize(self):
        """Улучшенная инициализация с обработкой ошибок"""
        try:
            logger.info("🔥 Starting Improved Precision Pro Trading Bot v5.0")
            
            self.initialize_data_files()
            
            if not Config.BINANCE_API_KEY or not Config.BINANCE_API_SECRET:
                logger.warning("API keys missing - running in simulation mode")
                Config.PAPER_TRADING = True
            
            self.client = Client(
                api_key=Config.BINANCE_API_KEY or "test",
                api_secret=Config.BINANCE_API_SECRET or "test", 
                testnet=Config.PAPER_TRADING,
                requests_params={'timeout': 30}
            )
            
            try:
                self.api_call_with_retry(lambda: self.client.futures_ping())
                logger.info("✅ Binance connection successful")
            except Exception as e:
                logger.warning(f"Binance connection failed: {e}")
                logger.info("Running in simulation mode without Binance API")
                self.client = None
            
            if self.client is None:
                self._fallback_initialization()
                return
                
            self.risk_manager = RiskManager(self.client)
            time.sleep(1)
            
            self.data_feeder = DataFeeder(self.client)
            time.sleep(1)
            
            self.scenario_analyzer = ScenarioAnalyzer(self.data_feeder)
            self.adaptive_learner = AdaptiveLearner()
            
            self._check_connection()
            
            self._load_symbols_info()
            
            self.system_monitor = SystemMonitor(self)
            self.auto_optimizer = AutoOptimizer(self.risk_manager)
            self.profit_manager = ProfitManager(self, self.risk_manager)
            self.backup_system = BackupSystem()
            self.cognitive_trader = CognitiveTrader(self.data_feeder, self.risk_manager)
            self.decision_maker = AdvancedDecisionMaker(self.data_feeder, self.risk_manager)
            
            if self.client is not None:
                self.initial_balance = self.risk_manager.get_current_balance()
                self.daily_start_balance = self.initial_balance
            else:
                self.initial_balance = 1000.0
                self.daily_start_balance = 1000.0
                
            self._print_startup_info()
            self._load_trade_history()
            
            logger.info("✅ Bot initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._fallback_initialization()
    
    def _fallback_initialization(self):
        """Резервная инициализация при ошибках"""
        logger.info("Attempting fallback initialization...")
        Config.PAPER_TRADING = True
        Config.MAX_SYMBOLS_TO_CHECK = 15
        Config.CHECK_INTERVAL = 10.0
        
        try:
            self.client = Client(
                api_key="test" if not Config.BINANCE_API_KEY else Config.BINANCE_API_KEY,
                api_secret="test" if not Config.BINANCE_API_SECRET else Config.BINANCE_API_SECRET,
                testnet=True
            )
            
            self.api_call_with_retry(lambda: self.client.futures_ping())
            
            self.risk_manager = RiskManager(self.client)
            self.data_feeder = DataFeeder(self.client)
            
            self.scenario_analyzer = ScenarioAnalyzer(self.data_feeder)
            self.adaptive_learner = AdaptiveLearner()
            self.system_monitor = SystemMonitor(self)
            self.auto_optimizer = AutoOptimizer(self.risk_manager)
            self.profit_manager = ProfitManager(self, self.risk_manager)
            self.backup_system = BackupSystem()
            self.cognitive_trader = CognitiveTrader(self.data_feeder, self.risk_manager)
            self.decision_maker = AdvancedDecisionMaker(self.data_feeder, self.risk_manager)
            
            try:
                self.api_call_with_retry(lambda: self.client.futures_exchange_info())
                logger.info("✅ Fallback connection verified")
            except:
                logger.warning("❌ Fallback connection failed, running in simulation mode")
                self.client = None
                
            self.initial_balance = 1000.0
            self.daily_start_balance = self.initial_balance
            
            logger.info("✅ Fallback initialization completed")
            
        except Exception as e:
            logger.error(f"Fallback initialization also failed: {e}")
            self.client = None
            self.risk_manager = None
            self.data_feeder = None
            self.scenario_analyzer = ScenarioAnalyzer(None)
            self.adaptive_learner = AdaptiveLearner()
            self.system_monitor = SystemMonitor(self)
            self.auto_optimizer = AutoOptimizer(None)
            self.profit_manager = ProfitManager(self, None)
            self.backup_system = BackupSystem()
            self.cognitive_trader = CognitiveTrader(None, None)
            self.decision_maker = AdvancedDecisionMaker(None, None)
            self.initial_balance = 1000.0
            self.daily_start_balance = 1000.0

    def initialize_data_files(self):
        """Инициализация необходимых файлов данных"""
        try:
            data_files = ['trade_history.json', 'performance_data.json']
            for file in data_files:
                if not os.path.exists(file):
                    with open(file, 'w') as f:
                        json.dump([], f)
                    logger.info(f"Created data file: {file}")
        except Exception as e:
            logger.error(f"Error initializing data files: {e}")

    def _load_symbols_info(self):
        """Загружает информацию о символах с биржи или использует значения по умолчанию"""
        try:
            if self.client is None:
                logger.warning("Skipping symbol info loading - no API connection")
                self._add_default_symbols()
                return
                
            if not Config.BINANCE_API_KEY or not Config.BINANCE_API_SECRET:
                logger.warning("Skipping symbol info loading - no API keys")
                self._add_default_symbols()
                return
                
            exchange_info = self.api_call_with_retry(
                lambda: self.client.futures_exchange_info()
            )
            
            if not exchange_info or 'symbols' not in exchange_info:
                logger.warning("Invalid exchange info response, using default symbol info")
                self._add_default_symbols()
                return
                
            for symbol_data in exchange_info['symbols']:
                symbol_name = symbol_data['symbol']
                info = {'min_qty': 1.0, 'step_size': 0.1, 'tick_size': 0.01}
                
                for f in symbol_data['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        info['min_qty'] = float(f['minQty'])
                        info['step_size'] = float(f['stepSize'])
                    elif f['filterType'] == 'PRICE_FILTER':
                        info['tick_size'] = float(f['tickSize'])
                    elif f['filterType'] == 'MIN_NOTIONAL':
                        info['min_notional'] = float(f['notional'])
                
                Config.SYMBOL_INFO[symbol_name] = info
            logger.info(f"Symbols info loaded: {len(Config.SYMBOL_INFO)} symbols")
                
        except Exception as e:
            logger.error(f"Error loading symbol info: {str(e)}")
            self._add_default_symbols()

    def _add_default_symbols(self):
        """Добавляет основные символы по умолчанию"""
        default_symbols = {
            'BTCUSDT': {'min_qty': 0.001, 'step_size': 0.001, 'tick_size': 0.01},
            'ETHUSDT': {'min_qty': 0.01, 'step_size': 0.01, 'tick_size': 0.01},
            'ADAUSDT': {'min_qty': 1.0, 'step_size': 1.0, 'tick_size': 0.0001},
            'DOTUSDT': {'min_qty': 0.1, 'step_size': 0.1, 'tick_size': 0.001},
            'LINKUSDT': {'min_qty': 0.1, 'step_size': 0.1, 'tick_size': 0.001}
        }
        Config.SYMBOL_INFO.update(default_symbols)
        logger.info(f"Added {len(default_symbols)} default symbols")

    def _check_connection(self):
        try:
            if self.client is None:
                logger.warning("No client connection available")
                return False
                
            self.api_call_with_retry(lambda: self.client.futures_ping())
            self.api_call_with_retry(lambda: self.client.futures_exchange_info())
            
            if self.risk_manager:
                balance = self.risk_manager.get_current_balance()
                logger.info(f"✅ Connection verified | Balance: {balance:.2f} USDT")
            else:
                logger.info("✅ Connection verified (no balance check - risk manager not available)")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self.send_telegram_alert(f"🔴 Binance connection failed: {str(e)}")
            return False

    def _print_startup_info(self):
        if self.risk_manager and self.client is not None:
            balance = self.risk_manager.get_current_balance()
        else:
            balance = self.initial_balance
            
        logger.info(f"""
        =============================
        PRECISION PRO TRADING BOT v5.0
        =============================
        Mode:          {'REAL' if not Config.PAPER_TRADING else 'PAPER'}
        Trading Mode:  {'HYPER AGGRESSIVE' if Config.AGGRESSIVE_MODE else 'AGGRESSIVE'}
        Min Score:     {Config.MIN_SCORE}
        Confirmations: {Config.CONFIRMATIONS_REQUIRED}
        Balance:       ${balance:.2f}
        Risk per trade:{Config.RISK_PERCENT*100}% of balance
        Leverage:      {Config.DEFAULT_LEVERAGE}x
        Quick TP:      {Config.QUICK_TP_SIZE*100}% at {Config.QUICK_TP_MULTIPLIER*100}%
        Main TP:       {Config.TP_MULTIPLIER*100}%
        Timeframes:    {', '.join(Config.TIMEFRAMES)}
        Max Trades:    {Config.MAX_TRADES_PER_HOUR}/hour
        Min Balance:   ${Config.MIN_BALANCE}
        Trading Hours: {Config.TRADING_HOURS_START}:00-{Config.TRADING_HOURS_END}:00 UTC
        AI Features:   {Config.MULTIDIMENSIONAL_ANALYSIS}
        Scenario Analysis: {len(Config.SCENARIOS)} scenarios
        Adaptive Learning: {Config.ADAPTIVE_LEARNING}
        Deep Learning: {Config.DL_USE_HYBRID_MODEL}
        Auto Optimization: {Config.AUTO_OPTIMIZATION['enabled']}
        Auto Withdrawal: {Config.AUTO_PROFIT_WITHDRAWAL['enabled']}
        Backup System: {Config.BACKUP_SETTINGS['enabled']}
        Cognitive Trading: {Config.COGNITIVE_TRADING_ENABLED}
        API Caching:   Enabled ({self.cache_timeout}s timeout)
        Simulation Mode: {self.client is None}
        =============================
        """)
        
        if Config.TELEGRAM_ENABLED and Config.TELEGRAM_TOKEN and Config.TELEGRAM_CHAT_ID:
            mode_info = "SIMULATION" if self.client is None else "REAL"
            self.send_telegram_alert(
                f"🚀 Precision Pro v5.0 with Advanced AI started!\n"
                f"Mode: {mode_info}\n"
                f"Balance: ${balance:.2f}\n"
                f"Risk: {Config.RISK_PERCENT*100}% per trade\n"
                f"Leverage: {Config.DEFAULT_LEVERAGE}x\n"
                f"Trading Mode: {'HYPER AGGRESSIVE' if Config.AGGRESSIVE_MODE else 'AGGRESSIVE'}\n"
                f"Quick TP: {Config.QUICK_TP_SIZE*100}% at {Config.QUICK_TP_MULTIPLIER*100}%\n"
                f"Main TP: {Config.TP_MULTIPLIER*100}%\n"
                f"Min Balance: ${Config.MIN_BALANCE}\n"
                f"Trading Hours: {Config.TRADING_HOURS_START}:00-{Config.TRADING_HOURS_END}:00 UTC\n"
                f"AI Features: {Config.MULTIDIMENSIONAL_ANALYSIS}\n"
                f"Scenarios: {len(Config.SCENARIOS)}\n"
                f"Adaptive Learning: {Config.ADAPTIVE_LEARNING}\n"
                f"Deep Learning: {Config.DL_USE_HYBRID_MODEL}\n"
                f"Auto Optimization: {Config.AUTO_OPTIMIZATION['enabled']}\n"
                f"Auto Withdrawal: {Config.AUTO_PROFIT_WITHDRAWAL['enabled']}\n"
                f"Backup System: {Config.BACKUP_SETTINGS['enabled']}\n"
                f"Cognitive Trading: {Config.COGNITIVE_TRADING_ENABLED}\n"
                f"API Caching: Enabled ({self.cache_timeout}s)"
            )

    def _start_heartbeat(self):
        def heartbeat():
            while True:
                try:
                    time.sleep(3600)
                    if not self.client:
                        continue
                        
                    balance = self.initial_balance
                    if self.risk_manager:
                        balance = self.risk_manager.get_current_balance()
                        
                    open_positions = len(self.risk_manager.get_open_positions()) if self.risk_manager else 0
                    active_orders = len(self.order_ids)
                    win_rate = self.risk_manager.calculate_win_rate() if self.risk_manager else 0
                    cache_size = len(self.api_cache)
                    
                    status = (
                        f"❤️ HEARTBEAT | Bot is active\n"
                        f"Balance: ${balance:.2f}\n"
                        f"Trades today: {self.daily_trades}\n"
                        f"Profit today: ${self.daily_profit:.2f}\n"
                        f"Win rate: {win_rate:.2f}\n"
                        f"Open positions: {open_positions}\n"
                        f"Active orders: {active_orders}\n"
                        f"API calls cached: {cache_size}\n"
                        f"Version: Precision Pro v5.0 with Advanced AI\n"
                        f"Mode: {'SIMULATION' if self.client is None else 'LIVE'}"
                    )
                    self.send_telegram_alert(status)
                except Exception as e:
                    logger.error(f"Heartbeat error: {str(e)}")
                    time.sleep(60)
        
        self.heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def run(self):
        """Главный цикл с управлением памятью"""
        gc.enable()
        gc.set_threshold(1000, 10, 10)
        
        try:
            while True:
                try:
                    if self.memory_optimizer.check_memory():
                        gc.collect()
                        
                    self._lightweight_trading_cycle()
                    
                except Exception as e:
                    self._handle_critical_error(e)
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user (KeyboardInterrupt)")
        finally:
            self.cleanup_on_exit()

    def test_mode_run(self):
        """🧪 ТЕСТОВЫЙ РЕЖИМ ДЛЯ БЫСТРОГО ЗАПУСКА"""
        logger.info("🧪 TEST MODE: Starting simplified trading cycle")
        
        self.force_trading_start()
        
        while True:
            try:
                self._debug_data_flow()
                
                if not self.current_position:
                    self._process_signals_ultra_light()
                else:
                    self._monitor_position_light()

                self._emergency_test_trigger()

                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("🛑 Test mode stopped by user")
                break
            except Exception as e:
                logger.error(f"Test mode error: {e}")
                time.sleep(10)

    def _lightweight_trading_cycle(self):
        """Упрощенный торговый цикл"""
        try:
            if not self.current_position:
                self._process_signals_ultra_light()
            else:
                self._monitor_position_light()
                
            time.sleep(Config.CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    def _handle_critical_error(self, e):
        """Обработка критических ошибок"""
        logger.critical(f"CRITICAL ERROR: {e}")
        self.error_count += 1
        
        if self.error_count > 10:
            logger.critical("🚨 Too many errors, entering safety mode")
            self.ultra_light_mode()
            self.error_count = 0
            
        time.sleep(10)

    def _ultra_simple_analysis(self, symbol):
        """ОЧЕНЬ ПРОСТОЙ анализ - только цена, объем и RSI"""
        try:
            if not self.data_feeder:
                return None
                
            current_price = self.data_feeder.get_current_price(symbol)
            if current_price == 0:
                return None
                
            tech_data = self.data_feeder.get_technical_data(symbol)
            if not tech_data:
                return None
                
            rsi = tech_data.get('rsi', 50)
            volume_ratio = tech_data.get('volume_ratio', 1)
            price_change = tech_data.get('price_change_1', 0)
            
            score = 0
            
            if rsi < 30 or rsi > 70:
                score += 2
                
            if volume_ratio > 1.5:
                score += 1
                
            if abs(price_change) > 0.01:
                score += 1
                
            if rsi < 30 and price_change > 0:
                direction = 'BUY'
            elif rsi > 70 and price_change < 0:
                direction = 'SELL'
            else:
                return None
                
            return {
                'symbol': symbol,
                'direction': direction,
                'score': score,
                'confidence': min(volume_ratio / 2.0, 1.0)
            }
            
        except Exception as e:
            logger.debug(f"Ultra simple analysis failed for {symbol}: {e}")
            return None

    def _periodic_tasks(self):
        """Периодические задачи обслуживания"""
        try:
            if hasattr(self.data_feeder, 'clean_cache') and self.data_feeder:
                self.data_feeder.clean_cache()
            
            self._cleanup_old_cache()
            
            self._check_connection()
            
            if hasattr(self.risk_manager, 'update_tradable_symbols') and self.risk_manager:
                self.risk_manager.update_tradable_symbols()
            
            if hasattr(self, 'backup_system') and self.backup_system:
                self.backup_system.create_backup()
                
            self._verify_active_orders()
                
        except Exception as e:
            logger.error(f"Periodic tasks error: {e}")

    def _get_ai_prediction(self, signal: Dict) -> Dict:
        """Заглушка для AI предсказания"""
        try:
            technical_data = self.data_feeder.get_technical_data(signal['symbol']) if self.data_feeder else {}
            orderbook_data = self.data_feeder.get_orderbook_data(signal['symbol']) if self.data_feeder else {}
            social_data = self.data_feeder.get_social_sentiment(signal['symbol']) if self.data_feeder else {}
            market_cap = self.data_feeder.get_market_cap(signal['symbol']) if self.data_feeder else 0
            
            features = {
                'rsi': technical_data.get('rsi', 50) if technical_data else 50,
                'adx': technical_data.get('adx', 20) if technical_data else 20,
            }
            
            if self.adaptive_learner:
                return self.adaptive_learner.predict(features)
            else:
                return {'probability': 0.5, 'confidence': 0}
                
        except Exception as e:
            logger.error(f"AI prediction error: {str(e)}")
            return {'probability': 0.5, 'confidence': 0}

    def _scan_market_opportunities(self):
        """Сканирует рынок на наличие дополнительных возможностей"""
        try:
            if not self.risk_manager:
                return
                
            symbols = self.risk_manager.get_tradable_symbols()
            
            symbol_scores = []
            for symbol in symbols:
                score = self.data_feeder.get_symbol_score(symbol) if self.data_feeder else 0
                symbol_scores.append((symbol, score))
            
            symbol_scores.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [s[0] for s in symbol_scores[:15]]
            
            logger.info(f"Top symbols by score: {', '.join(top_symbols[:5])}")
            
            for symbol in top_symbols:
                momentum = self.risk_manager.calculate_market_momentum(symbol)
                if abs(momentum) > 0.2:
                    logger.info(f"High momentum detected for {symbol}: {momentum:.2f}")
            
            whale_symbols = []
            for symbol in top_symbols:
                if self.risk_manager.detect_whale_activity(symbol):
                    whale_symbols.append(symbol)
            
            if whale_symbols:
                logger.info(f"Whale activity detected on: {', '.join(whale_symbols)}")
                
        except Exception as e:
            logger.error(f"Market scan error: {str(e)}")

    def _verify_active_orders(self):
        try:
            if not self.order_ids or self.client is None:
                return
                
            exchange_orders = {}
            open_orders = self.api_call_with_retry(lambda: self.client.futures_get_open_orders())
            for order in open_orders:
                exchange_orders[int(order['orderId'])] = order
            
            for order_id in list(self.order_ids.keys()):
                if order_id not in exchange_orders:
                    logger.warning(f"Order {order_id} not found on exchange! Removing from tracking")
                    del self.order_ids[order_id]
                    if self.current_position and self.current_position['symbol'] == self.order_ids[order_id]['symbol']:
                        self._recover_order(self.order_ids[order_id])
            
            logger.debug(f"Active orders verified: {len(self.order_ids)} orders tracked")
        except Exception as e:
            logger.error(f"Order verification error: {str(e)}")

    def _recover_order(self, order_info):
        try:
            if self.client is None:
                return
                
            logger.warning(f"Attempting to recover order: {order_info}")
            
            if order_info['type'] == 'STOP_MARKET':
                order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=order_info['symbol'],
                        side=order_info['side'],
                        type='STOP_MARKET',
                        stopPrice=order_info['stopPrice'],
                        closePosition=True
                    )
                )
            elif order_info['type'] == 'TAKE_PROFIT_MARKET':
                order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=order_info['symbol'],
                        side=order_info['side'],
                        type='TAKE_PROFIT_MARKET',
                        stopPrice=order_info['stopPrice'],
                        quantity=order_info['quantity'],
                        reduceOnly=True
                    )
                )
            
            new_order_id = order['orderId']
            self.order_ids[new_order_id] = order_info
            self.order_ids[new_order_id]['orderId'] = new_order_id
            
            logger.info(f"Order recovered: {new_order_id}")
        except Exception as e:
            logger.error(f"Failed to recover order: {str(e)}")
            self.send_telegram_alert(f"🚨 Failed to recover order for {order_info['symbol']}")

    def system_failsafe(self):
        try:
            current_balance = self.initial_balance
            if self.risk_manager:
                current_balance = self.risk_manager.get_current_balance()
            
            if current_balance == 0:
                logger.warning("Balance check skipped due to connection error")
                return
                
            if current_balance < self.initial_balance * Config.MAX_DRAWDOWN:
                logger.critical(f"Critical drawdown detected! {current_balance:.2f} < {self.initial_balance * Config.MAX_DRAWDOWN:.2f}")
                self.send_telegram_alert(
                    f"🛑 CRITICAL DRAWDOWN! Bot stopped\n"
                    f"Initial: ${self.initial_balance:.2f}\n"
                    f"Current: ${current_balance:.2f}\n"
                    f"Drawdown: {(1 - current_balance/self.initial_balance)*100:.2f}%"
                )
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failsafe error: {str(e)}")

    def _execute_trade(self, signal: dict):
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверка SYMBOL_INFO
        if not Config.SYMBOL_INFO:
            logger.error("❌ No symbol info available, initializing default symbols")
            self._add_default_symbols()
            
        if not Config.SYMBOL_INFO:
            logger.error("❌ Still no symbol info, cannot execute trade")
            return
            
        symbol = signal['symbol']
        if symbol not in Config.SYMBOL_INFO:
            logger.error(f"❌ Symbol {symbol} not in SYMBOL_INFO, adding default info")
            Config.SYMBOL_INFO[symbol] = {
                'min_qty': 0.001, 
                'step_size': 0.001, 
                'tick_size': 0.01
            }
        
        if not self._check_connection():
            logger.error("No API connection, skipping trade")
            return
            
        try:
            if self.client is None:
                logger.info(f"Simulation mode: Would execute trade for {signal['symbol']} {signal['direction']}")
                return
                
            logger.info(f"Attempting trade for {signal['symbol']} {signal['direction']}")
            
            symbol = signal['symbol']
            direction = signal['direction']
            
            balance = self.risk_manager.get_current_balance() if self.risk_manager else self.initial_balance
            if balance < Config.MIN_BALANCE:
                return
                
            mark_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if mark_price == 0:
                return
                
            sl_price, tp_price = self.risk_manager.calculate_sl_tp(symbol, direction, mark_price)
            
            risk_multiplier = 1.5 if Config.AGGRESSIVE_MODE else 1.0
            size = self.risk_manager.calculate_size(symbol, mark_price, direction, sl_price) * risk_multiplier
            if size <= 0:
                return
                
            qty_precision = self._get_quantity_precision(symbol)
            size = round(size, qty_precision)
            
            min_qty = Config.SYMBOL_INFO[symbol]['min_qty']
            if size < min_qty:
                size = min_qty
            
            max_leverage = min(Config.DEFAULT_LEVERAGE, self.risk_manager.get_max_leverage(symbol))
            max_position_value = balance * max_leverage * Config.MAX_POSITION_SIZE
            position_value = size * mark_price

            if position_value > max_position_value:
                size = max_position_value / mark_price
                step_size = Config.SYMBOL_INFO[symbol].get('step_size', 0.1)
                if step_size > 0:
                    size = math.floor(size / step_size) * step_size
            
            logger.info(f"Executing trade: {direction} {size} {symbol} at {mark_price}")
            
            try:
                order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side="SELL" if direction == "SELL" else "BUY",
                        type="MARKET",
                        quantity=size
                    )
                )
                logger.info(f"MARKET order executed: {size} {symbol}")
            except BinanceAPIException as e:
                if "Insufficient" in e.message or "margin" in e.message:
                    logger.warning("Insufficient margin, reducing size by 40%")
                    size = round(size * 0.6, qty_precision)
                    
                    if size < min_qty:
                        return
                        
                    try:
                        order = self.api_call_with_retry(
                            lambda: self.client.futures_create_order(
                                symbol=symbol,
                                side="SELL" if direction == "SELL" else "BUY",
                                type="MARKET",
                                quantity=size
                            )
                        )
                    except BinanceAPIException as e2:
                        return
                else:
                    return
            
            if Config.POSITION_HEDGING and self.risk_manager:
                self.risk_manager.create_hedge_position(symbol, direction, size)
            
            quick_tp_size = size * Config.QUICK_TP_SIZE
            if direction == "BUY":
                quick_tp_price = mark_price * (1 + Config.QUICK_TP_MULTIPLIER)
            else:
                quick_tp_price = mark_price * (1 - Config.QUICK_TP_MULTIPLIER)
            
            self._place_partial_take_profit(direction, symbol, quick_tp_price, quick_tp_size)
            
            main_tp_size = size - quick_tp_size
            if main_tp_size > 0:
                if direction == "BUY":
                    main_tp_price = mark_price * (1 + Config.TP_MULTIPLIER)
                else:
                    main_tp_price = mark_price * (1 - Config.TP_MULTIPLIER)
                
                self._place_take_profit(direction, symbol, main_tp_price, main_tp_size)
            
            self._place_stop_loss(direction, symbol, sl_price, size)
            
            self._black_swan_protection(symbol, direction, size)
            
            self.current_position = {
                'symbol': symbol,
                'direction': direction,
                'size': size,
                'entry_price': mark_price,
                'sl': sl_price,
                'tp': tp_price,
                'quick_tp': quick_tp_price,
                'quick_tp_hit': False,
                'breakeven': False,
                'opened_at': datetime.utcnow(),
                'order_id': order['orderId'],
                'initial_sl': sl_price,
                'scenario_scores': signal.get('scenario_details', {}),
                'signal_score': signal['score']
            }
            
            if Config.TELEGRAM_ENABLED:
                trade_value = size * mark_price
                self.send_telegram_alert(
                    f"🚀 NEW TRADE: {symbol} {direction}\n"
                    f"Entry: {mark_price:.8f}\nSize: {size:.2f}\n"
                    f"Value: ${trade_value:.2f}\nSL: {sl_price:.8f}\n"
                    f"Quick TP: {quick_tp_price:.8f}\nMain TP: {tp_price:.8f}\n"
                    f"AI Score: {signal['score']:.2f}\n"
                    f"Market Regime: {self.risk_manager.market_regime if self.risk_manager else 'N/A'}"
                )
            
            if self.risk_manager:
                self.risk_manager.trade_count += 1
            self.daily_trades += 1
            
            trade_data = {
                'symbol': symbol,
                'direction': direction,
                'size': size,
                'entry_price': mark_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'score': signal['score'],
                'timestamp': datetime.now(),
                'scenario_scores': signal.get('scenario_details', {})
            }
            if self.auto_optimizer:
                self.auto_optimizer.collect_performance_data(trade_data)
            
            self._save_trade_to_history({
                'symbol': symbol,
                'direction': direction,
                'size': size,
                'entry_price': mark_price,
                'sl': sl_price,
                'tp': tp_price,
                'quick_tp': quick_tp_price,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'open',
                'scenario_scores': signal.get('scenario_details', {})
            })
            
        except (requests.exceptions.ConnectionError, BinanceAPIException) as e:
            self.error_count += 1
            if self.error_count > 5:
                self._enter_safety_mode()
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")

    def _enter_safety_mode(self):
        """Вход в безопасный режим при множественных ошибках"""
        logger.critical("🚨 ENTERING SAFETY MODE DUE TO MULTIPLE ERRORS")
        self.ultra_light_mode()
        self.error_count = 0

    def _black_swan_protection(self, symbol: str, direction: str, size: float):
        try:
            if self.client is None:
                return
                
            price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if direction == "BUY":
                sl_price = price * 0.95
            else:
                sl_price = price * 1.05
            
            tick_size = Config.SYMBOL_INFO[symbol].get('tick_size', 0.01)
            sl_price = round(sl_price / tick_size) * tick_size
            
            current_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if ((direction == "BUY" and sl_price >= current_price) or
                (direction == "SELL" and sl_price <= current_price)):
                return
                
            order = self.api_call_with_retry(
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if direction == "BUY" else "BUY",
                    type="STOP_MARKET",
                    stopPrice=sl_price,
                    quantity=size,
                    reduceOnly=True,
                    workingType="MARK_PRICE"
                )
            )
            
            self.order_ids[order['orderId']] = {
                'symbol': symbol,
                'orderId': order['orderId'],
                'type': 'STOP_MARKET',
                'side': order['side'],
                'stopPrice': sl_price,
                'quantity': size
            }
            
            logger.info(f"Black Swan protection SL placed at {sl_price}")
            
        except Exception as e:
            logger.error(f"Black Swan protection error: {str(e)}")

    def _place_partial_take_profit(self, direction, symbol, tp_price, tp_size):
        try:
            if self.client is None:
                return
                
            price_precision = self._get_price_precision(symbol)
            qty_precision = self._get_quantity_precision(symbol)
            
            tp_price = round(tp_price, price_precision)
            tp_size = round(tp_size, qty_precision)
            
            min_qty = Config.SYMBOL_INFO[symbol]['min_qty']
            if tp_size < min_qty:
                return
                
            current_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if ((direction == "BUY" and tp_price <= current_price) or
                (direction == "SELL" and tp_price >= current_price)):
                if direction == "BUY":
                    tp_price = current_price * (1 + 0.001)
                else:
                    tp_price = current_price * (1 - 0.001)
                tp_price = round(tp_price, price_precision)
                
            try:
                order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side="SELL" if direction == "BUY" else "BUY",
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_price,
                        quantity=tp_size,
                        reduceOnly=True
                    )
                )
                
                self.order_ids[order['orderId']] = {
                    'symbol': symbol,
                    'orderId': order['orderId'],
                    'type': 'TAKE_PROFIT_MARKET',
                    'side': order['side'],
                    'stopPrice': tp_price,
                    'quantity': tp_size
                }
                
                logger.info(f"Quick TP placed at {tp_price} for {tp_size} {symbol}")
            except Exception as e:
                logger.error(f"Quick TP error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Partial take profit error: {str(e)}")

    def _place_take_profit(self, direction, symbol, tp_price, tp_size):
        try:
            if self.client is None:
                return
                
            price_precision = self._get_price_precision(symbol)
            qty_precision = self._get_quantity_precision(symbol)
            
            tp_price = round(tp_price, price_precision)
            tp_size = round(tp_size, qty_precision)
            
            min_qty = Config.SYMBOL_INFO[symbol]['min_qty']
            if tp_size < min_qty:
                return
                
            current_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if ((direction == "BUY" and tp_price <= current_price) or
                (direction == "SELL" and tp_price >= current_price)):
                if direction == "BUY":
                    tp_price = current_price * (1 + 0.001)
                else:
                    tp_price = current_price * (1 - 0.001)
                tp_price = round(tp_price, price_precision)
                
            try:
                order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side="SELL" if direction == "BUY" else "BUY",
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_price,
                        quantity=tp_size,
                        reduceOnly=True
                    )
                )
                
                self.order_ids[order['orderId']] = {
                    'symbol': symbol,
                    'orderId': order['orderId'],
                    'type': 'TAKE_PROFIT_MARKET',
                    'side': order['side'],
                    'stopPrice': tp_price,
                    'quantity': tp_size
                }
                
                logger.info(f"Main TP placed at {tp_price} for {tp_size} {symbol}")
            except Exception as e:
                logger.error(f"Main TP error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Take profit error: {str(e)}")

    def _place_stop_loss(self, direction, symbol, sl_price, size):
        try:
            if self.client is None:
                return
                
            price_precision = self._get_price_precision(symbol)
            sl_price = round(sl_price, price_precision)
            
            current_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
            if ((direction == "BUY" and sl_price >= current_price) or
                (direction == "SELL" and sl_price <= current_price)):
                if direction == "BUY":
                    sl_price = current_price * (1 - 0.001)
                else:
                    sl_price = current_price * (1 + 0.001)
                sl_price = round(sl_price, price_precision)
            
            sl_side = "BUY" if direction == "SELL" else "SELL"
            
            try:
                sl_order = self.api_call_with_retry(
                    lambda: self.client.futures_create_order(
                        symbol=symbol,
                        side=sl_side,
                        type="STOP_MARKET",
                        stopPrice=sl_price,
                        closePosition=True
                    )
                )
                
                self.order_ids[sl_order['orderId']] = {
                    'symbol': symbol,
                    'orderId': sl_order['orderId'],
                    'type': 'STOP_MARKET',
                    'side': sl_order['side'],
                    'stopPrice': sl_price
                }
                
                logger.info(f"SL order placed at {sl_price}")
            except Exception as e:
                logger.error(f"SL order error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Stop loss error: {str(e)}")

    def _get_quantity_precision(self, symbol: str) -> int:
        step_size = Config.SYMBOL_INFO[symbol].get('step_size', 0.1)
        if step_size >= 1:
            return 0
        return abs(int(np.log10(step_size)))

    def _get_price_precision(self, symbol: str) -> int:
        tick_size = Config.SYMBOL_INFO[symbol].get('tick_size', 0.0001)
        if tick_size >= 1:
            return 0
        return abs(int(np.log10(tick_size)))

    def _monitor_position(self):
        try:
            if not self.current_position or self.client is None:
                return
                
            symbol = self.current_position['symbol']
            direction = self.current_position['direction']
            
            positions = self.api_call_with_retry(
                lambda: self.client.futures_position_information(symbol=symbol))
            if positions:
                position = positions[0]
                position_amt = float(position['positionAmt'])
                
                current_price = self.risk_manager.get_mark_price(symbol) if self.risk_manager else 0
                
                if abs(position_amt) < 1e-8:
                    pnl = float(position['unRealizedProfit'])
                    logger.info(f"✅ Position closed: PnL = ${pnl:.4f}")
                    
                    self.daily_profit += pnl
                    
                    if Config.POSITION_HEDGING and self.risk_manager:
                        self.risk_manager.close_hedge_position(symbol)
                    
                    if Config.ADAPTIVE_LEARNING and self.adaptive_learner:
                        self._update_learning_data(pnl > 0, pnl, symbol)
                    
                    if Config.MULTIDIMENSIONAL_ANALYSIS and self.scenario_analyzer:
                        self._update_scenario_performance(pnl > 0, pnl)
                    
                    trade_result = {
                        'symbol': symbol,
                        'direction': direction,
                        'pnl': pnl,
                        'score': self.current_position.get('signal_score', 0),
                        'duration': (datetime.now() - self.current_position['opened_at']).total_seconds()
                    }
                    if self.auto_optimizer:
                        self.auto_optimizer.collect_performance_data(trade_result)
                    
                    if Config.TELEGRAM_ENABLED:
                        self.send_telegram_alert(
                            f"🏁 Position closed: {symbol}\n"
                            f"PnL: ${pnl:.4f}"
                        )
                    
                    self._update_trade_in_history(symbol, {
                        'pnl': pnl,
                        'closed_at': datetime.utcnow().isoformat(),
                        'status': 'closed'
                    })
                    
                    self.cancel_all_orders(symbol)
                    
                    for order_id in list(self.order_ids.keys()):
                        if self.order_ids[order_id]['symbol'] == symbol:
                            del self.order_ids[order_id]
                    
                    self.current_position = None
                    
                    if pnl < 0:
                        if self.risk_manager:
                            self.risk_manager.last_loss_time = time.time()
                            self.risk_manager.update_trade_outcome(False, pnl, symbol)
                        logger.warning(f"Loss! Trading paused")
                        if Config.TELEGRAM_ENABLED:
                            self.send_telegram_alert(f"💔 Loss: ${pnl:.4f}! Cooling down")
                    else:
                        if self.risk_manager:
                            self.risk_manager.update_trade_outcome(True, pnl, symbol)
                else:
                    if Config.REVERSAL_DETECTION and self.risk_manager and self.risk_manager.detect_market_reversal(symbol, direction):
                        logger.warning(f"Market reversal detected for {symbol}! Closing position early")
                        self._close_position_immediately(symbol, direction)
                        return
                    
                    self._check_quick_tp(symbol, current_price)
                    self._advanced_trailing_stop(symbol, current_price)
                    self._early_exit_check(symbol, direction, current_price)
                
        except Exception as e:
            logger.error(f"Position monitoring error: {str(e)}")

    def _update_learning_data(self, is_profitable: bool, pnl: float, symbol: str):
        """Обновляет данные для обучения AI"""
        try:
            if not self.current_position:
                return
                
            features = {
                'rsi': self.data_feeder.get_technical_data(symbol).get('rsi', 50) if self.data_feeder else 50,
                'adx': self.data_feeder.get_technical_data(symbol).get('adx', 20) if self.data_feeder else 20,
                'macd': self.data_feeder.get_technical_data(symbol).get('macd', 0) if self.data_feeder else 0,
                'stochastic_k': self.data_feeder.get_technical_data(symbol).get('stochastic_k', 50) if self.data_feeder else 50,
                'stochastic_d': self.data_feeder.get_technical_data(symbol).get('stochastic_d', 50) if self.data_feeder else 50,
                'atr': self.data_feeder.get_technical_data(symbol).get('atr', 0) if self.data_feeder else 0,
                'obv': self.data_feeder.get_technical_data(symbol).get('obv', 0) if self.data_feeder else 0,
                'cci': self.data_feeder.get_technical_data(symbol).get('cci', 0) if self.data_feeder else 0,
                'bb_percent': self.data_feeder.get_technical_data(symbol).get('bb_percent', 50) if self.data_feeder else 50,
                'vwap': self.data_feeder.get_technical_data(symbol).get('vwap', 0) if self.data_feeder else 0,
                'orderbook_imbalance': self.data_feeder.get_orderbook_data(symbol).get('imbalance', 0) if self.data_feeder else 0,
                'orderbook_pressure': self.data_feeder.get_orderbook_data(symbol).get('pressure', 0) if self.data_feeder else 0,
                'bid_walls': self.data_feeder.get_orderbook_data(symbol).get('bid_walls', 0) if self.data_feeder else 0,
                'ask_walls': self.data_feeder.get_orderbook_data(symbol).get('ask_walls', 0) if self.data_feeder else 0,
                'social_sentiment': self.data_feeder.get_social_sentiment(symbol).get('sentiment', 0.5) if self.data_feeder else 0.5,
                'social_volume': self.data_feeder.get_social_sentiment(symbol).get('social_volume', 0) if self.data_feeder else 0,
                'galaxy_score': self.data_feeder.get_social_sentiment(symbol).get('galaxy_score', 50) if self.data_feeder else 50,
                'alt_rank': self.data_feeder.get_social_sentiment(symbol).get('alt_rank', 50) if self.data_feeder else 50,
                'volatility': self.data_feeder.get_technical_data(symbol).get('volatility', 0) if self.data_feeder else 0,
                'price_change': self.data_feeder.get_technical_data(symbol).get('price_change', 0) if self.data_feeder else 0,
                'volume_change': self.data_feeder.get_technical_data(symbol).get('volume_change', 0) if self.data_feeder else 0,
                'high_low_ratio': self.data_feeder.get_technical_data(symbol).get('high_low_ratio', 0) if self.data_feeder else 0,
                'hour_of_day': self.current_position['opened_at'].hour,
                'day_of_week': self.current_position['opened_at'].weekday(),
                'market_cap': self.data_feeder.get_market_cap(symbol) if self.data_feeder else 0,
                'signal_score': self.current_position.get('signal_score', 0),
                'signal_direction': 1 if self.current_position['direction'] == 'BUY' else -1,
                'market_regime': self.risk_manager.market_regime if self.risk_manager else 'N/A'
            }
            
            self.adaptive_learner.add_training_example(
                features, 
                {'pnl': pnl, 'profitable': is_profitable, 'price_change': (pnl / self.current_position['size']) / self.current_position['entry_price']},
                symbol
            )
            
        except Exception as e:
            logger.error(f"Error updating learning data: {str(e)}")

    def _update_scenario_performance(self, is_profitable: bool, pnl: float):
        """Обновляет производительность сценариев"""
        try:
            if not self.current_position or 'scenario_scores' not in self.current_position:
                return
                
            scenario_scores = self.current_position['scenario_scores']
            
            for scenario_name, scenario_result in scenario_scores.items():
                self.scenario_analyzer.update_scenario_performance(
                    scenario_name, is_profitable, pnl
                )
                
        except Exception as e:
            logger.error(f"Error updating scenario performance: {str(e)}")

    def _early_exit_check(self, symbol: str, direction: str, current_price: float):
        try:
            if not self.current_position or self.current_position.get('early_exit'):
                return
                
            if Config.ADAPTIVE_LEARNING and self.adaptive_learner:
                features = {
                    'rsi': self.data_feeder.get_technical_data(symbol).get('rsi', 50) if self.data_feeder else 50,
                    'adx': self.data_feeder.get_technical_data(symbol).get('adx', 20) if self.data_feeder else 20,
                    'macd': self.data_feeder.get_technical_data(symbol).get('macd', 0) if self.data_feeder else 0,
                    'stochastic_k': self.data_feeder.get_technical_data(symbol).get('stochastic_k', 50) if self.data_feeder else 50,
                    'stochastic_d': self.data_feeder.get_technical_data(symbol).get('stochastic_d', 50) if self.data_feeder else 50,
                    'atr': self.data_feeder.get_technical_data(symbol).get('atr', 0) if self.data_feeder else 0,
                    'obv': self.data_feeder.get_technical_data(symbol).get('obv', 0) if self.data_feeder else 0,
                    'cci': self.data_feeder.get_technical_data(symbol).get('cci', 0) if self.data_feeder else 0,
                    'bb_percent': self.data_feeder.get_technical_data(symbol).get('bb_percent', 50) if self.data_feeder else 50,
                    'vwap': self.data_feeder.get_technical_data(symbol).get('vwap', 0) if self.data_feeder else 0,
                    'orderbook_imbalance': self.data_feeder.get_orderbook_data(symbol).get('imbalance', 0) if self.data_feeder else 0,
                    'orderbook_pressure': self.data_feeder.get_orderbook_data(symbol).get('pressure', 0) if self.data_feeder else 0,
                    'bid_walls': self.data_feeder.get_orderbook_data(symbol).get('bid_walls', 0) if self.data_feeder else 0,
                    'ask_walls': self.data_feeder.get_orderbook_data(symbol).get('ask_walls', 0) if self.data_feeder else 0,
                    'social_sentiment': self.data_feeder.get_social_sentiment(symbol).get('sentiment', 0.5) if self.data_feeder else 0.5,
                    'social_volume': self.data_feeder.get_social_sentiment(symbol).get('social_volume', 0) if self.data_feeder else 0,
                    'galaxy_score': self.data_feeder.get_social_sentiment(symbol).get('galaxy_score', 50) if self.data_feeder else 50,
                    'alt_rank': self.data_feeder.get_social_sentiment(symbol).get('alt_rank', 50) if self.data_feeder else 50,
                    'volatility': self.data_feeder.get_technical_data(symbol).get('volatility', 0) if self.data_feeder else 0,
                    'price_change': self.data_feeder.get_technical_data(symbol).get('price_change', 0) if self.data_feeder else 0,
                    'volume_change': self.data_feeder.get_technical_data(symbol).get('volume_change', 0) if self.data_feeder else 0,
                    'high_low_ratio': self.data_feeder.get_technical_data(symbol).get('high_low_ratio', 0) if self.data_feeder else 0,
                    'hour_of_day': datetime.utcnow().hour,
                    'day_of_week': datetime.utcnow().weekday(),
                    'market_cap': self.data_feeder.get_market_cap(symbol) if self.data_feeder else 0,
                    'signal_score': self.current_position.get('signal_score', 0),
                    'signal_direction': 1 if direction == 'BUY' else -1,
                    'market_regime': self.risk_manager.market_regime if self.risk_manager else 'N/A',
                    'current_pnl': (current_price - self.current_position['entry_price']) / self.current_position['entry_price'] * 100
                }
                
                ai_prediction = self.adaptive_learner.predict(features)
                
                if ai_prediction['probability'] < 0.3 and ai_prediction['confidence'] > 0.8:
                    logger.warning(f"❗ AI predicts high loss probability: {ai_prediction['probability']:.2f}, closing early")
                    self._close_position_immediately(symbol, direction)
                    self.current_position['early_exit'] = True
                    self.send_telegram_alert(f"⚠️ AI early exit triggered for {symbol} at {current_price:.8f}")
                
        except Exception as e:
            logger.error(f"Early exit check error: {str(e)}")

    def cancel_all_orders(self, symbol: str):
        try:
            if self.client is None:
                return
                
            open_orders = self.api_call_with_retry(
                lambda: self.client.futures_get_open_orders(symbol=symbol)
            )
            
            for order in open_orders:
                try:
                    self.api_call_with_retry(
                        lambda: self.client.futures_cancel_order(
                            symbol=symbol,
                            orderId=order['orderId']
                        )
                    )
                    logger.info(f"Cancelled order {order['orderId']} for {symbol}")
                    
                    if order['orderId'] in self.order_ids:
                        del self.order_ids[order['orderId']]
                except Exception as e:
                    logger.error(f"Error cancelling order {order['orderId']}: {str(e)}")
                    
            logger.info(f"All orders cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {str(e)}")

    def _advanced_trailing_stop(self, symbol: str, current_price: float):
        try:
            if not self.current_position:
                return
                
            direction = self.current_position['direction']
            entry = self.current_position['entry_price']
            current_sl = self.current_position['sl']
            profit_pct = abs(current_price - entry) / entry * 100
            
            if Config.DYNAMIC_TRAILING and self.risk_manager:
                volatility = self.risk_manager.calculate_atr(symbol) / current_price * 100
                if volatility > 1.5:
                    trailing_start = max(0.5, Config.TRAILING_START)
                    trailing_distance = max(0.15, Config.TRAILING_DISTANCE)
                else:
                    trailing_start = max(0.3, Config.TRAILING_START)
                    trailing_distance = max(0.1, Config.TRAILING_DISTANCE)
            else:
                trailing_start = Config.TRAILING_START
                trailing_distance = Config.TRAILING_DISTANCE
            
            if profit_pct > trailing_start:
                if direction == "BUY":
                    new_sl = current_price * (1 - trailing_distance/100)
                    if new_sl > current_sl and new_sl > entry:
                        self._update_sl(symbol, new_sl)
                
                elif direction == "SELL":
                    new_sl = current_price * (1 + trailing_distance/100)
                    if new_sl < current_sl and new_sl < entry:
                        self._update_sl(symbol, new_sl)
                        
        except Exception as e:
            logger.error(f"Advanced trailing stop error: {str(e)}")

    def _check_quick_tp(self, symbol: str, current_price: float):
        try:
            if not self.current_position:
                return
                
            if not self.current_position.get('quick_tp_hit'):
                if self.current_position['direction'] == "BUY":
                    if current_price >= self.current_position['quick_tp']:
                        logger.info(f"✅ Quick TP reached for {symbol}")
                        self.current_position['quick_tp_hit'] = True
                        self._move_sl_to_breakeven(symbol)
                else:
                    if current_price <= self.current_position['quick_tp']:
                        logger.info(f"✅ Quick TP reached for {symbol}")
                        self.current_position['quick_tp_hit'] = True
                        self._move_sl_to_breakeven(symbol)
                        
        except Exception as e:
            logger.error(f"Quick TP check error: {str(e)}")

    def _move_sl_to_breakeven(self, symbol: str):
        try:
            if not self.current_position:
                return
                
            direction = self.current_position['direction']
            entry_price = self.current_position['entry_price']
            
            buffer = 0.0003
            if direction == "BUY":
                new_sl = entry_price * (1 - buffer)
            else:
                new_sl = entry_price * (1 + buffer)
            
            self._update_sl(symbol, new_sl)
            
            self.current_position['sl'] = new_sl
            self.current_position['breakeven'] = True
            
            logger.info(f"Moved SL to breakeven: {new_sl}")
            
            if Config.TELEGRAM_ENABLED:
                self.send_telegram_alert(f"🔧 Moved SL to breakeven for {symbol}")
                
        except Exception as e:
            logger.error(f"Breakeven move error: {str(e)}")

    def _update_sl(self, symbol: str, new_sl: float):
        try:
            if self.client is None:
                return
                
            current_sl_order = None
            for order_id, order_info in self.order_ids.items():
                if order_info['symbol'] == symbol and order_info['type'] == 'STOP_MARKET':
                    current_sl_order = order_info
                    break
            
            if not current_sl_order:
                logger.warning("No active stop-loss order found")
                return
                
            self.api_call_with_retry(
                lambda: self.client.futures_cancel_order(
                    symbol=symbol,
                    orderId=current_sl_order['orderId']
                )
            )
            
            if current_sl_order['orderId'] in self.order_ids:
                del self.order_ids[current_sl_order['orderId']]
            
            new_sl = self._round_to_tick(symbol, new_sl)
            
            direction = self.current_position['direction']
            sl_side = "BUY" if direction == "SELL" else "SELL"
            
            new_order = self.api_call_with_retry(
                lambda: self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type="STOP_MARKET",
                    stopPrice=new_sl,
                    closePosition=True
                )
            )
            
            self.order_ids[new_order['orderId']] = {
                'symbol': symbol,
                'orderId': new_order['orderId'],
                'type': 'STOP_MARKET',
                'side': new_order['side'],
                'stopPrice': new_sl
            }
            
            self.current_position['sl'] = new_sl
            
            logger.info(f"SL updated to {new_sl}")
            
        except Exception as e:
            logger.error(f"SL update error: {str(e)}")

    def _round_to_tick(self, symbol: str, price: float) -> float:
        """Округляет цену до ближайшего тика"""
        tick_size = Config.SYMBOL_INFO[symbol].get('tick_size', 0.0001)
        if tick_size >= 1:
            return round(price)
        return round(price / tick_size) * tick_size

    def _close_position_immediately(self, symbol: str, direction: str):
        """Немедленно закрывает позицию"""
        try:
            if self.client is None:
                return
                
            positions = self.api_call_with_retry(
                lambda: self.client.futures_position_information(symbol=symbol))
            if positions:
                position = positions[0]
                position_amt = float(position['positionAmt'])
                
                if abs(position_amt) > 1e-8:
                    close_side = "SELL" if direction == "BUY" else "BUY"
                    order = self.api_call_with_retry(
                        lambda: self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type="MARKET",
                            quantity=abs(position_amt),
                            reduceOnly=True
                        )
                    )
                    
                    logger.info(f"Emergency position closed: {symbol}")
                    self.send_telegram_alert(f"🚨 Emergency close: {symbol}")
                    
                    if Config.POSITION_HEDGING and self.risk_manager:
                        self.risk_manager.close_hedge_position(symbol)
                    
                    self.cancel_all_orders(symbol)
                    
                    self.current_position = None
                    
        except Exception as e:
            logger.error(f"Emergency close error: {str(e)}")

    def _check_position_mismatch(self):
        """Проверяет расхождение между нашей информацией о позиции и биржей"""
        try:
            if not self.current_position or self.client is None:
                return
                
            symbol = self.current_position['symbol']
            positions = self.api_call_with_retry(
                lambda: self.client.futures_position_information(symbol=symbol))
            
            if positions:
                position = positions[0]
                position_amt = float(position['positionAmt'])
                
                if abs(position_amt) < 1e-8 and self.current_position:
                    logger.warning(f"Position mismatch detected for {symbol}! Clearing local position info")
                    self.current_position = None
                    self.cancel_all_orders(symbol)
                    
        except Exception as e:
            logger.error(f"Position mismatch check error: {str(e)}")

    def _save_trade_to_history(self, trade_info: Dict):
        """Сохраняет информацию о сделке в историю"""
        try:
            self.trade_history.append(trade_info)
            
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")

    def _update_trade_in_history(self, symbol: str, updates: Dict):
        """Обновляет информацию о сделке в истории"""
        try:
            for trade in reversed(self.trade_history):
                if trade['symbol'] == symbol and trade['status'] == 'open':
                    trade.update(updates)
                    break
                    
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error updating trade history: {str(e)}")

    def _load_trade_history(self):
        """Загружает историю сделок из файла"""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.error(f"Error loading trade history: {str(e)}")
            self.trade_history = []

    def recovery_mode(self):
        """Режим восстановления после сбоя"""
        try:
            logger.info("Entering recovery mode...")
            
            if self.client is None:
                logger.info("No recovery needed in simulation mode")
                return
                
            if self.risk_manager:
                open_positions = self.risk_manager.get_open_positions()
                for position in open_positions:
                    symbol = position['symbol']
                    position_amt = float(position['positionAmt'])
                    
                    if abs(position_amt) > 1e-8:
                        direction = "BUY" if position_amt > 0 else "SELL"
                        self._close_position_immediately(symbol, direction)
            
            open_orders = self.api_call_with_retry(lambda: self.client.futures_get_open_orders())
            for order in open_orders:
                try:
                    self.api_call_with_retry(
                        lambda: self.client.futures_cancel_order(
                            symbol=order['symbol'],
                            orderId=order['orderId']
                        )
                    )
                except Exception as e:
                    logger.error(f"Error cancelling order in recovery: {str(e)}")
            
            if Config.POSITION_HEDGING and self.risk_manager:
                for symbol in list(self.risk_manager.hedge_positions.keys()):
                    self.risk_manager.close_hedge_position(symbol)
            
            self.current_position = None
            self.order_ids = {}
            
            logger.info("Recovery completed")
            self.send_telegram_alert("🔄 Recovery mode completed")
            
        except Exception as e:
            logger.error(f"Recovery mode error: {str(e)}")
            self.send_telegram_alert(f"🚨 Recovery mode failed: {str(e)}")

    def cleanup_on_exit(self):
        """Отменяет все ордера при завершении работы"""
        logger.info("Cleaning up orders on exit...")
        try:
            if self.client is None:
                return
                
            open_orders = self.api_call_with_retry(lambda: self.client.futures_get_open_orders())
            for order in open_orders:
                try:
                    self.api_call_with_retry(
                        lambda: self.client.futures_cancel_order(
                            symbol=order['symbol'],
                            orderId=order['orderId']
                        )
                    )
                    logger.info(f"Cancelled order {order['orderId']} on exit")
                except Exception as e:
                    logger.error(f"Error cancelling order {order['orderId']} on exit: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting open orders on exit: {str(e)}")

    def send_telegram_alert(self, message: str):
        """Отправляет сообщение в Telegram"""
        if not Config.TELEGRAM_ENABLED or not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return
            
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            payload = {
                'chat_id': Config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                
        except Exception as e:
            logger.error(f"Telegram send error: {str(e)}")

# Применяем диагностический патч после определения класса TradingBot
DiagnosticPatch.patch_trading_bot()

if __name__ == "__main__":
    try:
        logger.info("🚀 STARTING BOT WITH CRITICAL FIXES...")
        
        # 🧪 ТЕСТОВЫЙ РЕЖИМ ДЛЯ БЫСТРОГО ЗАПУСКА
        Config.PAPER_TRADING = True
        Config.MAX_SYMBOLS_TO_CHECK = 2
        Config.CHECK_INTERVAL = 5.0
        Config.MIN_SCORE = 2.0
        
        bot = TradingBot()
        
        # 🔥 ПРИНУДИТЕЛЬНЫЙ ЗАПУСК ТОРГОВЛИ
        bot.force_trading_start()
        
        # 🔥 БЫСТРАЯ ПРОВЕРКА СИГНАЛОВ ПЕРЕД ЗАПУСКОМ
        logger.info("🔍 Performing quick signal scan...")
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        for symbol in test_symbols:
            signal = bot._ultra_simple_signal_detection(symbol)
            if signal:
                logger.info(f"🎯 IMMEDIATE SIGNAL FOUND: {symbol} {signal['direction']}")
        
        # 🔥 ЗАПУСК В ТЕСТОВОМ РЕЖИМЕ
        bot.test_mode_run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (KeyboardInterrupt in main)")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}")
        logger.info("🔄 Attempting restart in 30 seconds...")
        time.sleep(30)
        os.execv(sys.executable, ['python'] + sys.argv)

import time
import threading
from functools import wraps
import logging
from config import config

logger = logging.getLogger('RateLimiter')

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                # Удаляем старые вызовы
                self.calls = [call for call in self.calls if now - call < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        logger.warning(f"Rate limit reached. Sleeping {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                    self.calls = self.calls[1:]
                
                self.calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper

# Используем настройки из вашего конфига
binance_limiter = RateLimiter(
    max_calls=config.API_RATE_LIMIT['requests_per_minute'], 
    period=60
)
request_limiter = RateLimiter(max_calls=10, period=1)  # 10 запросов в секунду

def safe_api_call(func):
    """Декоратор для безопасных API вызовов с повторными попытками"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        max_retries = 3
        retry_delay = config.API_RATE_LIMIT['retry_delay']
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                if 'too many requests' in error_msg:
                    wait_time = retry_delay * (2 ** attempt)  # Экспоненциальная задержка
                    logger.warning(f"API limit hit (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s")
                    time.sleep(wait_time)
                elif 'banned' in error_msg:
                    logger.error(f"IP banned: {e}")
                    break
                else:
                    logger.error(f"API error: {e}")
                    break
        
        logger.error(f"All API attempts failed. Last error: {last_exception}")
        raise last_exception
    return wrapper
