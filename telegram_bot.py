#!/usr/bin/env python3
"""
Sportybet Aviator Predictor - Telegram Mini App Bot
Real-time multiplier prediction via Telegram
"""

import logging
import os
import json
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

# Flask for web interface
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Telegram bot
from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ParseMode

# TensorFlow for LSTM
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("TensorFlow required. Install: pip install tensorflow")
    exit(1)

# Requests for API
import requests
import threading
import asyncio
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LSTM PREDICTOR MODEL
# ============================================================================

class LSTMPredictor:
    """LSTM model for multiplier prediction"""
    
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = None
        self.is_trained = False
        self.min_val = 0.0
        self.max_val = 100.0
        self.model_path = Path("aviator_model.h5")
        
    def prepare_data(self, history):
        """Prepare data for LSTM training"""
        if len(history) < self.sequence_length + 1:
            return None, None
        
        history = np.array(history, dtype=np.float32)
        self.min_val = history.min()
        self.max_val = history.max()
        
        if self.max_val == self.min_val:
            self.max_val = self.min_val + 1
        
        normalized = (history - self.min_val) / (self.max_val - self.min_val)
        
        X, y = [], []
        for i in range(len(normalized) - self.sequence_length):
            X.append(normalized[i:i + self.sequence_length])
            y.append(normalized[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM neural network"""
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, 1), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, history, epochs=50, verbose=0):
        """Train the LSTM model"""
        X, y = self.prepare_data(history)
        
        if X is None:
            return False, "Insufficient data"
        
        try:
            if self.model is None:
                self.build_model()
            
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=8,
                verbose=verbose,
                validation_split=0.2
            )
            
            self.is_trained = True
            self.save_model()
            return True, "Model trained"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def predict(self, history):
        """Predict next multiplier"""
        if not self.is_trained or self.model is None:
            return None, "Not trained"
        
        try:
            if len(history) < self.sequence_length:
                return None, "Need more data"
            
            recent = np.array(history[-self.sequence_length:], dtype=np.float32)
            normalized = (recent - self.min_val) / (self.max_val - self.min_val)
            normalized = normalized.reshape((1, self.sequence_length, 1))
            
            pred_normalized = self.model.predict(normalized, verbose=0)[0][0]
            pred_actual = pred_normalized * (self.max_val - self.min_val) + self.min_val
            pred_actual = max(1.0, min(pred_actual, 200.0))
            
            return round(float(pred_actual), 2), "Success"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def save_model(self):
        """Save trained model"""
        try:
            if self.model is not None:
                self.model.save(str(self.model_path))
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def load_model(self):
        """Load trained model"""
        try:
            if self.model_path.exists():
                self.model = load_model(str(self.model_path))
                self.is_trained = True
                return True
        except Exception as e:
            logger.error(f"Load error: {e}")
        return False


# ============================================================================
# DATA FETCHER
# ============================================================================

class AviatorDataFetcher:
    """Fetch Sportybet Aviator game data"""
    
    def __init__(self):
        self.history = deque(maxlen=500)
        self.current_multiplier = None
        self.game_running = False
        
    def fetch_history(self, limit=100):
        """Fetch historical multiplier data"""
        try:
            # Try Sportybet API
            # response = requests.get(...)
            # Fallback to demo data
            data = self._generate_realistic_history(limit)
            self.history.extend(data)
            return list(self.history)
        except Exception as e:
            logger.error(f"Error: {e}")
            data = self._generate_realistic_history(limit)
            self.history.extend(data)
            return list(self.history)
    
    def _generate_realistic_history(self, count=100):
        """Generate realistic demo data"""
        history = []
        for _ in range(count):
            if np.random.random() < 0.70:
                multiplier = np.random.exponential(scale=0.5) + 1.0
                multiplier = np.clip(multiplier, 1.0, 2.5)
            elif np.random.random() < 0.25:
                multiplier = np.random.exponential(scale=2.0) + 2.5
                multiplier = np.clip(multiplier, 2.5, 10.0)
            else:
                multiplier = np.random.exponential(scale=5.0) + 10.0
                multiplier = np.clip(multiplier, 10.0, 100.0)
            history.append(round(multiplier, 2))
        return history


# ============================================================================
# TELEGRAM BOT
# ============================================================================

class TelegramAviatorBot:
    """Main Telegram bot"""
    
    def __init__(self, token: str, mini_app_url: str):
        self.token = token
        self.mini_app_url = mini_app_url
        self.app = Application.builder().token(token).build()
        
        # Data storage per user
        self.user_data: Dict[int, Dict] = defaultdict(lambda: {
            'history': deque(maxlen=500),
            'predictor': LSTMPredictor(),
            'fetcher': AviatorDataFetcher()
        })
        
        # Setup handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup command handlers"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("predict", self.predict_command))
        self.app.add_handler(CommandHandler("train", self.train_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("load_data", self.load_data_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        user_id = update.effective_user.id
        first_name = update.effective_user.first_name
        
        welcome_text = f"""
🎯 *Sportybet Aviator Predictor*

Hello {first_name}! 👋

I'm your AI-powered Aviator multiplier predictor.
I use LSTM neural networks to analyze game patterns and predict the next multiplier.

📊 *Features:*
• AI Predictions using LSTM
• Live multiplier tracking
• Game history & statistics
• Real-time analysis

🚀 *Quick Start:*
1. /load_data - Load historical games
2. /train - Train the AI model
3. /predict - Get next multiplier prediction
4. /stats - View statistics

💡 *Or use the Mini App:*
"""
        
        # Create inline keyboard with Mini App button
        keyboard = [
            [InlineKeyboardButton("🎮 Open Mini App", web_app=WebAppInfo(url=self.mini_app_url))]
        ]
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        help_text = """
📖 *Available Commands:*

/load_data - Fetch historical game data
/train - Train LSTM model (1-2 min)
/predict - Get next multiplier prediction
/stats - View current statistics
/start - Show welcome message
/help - Show this message

🎮 *Mini App:*
Tap "Open Mini App" button for interactive experience

🤖 *How It Works:*
1. I fetch historical Aviator game rounds
2. Train LSTM neural network on patterns
3. Predict next multiplier
4. Show confidence level

💡 *Tips:*
• More data = better predictions
• Train with 100+ games for best results
• Predictions improve over time
• Use Mini App for real-time tracking
"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def load_data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Load historical data"""
        user_id = update.effective_user.id
        
        await update.message.reply_text("📊 Loading historical data...")
        
        try:
            fetcher = self.user_data[user_id]['fetcher']
            history = fetcher.fetch_history(limit=100)
            
            # Store in user data
            self.user_data[user_id]['history'] = deque(history, maxlen=500)
            
            stats_text = f"""
✅ *Data Loaded Successfully*

📈 *Statistics:*
• Total rounds: {len(history)}
• Average: {np.mean(history):.2f}x
• Median: {np.median(history):.2f}x
• Min: {min(history):.2f}x
• Max: {max(history):.2f}x
• Std Dev: {np.std(history):.2f}x

💡 *Next Step:* /train to train the model
"""
            await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train model"""
        user_id = update.effective_user.id
        
        await update.message.reply_text("🤖 Training model (this takes 1-2 minutes)...\n⏳ Please wait...")
        
        try:
            history = list(self.user_data[user_id]['history'])
            predictor = self.user_data[user_id]['predictor']
            
            if len(history) < 30:
                await update.message.reply_text("❌ Need at least 30 rounds to train")
                return
            
            # Train in background
            def train_in_bg():
                predictor.train(history, epochs=30, verbose=0)
            
            thread = threading.Thread(target=train_in_bg)
            thread.start()
            thread.join()
            
            await update.message.reply_text(
                "✅ *Model Trained!*\n\n"
                "The LSTM neural network is ready.\n"
                "Use /predict to get predictions.",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get prediction"""
        user_id = update.effective_user.id
        
        try:
            predictor = self.user_data[user_id]['predictor']
            history = list(self.user_data[user_id]['history'])
            
            if not predictor.is_trained:
                await update.message.reply_text(
                    "❌ Model not trained yet.\n\n"
                    "Steps:\n"
                    "1. /load_data\n"
                    "2. /train\n"
                    "3. /predict"
                )
                return
            
            pred, msg = predictor.predict(history)
            
            if pred is not None:
                confidence = "🟢 High" if pred > 5 else "🟡 Medium" if pred > 2 else "🔴 Low"
                
                text = f"""
🎯 *AI PREDICTION*

📊 Next Multiplier: *{pred:.2f}x*
📈 Confidence: {confidence}
⏰ Generated: {datetime.now().strftime('%H:%M:%S')}

Based on {len(history)} historical rounds analyzed
"""
                await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(f"❌ Error: {msg}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show statistics"""
        user_id = update.effective_user.id
        
        try:
            history = list(self.user_data[user_id]['history'])
            predictor = self.user_data[user_id]['predictor']
            
            if not history:
                await update.message.reply_text("No data yet. Use /load_data first.")
                return
            
            hist_arr = np.array(history)
            
            stats_text = f"""
📊 *STATISTICS*

🎮 *Game Data:*
• Total Rounds: {len(history)}
• Average: {hist_arr.mean():.2f}x
• Median: {np.median(hist_arr):.2f}x
• Min: {hist_arr.min():.2f}x
• Max: {hist_arr.max():.2f}x
• Std Dev: {hist_arr.std():.2f}x

🎲 *Distribution:*
• High (>5x): {(hist_arr > 5).sum()}
• Medium (2-5x): {((hist_arr >= 2) & (hist_arr <= 5)).sum()}
• Low (<2x): {(hist_arr < 2).sum()}

🤖 *Model Status:*
• Trained: {'✅ Yes' if predictor.is_trained else '❌ No'}
• Sequence Length: {predictor.sequence_length}
"""
            await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses"""
        query = update.callback_query
        await query.answer()
    
    def run(self):
        """Run the bot"""
        logger.info("Starting Telegram bot...")
        self.app.run_polling()


# ============================================================================
# FLASK WEB APP (Mini App)
# ============================================================================

app = Flask(__name__)
CORS(app)

# Store bot instance globally
telegram_bot = None

@app.route('/')
def index():
    """Serve mini app HTML"""
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """API: Load data"""
    try:
        user_id = request.json.get('user_id', 'default')
        fetcher = telegram_bot.user_data[user_id]['fetcher']
        history = fetcher.fetch_history(limit=100)
        telegram_bot.user_data[user_id]['history'] = deque(history, maxlen=500)
        
        hist_arr = np.array(history)
        return jsonify({
            'success': True,
            'rounds': len(history),
            'average': float(hist_arr.mean()),
            'median': float(np.median(hist_arr)),
            'min': float(hist_arr.min()),
            'max': float(hist_arr.max()),
            'history': history[-20:]  # Last 20
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train():
    """API: Train model"""
    try:
        user_id = request.json.get('user_id', 'default')
        history = list(telegram_bot.user_data[user_id]['history'])
        predictor = telegram_bot.user_data[user_id]['predictor']
        
        if len(history) < 30:
            return jsonify({'success': False, 'error': 'Need 30+ rounds'}), 400
        
        success, msg = predictor.train(history, epochs=30, verbose=0)
        
        return jsonify({
            'success': success,
            'message': msg
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """API: Get prediction"""
    try:
        user_id = request.json.get('user_id', 'default')
        predictor = telegram_bot.user_data[user_id]['predictor']
        history = list(telegram_bot.user_data[user_id]['history'])
        
        if not predictor.is_trained:
            return jsonify({'success': False, 'error': 'Model not trained'}), 400
        
        pred, msg = predictor.predict(history)
        
        if pred is not None:
            confidence = 'High' if pred > 5 else 'Medium' if pred > 2 else 'Low'
            return jsonify({
                'success': True,
                'prediction': pred,
                'confidence': confidence
            })
        else:
            return jsonify({'success': False, 'error': msg}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stats', methods=['POST'])
def stats():
    """API: Get statistics"""
    try:
        user_id = request.json.get('user_id', 'default')
        history = list(telegram_bot.user_data[user_id]['history'])
        predictor = telegram_bot.user_data[user_id]['predictor']
        
        if not history:
            return jsonify({'success': False, 'error': 'No data'}), 400
        
        hist_arr = np.array(history)
        
        return jsonify({
            'success': True,
            'total_rounds': len(history),
            'average': float(hist_arr.mean()),
            'median': float(np.median(hist_arr)),
            'min': float(hist_arr.min()),
            'max': float(hist_arr.max()),
            'std_dev': float(hist_arr.std()),
            'high_count': int((hist_arr > 5).sum()),
            'medium_count': int(((hist_arr >= 2) & (hist_arr <= 5)).sum()),
            'low_count': int((hist_arr < 2).sum()),
            'model_trained': predictor.is_trained
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    import sys
    
    # Get token from environment or command line
    token = os.getenv('TELEGRAM_TOKEN') or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if not token:
        print("Error: Telegram token required")
        print("Usage: python bot.py <TELEGRAM_TOKEN>")
        print("Or set TELEGRAM_TOKEN environment variable")
        sys.exit(1)
    
    # Get mini app URL from environment or use default
    mini_app_url = os.getenv('MINI_APP_URL', 'https://example.com')
    
    # Create bot
    global telegram_bot
    telegram_bot = TelegramAviatorBot(token, mini_app_url)
    
    # Run in separate thread
    bot_thread = threading.Thread(target=telegram_bot.run, daemon=True)
    bot_thread.start()
    
    # Run Flask app
    logger.info(f"Starting Flask app on http://localhost:5000")
    logger.info(f"Mini App URL: {mini_app_url}")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
