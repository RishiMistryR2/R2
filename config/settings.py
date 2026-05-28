"""
config/settings.py — single source of truth for all R2 constants.

Rules:
  - All magic numbers live here, nowhere else.
  - Secrets come from environment variables only.
  - Feature flags live here and gate behavior in other modules.
  - Do not import from engine/ or strategy/ — settings is imported by everyone.
"""

import os
from pathlib import Path

# ── Identity ─────────────────────────────────────────────────────────────────
ASSISTANT_NAME        = "R2"
ASSISTANT_DESCRIPTION = "Real-time SPY trading analyst and decision partner"
ASSISTANT_VERSION     = "0.1.0"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MEMORY_DIR   = DATA_DIR / "memory"
JOURNAL_DIR  = DATA_DIR / "journal"
LOGS_DIR     = ROOT_DIR / "logs"

for _d in (SESSIONS_DIR, MEMORY_DIR, JOURNAL_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE     = os.getenv("LOG_TO_FILE", "0") == "1"
LOG_FILE        = LOGS_DIR / "r2.log"
LOG_FORMAT      = "[R2] %(asctime)s %(levelname)-8s %(name)s — %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# ── Credentials ───────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")
ALPACA_DATA_URL   = "https://data.alpaca.markets"
ALPACA_WS_URL     = "wss://stream.data.alpaca.markets/v2/iex"
ALPACA_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
DATA_FEED_TIMEOUT = 10  # seconds for sync REST calls to Alpaca

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_KEY       = os.getenv("FINNHUB_API_KEY",       "")
TRADIER_KEY       = os.getenv("TRADIER_API_KEY",        "")

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_NEWS_FEED = os.getenv("ENABLE_NEWS_FEED",  "0") == "1"
ENABLE_INTERNALS = os.getenv("ENABLE_INTERNALS",  "0") == "1"
ENABLE_VOICE     = os.getenv("ENABLE_VOICE",       "0") == "1"

# ── Market / instrument ───────────────────────────────────────────────────────
SYMBOL     = "SPY"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]

CANDLE_HISTORY: dict = {
    "1m":  500,
    "5m":  300,
    "15m": 200,
    "1h":  150,
    "1d":  100,
}

TIMEFRAME_SECONDS: dict = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "1d":  86400,
}

# Alpaca timeframe strings for REST seed requests
ALPACA_TF_MAP: dict = {
    "1m":  "1Min",
    "5m":  "5Min",
    "15m": "15Min",
    "1h":  "1Hour",
    "1d":  "1Day",
}

# Candle thresholds per readiness tier (both 1h AND 15m must meet their threshold)
READINESS_THRESHOLDS: dict = {
    "not_ready":       {"1d": 0,   "1h": 0,   "15m": 0,   "5m": 0,   "1m": 0},
    "partially_ready": {"1d": 30,  "1h": 30,  "15m": 30,  "5m": 30,  "1m": 30},
    "fully_ready":     {"1d": 100, "1h": 150, "15m": 200, "5m": 300, "1m": 200},
}

# How many bars to fetch per timeframe at startup (REST seeding)
SEED_BAR_COUNTS: dict = {
    "1d":  100,
    "1h":  150,
    "15m": 200,
    "5m":  300,
    "1m":  500,
}

# Seed order: largest timeframe first so regime is available ASAP
SEED_ORDER = ["1d", "1h", "15m", "5m", "1m"]

# ── Indicators ────────────────────────────────────────────────────────────────
EMA_PERIODS                = [9, 20, 50, 200]
RSI_PERIOD                 = 14
MACD_FAST                  = 12
MACD_SLOW                  = 26
MACD_SIGNAL                = 9
ATR_PERIOD                 = 14
MIN_CANDLES_FOR_INDICATORS = 30

# ── Structure detection ───────────────────────────────────────────────────────
SWING_LOOKBACK   = 10      # candles each side for swing high/low
SR_ZONE_ATR_MULT = 0.5     # ATR multiplier for S/R zone width
SR_MIN_TOUCHES   = 2       # minimum touches to call a level significant

# ── RSI zones ─────────────────────────────────────────────────────────────────
RSI_OVERBOUGHT   = 70
RSI_OVERSOLD     = 30
RSI_NEUTRAL_HIGH = 55
RSI_NEUTRAL_LOW  = 45

# ── Regime detection ──────────────────────────────────────────────────────────
REGIME_PERSISTENCE        = 3    # consecutive matching candidates to confirm switch
BREAKOUT_MAX_WAIT_CANDLES = 6    # 15m candles before BREAKOUT_ATTEMPT reverts to RANGE

# ── Scoring ───────────────────────────────────────────────────────────────────
MIN_SCORE_DEFAULT           = 60.0   # setups below this not presented
PARTIALLY_READY_PENALTY     = 10.0   # subtracted from adjusted_total when partially ready
POOR_RR_SCORE_DEDUCTION     = 8.0    # deducted from entry_timing component
MIN_RR_RATIO_DEFAULT        = 1.5
MIN_RR_RATIO_EVENT_AGGRESSIVE = 1.2

# ── Position sizing ───────────────────────────────────────────────────────────
ACCOUNT_SIZE_DEFAULT       = float(os.getenv("ACCOUNT_SIZE", "25000"))
RISK_PER_TRADE_PCT_DEFAULT = 1.0    # % of account risked per trade
MAX_POSITION_SIZE_PCT      = 20.0   # cap: max position as % of account
MIN_STOP_DISTANCE_PCT      = 0.15   # stops tighter than this are flagged

# ── Options stop framework ────────────────────────────────────────────────────
MAX_PREMIUM_LOSS_PCT_DEFAULT    = 35.0
PREMIUM_EMERGENCY_BUFFER        = 1.5     # emergency = estimated_loss × 1.5
ATM_DELTA_APPROX                = 0.575
NEAR_OTM_DELTA_APPROX           = 0.425
OTM_DELTA_APPROX                = 0.275
ATR_BUFFER_OPTIONS              = 0.20    # ATR fraction for options stop buffer
ATR_BUFFER_SHARES               = 0.50    # ATR fraction for shares stop buffer

DEFAULT_TIME_STOP_WINDOWS: dict = {
    "event_play":    5,
    "breakout":      8,
    "continuation":  12,
    "reversal":      15,
    "default":       10,
}

TIME_STOP_PHASE_MULTIPLIERS: dict = {
    "open":       0.60,
    "midday":     1.00,
    "power_hour": 0.70,
}

TIME_STOP_CHECK_INTERVAL_MINUTES         = 5
TIME_STOP_PREMIUM_DECAY_THRESHOLD_PCT    = 15.0
TIME_STOP_PROGRESS_THRESHOLD             = 0.30   # 30% toward target = developing

# ── Trade management ──────────────────────────────────────────────────────────
DEFAULT_PROFIT_TARGET_PCT    = 0.015
MAX_CONCURRENT_TRADES        = 3
RUNNER_PARTIAL_PCT           = 0.50
TRAILING_STOP_ACTIVATION_PCT = 0.008

# ── Session rules ─────────────────────────────────────────────────────────────
MAX_TRADES_PER_DAY_DEFAULT  = 5
MAX_DAILY_LOSS_PCT_DEFAULT  = 2.0
COOLDOWN_AFTER_LOSS_MINUTES = 0

# ── News / context ────────────────────────────────────────────────────────────
NEWS_REFRESH_SECONDS            = 300
NEWS_DEDUP_WINDOW_MINUTES       = 60
NEWS_DEDUP_ROLLING_SIZE         = 500
NEWS_MIN_SCORE_FOR_AI           = 2.0
NEWS_MAX_HEADLINES_AI           = 5
NEWS_MAX_TRADE_HEADLINES        = 3
NEWS_STALENESS_WARN_MINUTES     = 30

MACRO_EVENT_RISK_WINDOW_HOURS   = 2.0
MACRO_SURPRISE_HIGH_THRESHOLD   = 0.5

PRINCIPAL_ALERT_WINDOW_MINUTES  = 90
PRINCIPAL_MIN_SCORE             = 7.0
PRINCIPAL_IMPORTANCE_BOOST      = 2.0

# ── Event confirmation ────────────────────────────────────────────────────────
EVENT_DISPLACEMENT_ATR_MULT     = 1.5
EVENT_VOLUME_SPIKE_MULT         = 2.0
EVENT_RANGE_EXPANSION_MULT      = 2.0
EVENT_MICRO_STRUCTURE_BARS      = 5
EVENT_CONFIRMATION_MIN_SIGNALS  = 2
EVENT_CONFIRMATION_CANDLE_LIMIT = 3
EVENT_5M_PROGRESS_THRESHOLD     = 0.30
EVENT_5M_RETRACE_THRESHOLD      = 0.50
EVENT_MODE_EXPIRY_MINUTES       = 30

# ── AI / Claude ───────────────────────────────────────────────────────────────
CLAUDE_MODEL           = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS      = 1500
CLAUDE_UPDATE_TOKENS   = 400
AI_HISTORY_MAX_TURNS   = 8
AI_COMMENTARY_COOLDOWN = 120

# ── Memory / reflection ───────────────────────────────────────────────────────
MEMORY_SCHEMA_VERSION           = "1.0"
PATTERN_MIN_OCCURRENCES         = 10
PATTERN_CONFIDENCE_THRESHOLD    = 0.70
PATTERN_SATURATION_OCCURRENCES  = 30
PATTERN_RECENCY_DECAY_DAYS      = 30
SESSION_DRAWDOWN_WARN_MULTIPLIER = 1.5

# ── Behavioral enforcement ────────────────────────────────────────────────────
BEHAVIORAL_WARNING_THRESHOLD        = 1
BEHAVIORAL_CRITICAL_THRESHOLD       = 3
BEHAVIORAL_CRITICAL_SCORE_PENALTY   = 8.0
BEHAVIORAL_WARNING_SCORE_PENALTY    = 3.0

# ── Market internals ──────────────────────────────────────────────────────────
INTERNALS_SYMBOLS     = ["XLK", "XLF", "XLV", "XLY", "XLC", "XLI"]
SPY_TOP_COMPONENTS    = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                          "TSLA", "BRK.B", "UNH", "JPM"]
BREADTH_TREND_LOOKBACK   = 5
BREADTH_TREND_FLAT_RANGE = 0.08
BREADTH_TREND_STRONG_MIN = 0.25

# ── Psychology / market character ─────────────────────────────────────────────
PSYCHOLOGY_MAX_MODIFIER      = 5.0
PANIC_DISPLACEMENT_ATR_MULT  = 3.0
PANIC_VOLUME_SPIKE_MULT      = 3.0
EUPHORIA_EMA20_EXTENSION_ATR = 2.5
EUPHORIA_RSI_THRESHOLD       = 72.0
INDECISION_BODY_ATR_FRACTION = 0.30
INDECISION_VOLUME_FRACTION   = 0.70
TRAP_RETEST_CANDLE_MAX       = 3

# ── Premarket ─────────────────────────────────────────────────────────────────
PREMARKET_CONFIDENCE_PENALTY = 0.15
PREMARKET_WINDOW_FRACTION    = 0.20
