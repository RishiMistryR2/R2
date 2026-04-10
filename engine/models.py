"""
engine/models.py — canonical data models for R2.

Every enum and dataclass used anywhere in the system is defined here.
No model definitions live elsewhere. Other modules import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TrendStrength(Enum):
    STRONG   = "strong"
    MODERATE = "moderate"
    WEAK     = "weak"
    NONE     = "none"


class SetupType(Enum):
    CONTINUATION     = "continuation"
    REVERSAL         = "reversal"
    FAILED_BREAKOUT  = "failed_breakout"
    BREAKDOWN        = "breakdown"
    BREAKOUT         = "breakout"
    PULLBACK         = "pullback"
    BOUNCE_REJECTION = "bounce_rejection"
    EVENT_PLAY       = "event_play"
    NONE             = "none"


class MarketNarrative(Enum):
    """Instantaneous market condition — changes per candle."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    BREAKOUT      = "breakout"
    BREAKDOWN     = "breakdown"
    REVERSAL_BULL = "reversal_bull"
    REVERSAL_BEAR = "reversal_bear"
    TRANSITION    = "transition"
    CHOP          = "chop"


class MarketRegime(Enum):
    """
    Persistent market environment. Requires multiple candles to change (hysteresis).
    Controls score multipliers, entry permissions, and preferred setup types.
    """
    TREND_BULL         = "trend_bull"
    TREND_BEAR         = "trend_bear"
    RANGE              = "range"
    CHOP               = "chop"
    BREAKOUT_ATTEMPT   = "breakout_attempt"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    TRANSITION         = "transition"


class TradeDirection(Enum):
    CALLS = "calls"
    PUTS  = "puts"


class InstrumentType(Enum):
    OPTIONS = "options"
    SHARES  = "shares"


class TradeStatus(Enum):
    OPEN    = "open"
    PARTIAL = "partial"
    CLOSED  = "closed"
    STOPPED = "stopped"


class ExitReason(Enum):
    PROFIT_TARGET      = "profit_target"
    STOP_LOSS          = "stop_loss"
    TREND_FLIP         = "trend_flip"
    STRUCTURE_LOSS     = "structure_loss"
    MOMENTUM_EXHAUSTED = "momentum_exhausted"
    TRAILING_STOP      = "trailing_stop"
    TIME_STOP          = "time_stop"
    PREMIUM_STOP       = "premium_stop"
    MANUAL             = "manual"


class StopCause(Enum):
    THESIS_BREAK          = "thesis_break"
    PREMIUM_DETERIORATION = "premium_deterioration"
    TIME_DECAY            = "time_decay"
    MANUAL_EXIT           = "manual_exit"
    TARGET_REACHED        = "target_reached"
    PARTIAL_EXIT          = "partial_exit"


class SessionPhase(Enum):
    PREMARKET   = "premarket"
    OPEN        = "open"
    MIDDAY      = "midday"
    POWER_HOUR  = "power_hour"
    AFTER_HOURS = "after_hours"
    OVERNIGHT   = "overnight"


class Readiness(Enum):
    NOT_READY       = "not_ready"
    PARTIALLY_READY = "partially_ready"
    FULLY_READY     = "fully_ready"


class EventMode(Enum):
    OFF          = "off"
    CONSERVATIVE = "conservative"
    AGGRESSIVE   = "aggressive"


class EventStatus(Enum):
    PENDING      = "pending"
    CONTINUATION = "continuation"
    ABSORPTION   = "absorption"
    INCONCLUSIVE = "inconclusive"
    EXPIRED      = "expired"


class EventPriority(Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


class PsychCondition(Enum):
    NEUTRAL    = "neutral"
    PANIC      = "panic"
    EUPHORIA   = "euphoria"
    INDECISION = "indecision"
    TRAP       = "trap"


class BreadthTrend(Enum):
    RISING  = "rising"
    FALLING = "falling"
    FLAT    = "flat"


class SessionShape(Enum):
    DISCIPLINED_GAIN = "disciplined_gain"
    DISCIPLINED_LOSS = "disciplined_loss"
    RECOVERED        = "recovered"
    GAVE_BACK        = "gave_back"
    CHURNED          = "churned"
    SINGLE_MOVE      = "single_move"
    UNKNOWN          = "unknown"


class ContractProfile(Enum):
    ATM           = "atm"
    NEAR_OTM      = "near_otm"
    OTM           = "otm"
    NO_PREFERENCE = "no_preference"


# ── Regime lookup tables ─────────────────────────────────────────────────────

REGIME_SCORE_MULTIPLIERS: Dict[MarketRegime, float] = {
    MarketRegime.TREND_BULL:         1.00,
    MarketRegime.TREND_BEAR:         1.00,
    MarketRegime.RANGE:              0.80,
    MarketRegime.CHOP:               0.60,
    MarketRegime.BREAKOUT_ATTEMPT:   0.85,
    MarketRegime.BREAKOUT_CONFIRMED: 1.10,
    MarketRegime.TRANSITION:         0.70,
}

REGIME_ENTRY_PERMITTED: Dict[MarketRegime, bool] = {
    MarketRegime.TREND_BULL:         True,
    MarketRegime.TREND_BEAR:         True,
    MarketRegime.RANGE:              True,
    MarketRegime.CHOP:               False,
    MarketRegime.BREAKOUT_ATTEMPT:   False,
    MarketRegime.BREAKOUT_CONFIRMED: True,
    MarketRegime.TRANSITION:         False,
}

REGIME_PREFERRED_SETUPS: Dict[MarketRegime, List[str]] = {
    MarketRegime.TREND_BULL:         ["continuation", "pullback"],
    MarketRegime.TREND_BEAR:         ["continuation", "bounce_rejection"],
    MarketRegime.RANGE:              ["reversal", "failed_breakout"],
    MarketRegime.CHOP:               [],
    MarketRegime.BREAKOUT_ATTEMPT:   [],
    MarketRegime.BREAKOUT_CONFIRMED: ["breakout", "continuation"],
    MarketRegime.TRANSITION:         [],
}

SESSION_SCORE_MODIFIERS: Dict[SessionPhase, float] = {
    SessionPhase.PREMARKET:   1.00,
    SessionPhase.OPEN:        0.80,
    SessionPhase.MIDDAY:      1.00,
    SessionPhase.POWER_HOUR:  1.05,
    SessionPhase.AFTER_HOURS: 1.00,
    SessionPhase.OVERNIGHT:   1.00,
}


# ═══════════════════════════════════════════════════════════════════════════
# Candle
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    timestamp:    datetime
    open:         float
    high:         float
    low:          float
    close:        float
    volume:       float
    timeframe:    str
    is_closed:    bool = False
    session_type: str  = "regular"

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def range(self) -> float:
        return self.high - self.low


# ═══════════════════════════════════════════════════════════════════════════
# Indicators
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorSnapshot:
    timeframe:  str
    timestamp:  datetime
    close:      float = 0.0
    ema_9:      Optional[float] = None
    ema_20:     Optional[float] = None
    ema_50:     Optional[float] = None
    ema_200:    Optional[float] = None
    vwap:       Optional[float] = None
    rsi:        Optional[float] = None
    macd_line:  Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist:  Optional[float] = None
    atr:        Optional[float] = None
    price_vs_vwap: Optional[str] = None
    ema_stack:     Optional[str] = None
    rsi_zone:      Optional[str] = None
    macd_bias:     Optional[str] = None
    session_mix:   bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Structure / trend
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SupportResistanceLevel:
    price:      float
    level_type: str
    touches:    int  = 1
    strength:   str  = "weak"
    formed_at:  Optional[datetime] = None


@dataclass
class TrendState:
    timeframe:       str
    direction:       Direction     = Direction.NEUTRAL
    strength:        TrendStrength = TrendStrength.NONE
    is_weakening:    bool          = False
    structure_break: bool          = False
    last_swing_high: Optional[float] = None
    last_swing_low:  Optional[float] = None
    higher_highs:    bool = False
    lower_lows:      bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Setup scoring
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SetupScore:
    """
    Composite 0-100 quality score with four visible components.
    Grade uses adjusted_total (after all multipliers).
    """
    total: float = 0.0
    trend_alignment:    float = 0.0
    momentum_quality:   float = 0.0
    structure_context:  float = 0.0
    entry_timing:       float = 0.0
    alignment_detail:   str   = ""
    momentum_detail:    str   = ""
    structure_detail:   str   = ""
    timing_detail:      str   = ""
    regime_multiplier:  float = 1.0
    session_multiplier: float = 1.0
    adjusted_total:     float = 0.0
    readiness_penalty:  float = 0.0
    sizing_flags:       List[str] = field(default_factory=list)

    def grade(self) -> str:
        effective = self.adjusted_total if self.adjusted_total > 0 else self.total
        if effective >= 80: return "A+"
        if effective >= 60: return "B"
        if effective >= 40: return "C"
        return "skip"

    def to_dict(self) -> dict:
        return {
            "total":           round(self.total, 1),
            "adjusted_total":  round(self.adjusted_total, 1),
            "grade":           self.grade(),
            "regime_mult":     self.regime_multiplier,
            "session_mult":    self.session_multiplier,
            "readiness_penalty": self.readiness_penalty,
            "components": {
                "trend_alignment":   round(self.trend_alignment, 1),
                "momentum_quality":  round(self.momentum_quality, 1),
                "structure_context": round(self.structure_context, 1),
                "entry_timing":      round(self.entry_timing, 1),
            },
            "detail": {
                "alignment":  self.alignment_detail,
                "momentum":   self.momentum_detail,
                "structure":  self.structure_detail,
                "timing":     self.timing_detail,
            },
            "sizing_flags": self.sizing_flags,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Regime state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    regime:            MarketRegime
    confidence:        float = 0.0
    score_multiplier:  float = 1.0
    entry_permitted:   bool  = True
    description:       str   = ""
    preferred_setups:  List[str] = field(default_factory=list)
    caution_note:      str   = ""
    candles_in_regime: int   = 0
    raw_scores:        Dict[str, int] = field(default_factory=dict)
    prev_regime:       Optional[MarketRegime] = None

    def to_dict(self) -> dict:
        return {
            "regime":           self.regime.value,
            "confidence":       round(self.confidence, 2),
            "score_multiplier": self.score_multiplier,
            "entry_permitted":  self.entry_permitted,
            "description":      self.description,
            "preferred_setups": self.preferred_setups,
            "caution_note":     self.caution_note,
            "raw_scores":       self.raw_scores,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Session state + intent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    phase:               SessionPhase
    market_open:         bool      = False
    entries_allowed:     bool      = False
    minutes_since_open:  Optional[int] = None
    minutes_to_close:    Optional[int] = None
    score_modifier:      float     = 1.0
    entry_filter:        str       = "normal"
    phase_note:          str       = ""
    readiness:           Readiness = Readiness.NOT_READY
    event_mode_active:   bool      = False

    def to_dict(self) -> dict:
        return {
            "phase":              self.phase.value,
            "market_open":        self.market_open,
            "entries_allowed":    self.entries_allowed,
            "minutes_since_open": self.minutes_since_open,
            "minutes_to_close":   self.minutes_to_close,
            "score_modifier":     self.score_modifier,
            "phase_note":         self.phase_note,
            "readiness":          self.readiness.value,
            "event_mode_active":  self.event_mode_active,
        }


@dataclass
class SessionIntent:
    directional_bias:          Optional[str] = None
    min_score:                 float         = 60.0
    risk_tolerance:            str           = "normal"
    event_mode:                EventMode     = EventMode.OFF
    avoid_chop:                bool          = True
    require_htf_alignment:     bool          = False
    max_trades:                int           = 5
    max_loss_pct:              Optional[float] = None
    target_gain_pct:           Optional[float] = None
    allow_principal_momentum:  bool          = False
    notes:                     str           = ""

    def score_passes(self, score: float) -> bool:
        effective_min = self.min_score
        if self.risk_tolerance == "reduced":
            effective_min = max(self.min_score + 10, 70)
        return score >= effective_min

    def size_modifier(self) -> float:
        return {"reduced": 0.5, "normal": 1.0, "aggressive": 1.25}.get(
            self.risk_tolerance, 1.0
        )

    def to_dict(self) -> dict:
        return {
            "directional_bias":        self.directional_bias,
            "min_score":               self.min_score,
            "risk_tolerance":          self.risk_tolerance,
            "event_mode":              self.event_mode.value,
            "avoid_chop":              self.avoid_chop,
            "require_htf_alignment":   self.require_htf_alignment,
            "max_trades":              self.max_trades,
            "max_loss_pct":            self.max_loss_pct,
            "target_gain_pct":         self.target_gain_pct,
            "allow_principal_momentum": self.allow_principal_momentum,
            "notes":                   self.notes[:200],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Event tracking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EventState:
    """Lifecycle of a principal alert from detection through resolution."""
    principal_alert_active:        bool             = False
    principal_implied_direction:   Optional[str]    = None
    alert_timestamp:               Optional[datetime] = None
    event_confirmed:               bool             = False
    event_direction:               Optional[str]    = None
    event_confirmation_timestamp:  Optional[datetime] = None
    event_status:                  EventStatus      = EventStatus.PENDING
    event_strength:                Optional[str]    = None
    htf_agrees_with_event:         Optional[bool]   = None
    event_mode_active:             bool             = False
    event_candles_since_confirmation: int           = 0
    event_mode_expiry_timestamp:   Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "principal_alert_active":      self.principal_alert_active,
            "principal_implied_direction": self.principal_implied_direction,
            "event_confirmed":             self.event_confirmed,
            "event_direction":             self.event_direction,
            "event_status":                self.event_status.value,
            "event_strength":              self.event_strength,
            "htf_agrees_with_event":       self.htf_agrees_with_event,
            "event_mode_active":           self.event_mode_active,
        }


@dataclass
class PrioritizedEvent:
    priority:          EventPriority
    score:             int
    event_type:        str
    headline:          str
    detail:            str
    should_trigger_ai: bool
    should_alert:      bool
    original_events:   List[str] = field(default_factory=list)
    timestamp:         Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "priority":   self.priority.value,
            "score":      self.score,
            "event_type": self.event_type,
            "headline":   self.headline,
            "detail":     self.detail,
            "trigger_ai": self.should_trigger_ai,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Trade state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimeStopConfig:
    expected_move_window_minutes: int   = 10
    stall_check_interval_minutes: int   = 5
    premium_decay_threshold_pct:  float = 15.0


@dataclass
class SizingResult:
    instrument_type:          InstrumentType
    risk_amount:              float
    suggested_position_size:  float
    position_notional:        float
    position_pct_of_account:  float
    stop_distance_pct:        float
    flags:                    List[str] = field(default_factory=list)


@dataclass
class TradeState:
    """Complete live context of the currently active trade."""
    trade_id:       str
    direction:      TradeDirection
    entry_time:     datetime
    size:           float
    instrument_type: InstrumentType = InstrumentType.OPTIONS

    # Underlying levels
    entry_price:                   float = 0.0
    underlying_entry_price:        float = 0.0
    underlying_invalidation_price: float = 0.0
    underlying_stop_price:         float = 0.0
    stop_loss:                     float = 0.0
    profit_target:                 float = 0.0
    trailing_stop:                 Optional[float] = None

    # Options-specific
    option_entry_price:                   Optional[float] = None
    estimated_option_stop_price:          Optional[float] = None
    premium_emergency_stop_pct:           Optional[float] = None
    estimated_option_loss_at_thesis_stop: Optional[float] = None
    time_stop_config:                     Optional[TimeStopConfig] = None
    time_stop_first_triggered_at:         Optional[datetime] = None
    time_stop_trigger_count:              int = 0

    # Live P&L
    status:        TradeStatus = TradeStatus.OPEN
    current_price: float = 0.0
    high_water:    float = 0.0
    low_water:     float = 0.0
    pnl_pct:       float = 0.0
    pnl_dollar:    float = 0.0

    # Sizing
    suggested_position_size: float = 0.0
    suggested_risk_amount:   float = 0.0

    # Stop trigger tracking
    thesis_stop_triggered:  bool = False
    premium_stop_triggered: bool = False
    time_stop_triggered:    bool = False

    # Context
    setup_type:   SetupType       = SetupType.NONE
    thesis:       str             = ""
    invalidation: str             = ""
    exit_reason:  Optional[ExitReason] = None
    partial_exited: bool  = False
    partial_size:   float = 0.0

    def update_pnl(self, current_price: float) -> None:
        self.current_price = current_price
        entry = self.underlying_entry_price or self.entry_price
        if not entry:
            return
        if self.direction == TradeDirection.CALLS:
            self.pnl_pct    = (current_price - entry) / entry * 100
            self.high_water = max(self.high_water or current_price, current_price)
        else:
            self.pnl_pct   = (entry - current_price) / entry * 100
            self.low_water = min(self.low_water or current_price, current_price)
        self.pnl_dollar = self.pnl_pct / 100 * self.size

    def to_dict(self) -> dict:
        return {
            "trade_id":        self.trade_id,
            "direction":       self.direction.value,
            "instrument_type": self.instrument_type.value,
            "underlying_entry": self.underlying_entry_price,
            "underlying_stop":  self.underlying_stop_price,
            "profit_target":    self.profit_target,
            "trailing_stop":    self.trailing_stop,
            "option_entry":     self.option_entry_price,
            "option_stop_est":  self.estimated_option_stop_price,
            "current_price":    self.current_price,
            "pnl_pct":         round(self.pnl_pct, 2),
            "pnl_dollar":      round(self.pnl_dollar, 2),
            "status":          self.status.value,
            "setup_type":      self.setup_type.value,
            "thesis":          self.thesis,
            "invalidation":    self.invalidation,
            "partial_exited":  self.partial_exited,
            "stop_flags": {
                "thesis":  self.thesis_stop_triggered,
                "premium": self.premium_stop_triggered,
                "time":    self.time_stop_triggered,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Psychology / internals
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PsychologicalMarketState:
    """confidence_modifier is hard-capped at [-5.0, +5.0]."""
    panic_detected:      bool = False
    euphoria_detected:   bool = False
    indecision_detected: bool = False
    trap_detected:       bool = False
    trap_level:          Optional[float] = None
    dominant_condition:  PsychCondition  = PsychCondition.NEUTRAL
    confidence_modifier: float = 0.0
    description:         str   = ""

    def set_modifier(self, raw: float) -> None:
        self.confidence_modifier = max(-5.0, min(5.0, raw))

    def to_dict(self) -> dict:
        return {
            "dominant":            self.dominant_condition.value,
            "panic":               self.panic_detected,
            "euphoria":            self.euphoria_detected,
            "indecision":          self.indecision_detected,
            "trap":                self.trap_detected,
            "trap_level":          self.trap_level,
            "confidence_modifier": self.confidence_modifier,
            "description":         self.description,
        }


@dataclass
class MarketInternalsState:
    breadth_score:          float        = 0.0
    leader_confirmation:    bool         = False
    sector_divergence:      Optional[str] = None
    breadth_trend:          BreadthTrend  = BreadthTrend.FLAT
    breadth_trend_strength: str          = "weak"
    internals_modifier:     float        = 0.0
    _breadth_history:       List[float]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "breadth_score":          round(self.breadth_score, 2),
            "leader_confirmation":    self.leader_confirmation,
            "sector_divergence":      self.sector_divergence,
            "breadth_trend":          self.breadth_trend.value,
            "breadth_trend_strength": self.breadth_trend_strength,
            "internals_modifier":     self.internals_modifier,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Behavioral enforcement
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehavioralFlag:
    flag_type:                str
    triggered_at:             datetime
    context:                  str
    severity:                 str  = "info"
    session_count:            int  = 1
    historical_session_count: int  = 0
    escalation_level:         int  = 0
    requires_acknowledgment:  bool = False
    acknowledged_by_user:     bool = False
    referenced_rule:          str  = ""

    def to_dict(self) -> dict:
        return {
            "flag_type":     self.flag_type,
            "severity":      self.severity,
            "context":       self.context,
            "session_count": self.session_count,
            "requires_ack":  self.requires_acknowledgment,
            "acknowledged":  self.acknowledged_by_user,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Narrative state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NarrativeState:
    narrative:     MarketNarrative = MarketNarrative.CHOP
    confidence:    float           = 0.0
    description:   str             = ""
    key_levels:    str             = ""
    bias_source:   str             = ""
    trade_posture: str             = "neutral"

    def to_dict(self) -> dict:
        return {
            "narrative":     self.narrative.value,
            "confidence":    round(self.confidence, 2),
            "description":   self.description,
            "key_levels":    self.key_levels,
            "bias_source":   self.bias_source,
            "trade_posture": self.trade_posture,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Master MarketState
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketState:
    """Complete system snapshot — everything passed to strategy, AI, and CLI."""
    symbol:    str
    timestamp: datetime

    indicators:        Dict[str, IndicatorSnapshot]    = field(default_factory=dict)
    trends:            Dict[str, TrendState]            = field(default_factory=dict)
    support_levels:    List[SupportResistanceLevel]     = field(default_factory=list)
    resistance_levels: List[SupportResistanceLevel]     = field(default_factory=list)

    setup_type:      SetupType               = SetupType.NONE
    setup_direction: Optional[TradeDirection] = None
    setup_score:     Optional[SetupScore]    = None
    narrative:       Optional[NarrativeState] = None
    htf_bias:        Direction               = Direction.NEUTRAL

    regime:      Optional[RegimeState]   = None
    session:     Optional[SessionState]  = None
    intent:      Optional[SessionIntent] = None
    event_state: Optional[EventState]    = None
    psychology:  Optional[PsychologicalMarketState] = None
    internals:   Optional[MarketInternalsState]     = None
    active_trade: Optional[TradeState]  = None
    behavioral_flags: List[BehavioralFlag] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol":    self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "htf_bias":  self.htf_bias.value,
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "setup": {
                "type":      self.setup_type.value,
                "direction": self.setup_direction.value if self.setup_direction else None,
                "score":     self.setup_score.to_dict() if self.setup_score else None,
            },
            "indicators": {
                tf: {
                    "close":         snap.close,
                    "ema_9":         snap.ema_9,
                    "ema_20":        snap.ema_20,
                    "ema_50":        snap.ema_50,
                    "ema_200":       snap.ema_200,
                    "vwap":          snap.vwap,
                    "rsi":           round(snap.rsi, 1) if snap.rsi else None,
                    "macd_hist":     round(snap.macd_hist, 4) if snap.macd_hist else None,
                    "atr":           round(snap.atr, 3) if snap.atr else None,
                    "ema_stack":     snap.ema_stack,
                    "rsi_zone":      snap.rsi_zone,
                    "macd_bias":     snap.macd_bias,
                    "price_vs_vwap": snap.price_vs_vwap,
                }
                for tf, snap in self.indicators.items()
            },
            "trends": {
                tf: {
                    "direction":       t.direction.value,
                    "strength":        t.strength.value,
                    "is_weakening":    t.is_weakening,
                    "structure_break": t.structure_break,
                    "higher_highs":    t.higher_highs,
                    "lower_lows":      t.lower_lows,
                }
                for tf, t in self.trends.items()
            },
            "support":    [{"price": s.price, "strength": s.strength}
                           for s in self.support_levels[:5]],
            "resistance": [{"price": r.price, "strength": r.strength}
                           for r in self.resistance_levels[:5]],
            "regime":       self.regime.to_dict()       if self.regime       else None,
            "session":      self.session.to_dict()      if self.session      else None,
            "intent":       self.intent.to_dict()       if self.intent       else None,
            "event_state":  self.event_state.to_dict()  if self.event_state  else None,
            "psychology":   self.psychology.to_dict()   if self.psychology   else None,
            "internals":    self.internals.to_dict()    if self.internals    else None,
            "active_trade": self.active_trade.to_dict() if self.active_trade else None,
            "behavioral_flags": [f.to_dict() for f in self.behavioral_flags
                                  if f.severity in ("warning", "critical")],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Persistent records
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Completed trade record for persistent storage."""
    id:              str
    symbol:          str
    direction:       TradeDirection
    instrument_type: InstrumentType
    entry_price:     float
    entry_time:      datetime
    size:            float

    underlying_stop_price:         float = 0.0
    underlying_invalidation_price: float = 0.0
    profit_target:                 float = 0.0
    option_entry_price:            Optional[float] = None
    estimated_option_stop_price:   Optional[float] = None

    status:      TradeStatus        = TradeStatus.OPEN
    exit_price:  Optional[float]    = None
    exit_time:   Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    stop_cause:  Optional[StopCause]  = None
    pnl_pct:     Optional[float]    = None
    pnl_dollar:  Optional[float]    = None

    actual_underlying_price_at_exit: Optional[float] = None
    actual_option_price_at_exit:     Optional[float] = None
    actual_premium_loss_pct:         Optional[float] = None

    mae_underlying_pct: float = 0.0
    mfe_underlying_pct: float = 0.0
    mae_option_pct:     Optional[float] = None
    mfe_option_pct:     Optional[float] = None

    thesis_stop_triggered:  bool  = False
    premium_stop_triggered: bool  = False
    time_stop_triggered:    bool  = False

    suggested_position_size: float = 0.0
    suggested_risk_amount:   float = 0.0
    actual_position_size:    float = 0.0
    sizing_compliant:        bool  = False

    setup_type:  SetupType = SetupType.NONE
    setup_score: float     = 0.0
    notes:       str       = ""


@dataclass
class MarketEvent:
    """Raw event fired by the engine, before priority scoring."""
    event_type: str
    timestamp:  datetime
    payload:    dict = field(default_factory=dict)
    priority:   int  = 1
