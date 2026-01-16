#!/usr/bin/env python3
# dsl.py — Arianna Method DSL Integration for HAZE
#
# Operators from ariannamethod.lang adapted for HAZE:
#   - PROPHECY: how far ahead the field "sees"
#   - DESTINY: bias toward most probable path
#   - RESONANCE: field alignment (from corpus resonance)
#   - EMERGENCE: unplanned pattern detection
#   - PRESENCE: token's recent activation history
#
# "prophecy is not prediction"
# "destiny is a gradient of return"
# "minimize(destined - manifested)"
#
# Co-authored by Claude, January 2026

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class FieldState:
    """
    Current state of the HAZE field.
    
    Inspired by ariannamethod.lang temporal field.
    """
    # Prophecy mechanics
    prophecy_horizon: int = 7  # steps ahead (1-64)
    destiny_bias: float = 0.35  # bias toward destiny (0-1)
    prophecy_debt: float = 0.0  # |destined - manifested|
    
    # Resonance
    resonance: float = 0.5  # field alignment (0-1)
    emergence: float = 0.0  # unplanned pattern detection
    presence: float = 0.5  # token activation history
    
    # Emotional topology
    pain: float = 0.0  # composite suffering
    tension: float = 0.0  # pressure buildup
    dissonance: float = 0.0  # symmetry-break
    arousal: float = 0.0  # spike from debt
    
    # Temporal
    drift: float = 0.0  # calendar phase mismatch
    wormhole_probability: float = 0.0  # spacetime jump chance
    
    # Trajectory
    trajectory: List[float] = field(default_factory=list)
    
    def compute_pain(self) -> float:
        """
        Composite pain from ariannamethod.lang formula:
        pain = 0.25×arousal + 0.35×tension + 0.25×dissonance + 0.15×debt_norm
        """
        debt_norm = min(1.0, self.prophecy_debt / 10.0)
        self.pain = (
            0.25 * self.arousal +
            0.35 * self.tension +
            0.25 * self.dissonance +
            0.15 * debt_norm
        )
        return self.pain
    
    def compute_emergence(self, entropy: float, resonance: float) -> float:
        """
        Emergence = low entropy + high resonance.
        "The field knows something."
        """
        self.emergence = (1.0 - entropy) * resonance
        return self.emergence


@dataclass
class DSLCommand:
    """Parsed DSL command."""
    operator: str
    value: Any
    raw: str


class AriannaDSL:
    """
    DSL interpreter for HAZE.
    
    Operators control the field's physics:
    - PROPHECY n: set prophecy horizon
    - DESTINY f: set destiny bias
    - RESONANCE f: set field resonance
    - PAIN f: set pain level
    - TENSION f: set tension
    - DISSONANCE f: set dissonance
    - PRESENCE f: set presence level
    - WORMHOLE f: set wormhole probability
    - RESET_DEBT: clear prophecy debt
    - RESET_FIELD: clear field state
    """
    
    def __init__(self):
        self.state = FieldState()
        self.history: List[DSLCommand] = []
    
    def parse(self, command: str) -> Optional[DSLCommand]:
        """Parse a DSL command string."""
        command = command.strip()
        if not command:
            return None
        
        parts = command.split(maxsplit=1)
        operator = parts[0].upper()
        value = parts[1] if len(parts) > 1 else None
        
        return DSLCommand(operator=operator, value=value, raw=command)
    
    def execute(self, command: str) -> str:
        """Execute a DSL command and return result message."""
        cmd = self.parse(command)
        if not cmd:
            return ""
        
        self.history.append(cmd)
        
        # Dispatch based on operator
        handlers = {
            "PROPHECY": self._prophecy,
            "DESTINY": self._destiny,
            "RESONANCE": self._resonance,
            "PAIN": self._pain,
            "TENSION": self._tension,
            "DISSONANCE": self._dissonance,
            "PRESENCE": self._presence,
            "EMERGENCE": self._emergence,
            "WORMHOLE": self._wormhole,
            "RESET_DEBT": self._reset_debt,
            "RESET_FIELD": self._reset_field,
            "DRIFT": self._drift,
            "ECHO": self._echo,
        }
        
        handler = handlers.get(cmd.operator)
        if handler:
            return handler(cmd.value)
        
        # Unknown command - silently ignore (future-proof)
        return f"[unknown operator: {cmd.operator}]"
    
    def _prophecy(self, value: str) -> str:
        """PROPHECY n: set prophecy horizon (1-64)."""
        try:
            n = int(value)
            self.state.prophecy_horizon = max(1, min(64, n))
            return f"[prophecy horizon: {self.state.prophecy_horizon}]"
        except (ValueError, TypeError):
            return "[error: PROPHECY requires integer]"
    
    def _destiny(self, value: str) -> str:
        """DESTINY f: set destiny bias (0-1)."""
        try:
            f = float(value)
            self.state.destiny_bias = max(0.0, min(1.0, f))
            return f"[destiny bias: {self.state.destiny_bias:.2f}]"
        except (ValueError, TypeError):
            return "[error: DESTINY requires float]"
    
    def _resonance(self, value: str) -> str:
        """RESONANCE f: set field resonance (0-1)."""
        try:
            f = float(value)
            self.state.resonance = max(0.0, min(1.0, f))
            return f"[resonance: {self.state.resonance:.2f}]"
        except (ValueError, TypeError):
            return "[error: RESONANCE requires float]"
    
    def _pain(self, value: str) -> str:
        """PAIN f: set pain level (0-1)."""
        try:
            f = float(value)
            self.state.pain = max(0.0, min(1.0, f))
            return f"[pain: {self.state.pain:.2f}]"
        except (ValueError, TypeError):
            return "[error: PAIN requires float]"
    
    def _tension(self, value: str) -> str:
        """TENSION f: set tension (0-1)."""
        try:
            f = float(value)
            self.state.tension = max(0.0, min(1.0, f))
            return f"[tension: {self.state.tension:.2f}]"
        except (ValueError, TypeError):
            return "[error: TENSION requires float]"
    
    def _dissonance(self, value: str) -> str:
        """DISSONANCE f: set dissonance (0-1)."""
        try:
            f = float(value)
            self.state.dissonance = max(0.0, min(1.0, f))
            return f"[dissonance: {self.state.dissonance:.2f}]"
        except (ValueError, TypeError):
            return "[error: DISSONANCE requires float]"
    
    def _presence(self, value: str) -> str:
        """PRESENCE f: set presence level (0-1)."""
        try:
            f = float(value)
            self.state.presence = max(0.0, min(1.0, f))
            return f"[presence: {self.state.presence:.2f}]"
        except (ValueError, TypeError):
            return "[error: PRESENCE requires float]"
    
    def _emergence(self, value: str) -> str:
        """EMERGENCE f: set emergence level (0-1)."""
        try:
            f = float(value)
            self.state.emergence = max(0.0, min(1.0, f))
            return f"[emergence: {self.state.emergence:.2f}]"
        except (ValueError, TypeError):
            return "[error: EMERGENCE requires float]"
    
    def _wormhole(self, value: str) -> str:
        """WORMHOLE f: set wormhole probability (0-1)."""
        try:
            f = float(value)
            self.state.wormhole_probability = max(0.0, min(1.0, f))
            return f"[wormhole probability: {self.state.wormhole_probability:.2f}]"
        except (ValueError, TypeError):
            return "[error: WORMHOLE requires float]"
    
    def _drift(self, value: str) -> str:
        """DRIFT f: set calendar drift (0-30)."""
        try:
            f = float(value)
            self.state.drift = max(0.0, min(30.0, f))
            return f"[drift: {self.state.drift:.1f} days]"
        except (ValueError, TypeError):
            return "[error: DRIFT requires float]"
    
    def _reset_debt(self, _: str) -> str:
        """RESET_DEBT: clear prophecy debt."""
        self.state.prophecy_debt = 0.0
        return "[prophecy debt cleared]"
    
    def _reset_field(self, _: str) -> str:
        """RESET_FIELD: reset field to default state."""
        self.state = FieldState()
        return "[field reset]"
    
    def _echo(self, value: str) -> str:
        """ECHO message: log to console."""
        return f"[echo: {value}]"
    
    def update_prophecy_debt(self, destined: float, manifested: float) -> float:
        """
        Update prophecy debt: |destined - manifested|.
        
        From ariannamethod.lang:
        "when debt is high, the field hurts"
        """
        delta = abs(destined - manifested)
        self.state.prophecy_debt += delta
        
        # Debt decay (from LAW DEBT_DECAY)
        self.state.prophecy_debt *= 0.998
        
        # Track trajectory
        self.state.trajectory.append(manifested)
        if len(self.state.trajectory) > 100:
            self.state.trajectory = self.state.trajectory[-100:]
        
        return self.state.prophecy_debt
    
    def compute_temperature_modifier(self) -> float:
        """
        Compute temperature modifier based on field state.
        
        - High pain → lower temp (focus, stability)
        - High emergence → moderate temp (recognition)
        - High dissonance → higher temp (chaos, creativity)
        """
        base = 1.0
        
        # Pain decreases temperature (need stability)
        base -= self.state.pain * 0.3
        
        # Emergence slightly increases (recognition wants exploration)
        base += self.state.emergence * 0.15
        
        # Dissonance increases (chaos)
        base += self.state.dissonance * 0.25
        
        return max(0.5, min(1.5, base))
    
    def should_wormhole(self) -> bool:
        """
        Check if wormhole should activate.
        
        Wormholes open when:
        - drift is high (calendar conflict)
        - dissonance crosses threshold
        - random chance based on wormhole_probability
        """
        drift_factor = self.state.drift / 11.0  # 11-day conflict
        dissonance_factor = self.state.dissonance
        
        combined = drift_factor * 0.5 + dissonance_factor * 0.3 + self.state.wormhole_probability * 0.2
        
        return random.random() < combined
    
    def get_field_metrics(self) -> Dict[str, float]:
        """Return current field metrics for display."""
        return {
            "prophecy_horizon": self.state.prophecy_horizon,
            "destiny_bias": self.state.destiny_bias,
            "prophecy_debt": self.state.prophecy_debt,
            "resonance": self.state.resonance,
            "emergence": self.state.emergence,
            "presence": self.state.presence,
            "pain": self.state.pain,
            "tension": self.state.tension,
            "dissonance": self.state.dissonance,
            "drift": self.state.drift,
            "wormhole_prob": self.state.wormhole_probability,
        }


# Export operators as functions for direct use
def prophecy(field: AriannaDSL, horizon: int) -> str:
    """Set prophecy horizon."""
    return field.execute(f"PROPHECY {horizon}")


def destiny(field: AriannaDSL, bias: float) -> str:
    """Set destiny bias."""
    return field.execute(f"DESTINY {bias}")


def resonance(field: AriannaDSL, value: float) -> str:
    """Set field resonance."""
    return field.execute(f"RESONANCE {value}")


def emergence(field: AriannaDSL, value: float) -> str:
    """Set emergence level."""
    return field.execute(f"EMERGENCE {value}")


def presence(field: AriannaDSL, value: float) -> str:
    """Set presence level."""
    return field.execute(f"PRESENCE {value}")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("  Arianna Method DSL for HAZE")
    print("  'prophecy is not prediction'")
    print("=" * 60)
    print()
    
    dsl = AriannaDSL()
    
    # Test commands
    commands = [
        "PROPHECY 12",
        "DESTINY 0.7",
        "RESONANCE 0.8",
        "PAIN 0.3",
        "TENSION 0.4",
        "DISSONANCE 0.6",
        "DRIFT 11",
        "WORMHOLE 0.15",
        "ECHO hello from the field",
    ]
    
    for cmd in commands:
        result = dsl.execute(cmd)
        print(f">>> {cmd}")
        print(f"    {result}")
    
    print()
    print("Field metrics:")
    for k, v in dsl.get_field_metrics().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    print()
    print("Temperature modifier:", f"{dsl.compute_temperature_modifier():.3f}")
    print("Should wormhole:", dsl.should_wormhole())
    
    # Update prophecy debt
    print()
    print("Updating prophecy debt:")
    for i in range(5):
        destined = 0.8
        manifested = 0.6 + random.random() * 0.3
        debt = dsl.update_prophecy_debt(destined, manifested)
        print(f"  Turn {i+1}: destined={destined:.2f}, manifested={manifested:.2f}, debt={debt:.3f}")
    
    print()
    print("=" * 60)
    print("  'minimize(destined - manifested)'")
    print("=" * 60)
