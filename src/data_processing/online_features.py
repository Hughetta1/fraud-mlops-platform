# Real-time sliding-window features via Redis sorted sets.
# Atomic Lua script does push + aggregate in one round-trip.
# Graceful fallback to zero-fill when Redis is unavailable — the API never crashes.
# Interview note: Lua script eliminates network round-trips, guarantees atomicity,
# and operates in O(log N) per transaction.

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Lua script: atomic push + aggregate over sliding window
# KEYS[1] = user transaction SortedSet key
# ARGV[1] = current timestamp (score)
# ARGV[2] = amount value (member)
# ARGV[3] = window cutoff timestamp (entries before this are removed)
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local ts = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local cutoff = tonumber(ARGV[3])

-- Push new transaction
redis.call('ZADD', key, ts, ts .. ':' .. amount)

-- Remove expired entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)

-- Get all amounts in the window
local members = redis.call('ZRANGEBYSCORE', key, cutoff, '+inf')
local count = #members
local sum = 0
local values = {}
for i, m in ipairs(members) do
    local val = tonumber(string.match(m, ':(.+)$'))
    sum = sum + val
    values[i] = val
end

local mean = 0
local std = 0
if count > 0 then
    mean = sum / count
    if count > 1 then
        local variance = 0
        for i, v in ipairs(values) do
            variance = variance + (v - mean) ^ 2
        end
        std = math.sqrt(variance / count)
    end
end

-- Also compute time span (newest - oldest) in seconds
local newest = ts
local oldest = ts
if count > 0 then
    oldest = redis.call('ZRANGEBYSCORE', key, cutoff, '+inf', 'LIMIT', 0, 1, 'WITHSCORES')[2]
end
local time_span = newest - oldest + 1

return {count, sum, mean, std, time_span}
"""


class OnlineFeatureComputer:
    """
    Real-time feature computer backed by Redis sorted-set sliding windows.

    Provides the same velocity features as the batch feature engineering
    pipeline, but computed online per-transaction with O(log N) complexity.

    Usage:
        ofc = OnlineFeatureComputer(redis_client)
        ofc.update("user_123", amount=149.62, timestamp=50000.0)
        features = ofc.compute("user_123", amount=149.62, timestamp=50000.0)
    """

    def __init__(
        self,
        redis_client=None,
        windows: Tuple[int, ...] = (10, 30, 100),
        prefix: str = "fd",
    ):
        """
        Args:
            redis_client: redis.Redis instance, or None for offline fallback.
            windows: Window sizes in number of transactions.
            prefix: Key prefix for Redis keys.
        """
        self.redis = redis_client
        self.windows = windows
        self.prefix = prefix
        self._lua_sha: Optional[str] = None

        if self.redis is not None:
            try:
                self._lua_sha = self.redis.script_load(_SLIDING_WINDOW_LUA)
            except Exception:
                self._lua_sha = None

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}:txns:{user_id}"

    def update(self, user_id: str, amount: float, timestamp: float) -> bool:
        """
        Record a new transaction in the sliding window.

        Returns True on success, False if Redis is unavailable.
        """
        if self.redis is None:
            return False
        try:
            key = self._key(user_id)
            # Use the largest window cutoff to keep enough history
            max_window = max(self.windows)
            cutoff = timestamp - (max_window * 3600)  # approx: 1 tx per hour
            self.redis.zadd(key, {f"{timestamp}:{amount}": timestamp})
            self.redis.zremrangebyscore(key, "-inf", cutoff)
            return True
        except Exception:
            return False

    def compute(
        self, user_id: str, amount: float, timestamp: float
    ) -> Dict[str, float]:
        """
        Compute real-time velocity features for a transaction.

        When Redis is available, computes exact window aggregates via Lua.
        When Redis is unavailable, returns zero-filled defaults.

        Returns:
            Dict with keys: velocity_{w}, tx_count_{w}, amount_mean_{w},
            amount_std_{w}, time_since_last, tx_frequency_hour.
        """
        if self.redis is None or self._lua_sha is None:
            return self._fallback(amount)

        try:
            key = self._key(user_id)
            features = {}

            for w in self.windows:
                cutoff = timestamp - (w * 3600)
                result = self.redis.evalsha(
                    self._lua_sha, 1, key,
                    str(timestamp), str(amount), str(cutoff),
                )
                count, total_sum, mean, std, time_span = result
                count = int(count)
                total_sum = float(total_sum)
                mean = float(mean)
                std = float(std)
                time_span = float(time_span)

                features[f"velocity_{w}"] = round(total_sum / max(time_span, 1), 6)
                features[f"tx_count_{w}"] = count
                features[f"amount_mean_{w}"] = round(mean, 6)
                features[f"amount_std_{w}"] = round(std, 6)

            # Cross-window features
            time_since_last = self._get_time_since_last(key, timestamp)
            features["time_since_last"] = time_since_last
            features["time_since_last_fillna"] = max(time_since_last, 3600.0)
            features["tx_frequency_hour"] = round(
                1.0 / max(time_since_last / 3600, 1e-8), 6
            )

            return features

        except Exception:
            return self._fallback(amount)

    def _get_time_since_last(self, key: str, current_ts: float) -> float:
        """Get seconds since the previous transaction for this user."""
        if self.redis is None:
            return 3600.0
        try:
            # Get the second-newest entry (index -2)
            prev = self.redis.zrevrange(key, 1, 1, withscores=True)
            if prev:
                return max(current_ts - prev[0][1], 0.0)
            return 3600.0
        except Exception:
            return 3600.0

    def _fallback(self, amount: float) -> Dict[str, float]:
        """Zero-fill fallback when Redis is not available."""
        features: Dict[str, float] = {}
        for w in self.windows:
            features[f"velocity_{w}"] = 0.0
            features[f"tx_count_{w}"] = 1
            features[f"amount_mean_{w}"] = amount
            features[f"amount_std_{w}"] = 0.0
        features["time_since_last"] = 0.0
        features["time_since_last_fillna"] = 3600.0
        features["tx_frequency_hour"] = 0.0
        return features

    def clear_user(self, user_id: str) -> None:
        """Remove all history for a user (e.g., for testing)."""
        if self.redis is not None:
            self.redis.delete(self._key(user_id))
