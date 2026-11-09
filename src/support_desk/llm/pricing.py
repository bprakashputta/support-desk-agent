"""Per-token prices (USD per million tokens), Anthropic first-party API, Nov 2026.

Cache multipliers follow Anthropic's prompt-caching pricing: a 5-minute cache write costs
1.25x the input rate, a cache read 0.1x. Haiku 5.5 charges a higher "long prompt" rate
($0.50 / $2.50) when a prompt exceeds 100K tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    long_input: float | None = None  # rate above `long_threshold` prompt tokens
    long_output: float | None = None
    long_threshold: int = 100_000


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, long_input=0.50, long_output=2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
}

CACHE_WRITE = 1.25
CACHE_READ = 0.10


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Cost of one call. `input_tokens` is the *uncached* part, as the API reports it."""
    price = PRICES[model]
    prompt = input_tokens + cache_read + cache_write
    long = price.long_input is not None and prompt > price.long_threshold
    rate_in = price.long_input if long else price.input
    rate_out = price.long_output if long and price.long_output else price.output
    return (
        input_tokens * rate_in
        + cache_write * rate_in * CACHE_WRITE
        + cache_read * rate_in * CACHE_READ
        + output_tokens * rate_out
    ) / 1_000_000
