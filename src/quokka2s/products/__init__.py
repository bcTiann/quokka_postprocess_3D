"""Streaming products with per-line emission selection and independent gas statistics."""

# Saved arrays always put intrinsic first and dust-attenuated second.
DUST_STATES = ("intrinsic", "attenuated")

# Stable saved names for the two QUOKKA-temperature branches, in array order.
COLD_REGIME_KEY = "T_QUOKKA_lt_3000K"
HOT_REGIME_KEY = "T_QUOKKA_ge_3000K"
REGIME_KEYS = (COLD_REGIME_KEY, HOT_REGIME_KEY)
