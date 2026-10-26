"""
gpskit - small, readable tools for analysing STATSports GNSS data.

Modules
-------
io         read raw STATSports exports (one row per player per 10 Hz sample)
filtering  zero-phase Butterworth smoothing and acceleration
profile    in-situ acceleration-speed / force-velocity profiling
sprint     profiling an isolated maximal sprint (ballistic power)
stats      effect sizes, confidence intervals and mixed-effects models
synthetic  invented demo data in the STATSports export format
"""

__version__ = "1.0.0"
