# ANDES IL200 mixed-source extension

This extension uses `IL200_ffr.xlsx` from the CURENT demo repository. The
49 active-source terminals are measurement ports; only the 11 REGCA1 buses are
final gSCR ports.

Run from the gSCR case root:

```bash
python scripts/run_il200.py
```

The script regenerates seven regional fault records, identifies the 49-port
external network, terminates 38 synchronous-machine ports, calculates the final
11-port gSCR, and writes tables, matrices, figures, PMU records, and a JSON
summary below `cases/il200/outputs/`.
