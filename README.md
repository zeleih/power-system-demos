# Power System Demos

Public, reproducible simulation demos for power-system research.

## Available demos

| Demo | Description |
| --- | --- |
| [`deadband/`](deadband/) | Reproduction package for deadband-edge frequency accumulation and heterogeneous deadband design. |
| [`gscr/`](gscr/) | Analytical PMU-based gSCR identification, with CEPRI36 PSASP/ANDES workflows and an ANDES IL200 extension. |

Each demo is self-contained and includes its own environment description,
case documentation, scripts, reference outputs, and validation entry points.
The deadband package is also available as the standalone
[`zeleih/deadband`](https://github.com/zeleih/deadband) repository. The gSCR
package links to the paper by DOI and does not redistribute the article PDF or
license-unconfirmed raw PSASP records.

Additional demos are maintained privately until their release packages have
been reviewed and validated.

## Import policy

Each public demo is imported as a current, reviewed snapshot without its
earlier development history. This keeps reader downloads small and prevents
experimental checkpoints or generated binaries from entering the public
repository. See [`IMPORT_MANIFEST.md`](IMPORT_MANIFEST.md) for the exact source
commit.
