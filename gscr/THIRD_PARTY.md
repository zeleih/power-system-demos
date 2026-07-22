# Third-party material and provenance

## Article

The article is linked by DOI and is not redistributed as a PDF:

Z. Han, P. Ju, H. Li, and Y. Liu, "Analytical Identification Method of
Generalized Short-Circuit Ratio Using Phasor Measurement Units," 2025,
<https://doi.org/10.1049/gtd2.70026>.

## ANDES

ANDES is an external dependency and is not vendored here:

- source: <https://github.com/CURENT/andes>
- license: GNU General Public License v3.0
- reference version: 2.0.0

## IL200 case

`cases/il200/data/IL200_ffr.xlsx` originates from the CURENT demo repository:

- source: <https://github.com/CURENT/demo/tree/master/demo/measurements/cases>
- repository license: MIT, unless a subdirectory states otherwise

Retain the original attribution and repository license notice when
redistributing the case.

## PSASP records

Raw records extracted from a locally installed PSASP example case are not
included because their redistribution permission has not been documented.
The public package contains only the CSV import interface, the converted ANDES
workbook, archived reproduction PMU data, and derived matrices/results.

Users may supply their own authorized records locally under
`cases/cepri36/data/raw_psasp/`. That directory is excluded by `.gitignore`.
