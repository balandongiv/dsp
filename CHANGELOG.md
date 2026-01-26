# Versioning Policy

**Versioning must follow Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`**

- **MAJOR** — breaking changes
- **MINOR** — new features (backward compatible)
- **PATCH** — bug fixes

Examples:

- `2.0.0` → breaking API change
- `1.4.0` → new feature added
- `1.4.1` → bug fix

# Changelog

## [1.1.0] - 2026-01-16

### Added

- Enabled Table of Contents in `fin.sty`.

### Fixed

- Fixed empty List of Figures/Tables by adding `\clearpage` after `\tableofcontents`.
- Fixed LaTeX compilation error "Extra }, or forgotten \endgroup" in ToC by changing `\textbf` to `\bfseries` for `\cftchapfont`.
- Fixed formatting issue in ToC by removing unnecessary newline in `\cftchappresnum`.
- Improved `build.sh` robustness by allowing LaTeX tools to return non-zero exit codes without failing the entire build script.

## [1.0.0] - 2025-03-14

### Added

- Compliance with UMS Thesis Format: The template now fully aligns with the official UMS thesis guidelines.
- Line Numbering: Each line is now numbered to assist examiners in reviewing and correcting the thesis more efficiently.
- Clickable URLs: Hyperlinks have been added to references, allowing examiners to verify sources quickly.
- Improved LaTeX Code Structure: The syntax has been cleaned up, especially in Chapter 0 (Preface), to reduce confusion for students.
