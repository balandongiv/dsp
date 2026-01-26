# LaTeX Template User Guide

This guide provides instructions for using the thesis LaTeX template, specifically tailored for beginners. It covers the setup, including font configuration and unit handling with `siunitx`.

## 1. Getting Started

### Prerequisites
To compile the document successfully, you need a LaTeX distribution installed on your system.

**For Linux (Debian/Ubuntu):**
The template requires `texlive-full` or at least `texlive-science` for unit handling.
```bash
sudo apt-get update
sudo apt-get install texlive-latex-extra texlive-science texlive-xetex
```

### Compiling the Document
This template uses custom fonts (Tahoma), so you **must** use the `xelatex` compiler.

**Command Line:**
```bash
cd writing
xelatex main.tex
```

**VS Code (LaTeX Workshop):**
Ensure your recipe is set to `latexmk (xelatex)` or directly `xelatex`.

## 2. Using Units and Numbers (`siunitx`)

We use the `siunitx` package to ensure that all numbers and units are formatted correctly and use the document's main font (Tahoma).

**Why use it?**
- It ensures consistency (e.g., spacing between number and unit).
- It prevents units from looking like math variables (italicized).
- It handles complex units automatically.

### Basic Usage

**1. Quantities (`\SI`)
Use this for a number with a unit.
```latex
The volume is \SI{10}{\liter}.
The temperature is \SI{298}{\kelvin}.
```
*Output:* The volume is 10 L. (in Tahoma font)

**2. Just Units (`\si`)
Use this when you only need the unit symbol.
```latex
Measured in \si{\kilo\gram}.
```

**3. Just Numbers (`\num`)
Use this for formatting large numbers.
```latex
\num{10000} % Output: 10,000 (depending on settings)
```

**4. Ranges (`\SIrange`)
```latex
The temperature varies from \SIrange{10}{20}{\celsius}.
```

### Common Units
- `\liter` -> L
- `\kelvin` -> K
- `\meter` -> m
- `\second` -> s
- `\kilogram` -> kg
- `\degreeCelsius` -> °C

## 3. Font Settings (Advanced)

The project uses **Tahoma** as the main font. This is configured in `font/fin.sty`.

- **Regular Text:** Tahoma Regular
- **Bold Text:** Tahoma Bold
- **Italic Text:** Tahoma Regular with a "Fake Slant" (since a true italic font file is not provided).
- **Math:** Uses Tahoma for digits and Latin letters to match the text.

**Note:** If you see warnings about "Fake Slant", this is normal and intentional to create the italic effect.

## 4. Troubleshooting

**"Undefined control sequence \SI"**
- This means the `siunitx` package is not loaded or installed.
- **Fix:** Install `texlive-science` (Linux) or the `siunitx` package via your LaTeX package manager.

**"Font not found"**
- Ensure `font/tahoma.ttf` and `font/tahomabd.ttf` exist in the `writing/font/` directory.
- Ensure you are compiling with `xelatex`, not `pdflatex`.
