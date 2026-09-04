#!/bin/bash

# EASE Paper Compilation Script
# Usage: ./compile.sh [submission|camera-ready] [clean]

set -e

MODE="${1:-submission}"
ACTION="${2:-}"

MAIN_FILE=""
OUTPUT_NAME=""

case "$MODE" in
    submission)
        MAIN_FILE="ease-submission.tex"
        OUTPUT_NAME="ease-submission"
        echo "Compiling SUBMISSION version (anonymous, review mode)..."
        ;;
    camera-ready|camera)
        MAIN_FILE="ease-camera-ready.tex"
        OUTPUT_NAME="ease-camera-ready"
        echo "Compiling CAMERA-READY version..."
        ;;
    *)
        echo "Usage: $0 [submission|camera-ready] [clean]"
        echo "  submission   - Anonymous review version (default)"
        echo "  camera-ready - Camera-ready version with author names"
        echo "  clean        - Clean auxiliary files after compilation"
        exit 1
        ;;
esac

if [ ! -f "$MAIN_FILE" ]; then
    echo "Error: $MAIN_FILE not found!"
    exit 1
fi

echo "Using main file: $MAIN_FILE"
echo "Output will be: ${OUTPUT_NAME}.pdf"
echo ""

# Compile with latexmk (recommended) or fallback to pdflatex+bibtex
if command -v latexmk &> /dev/null; then
    echo "Using latexmk for compilation..."
    latexmk -pdf -interaction=nonstopmode -output-directory=. "$MAIN_FILE"
else
    echo "Using pdflatex + bibtex (latexmk not found)..."
    pdflatex -interaction=nonstopmode "$MAIN_FILE"
    bibtex "${MAIN_FILE%.tex}"
    pdflatex -interaction=nonstopmode "$MAIN_FILE"
    pdflatex -interaction=nonstopmode "$MAIN_FILE"
fi

# Rename output if needed
if [ -f "${MAIN_FILE%.tex}.pdf" ] && [ "${MAIN_FILE%.tex}.pdf" != "${OUTPUT_NAME}.pdf" ]; then
    mv "${MAIN_FILE%.tex}.pdf" "${OUTPUT_NAME}.pdf"
fi

echo ""
echo "✓ Compilation complete: ${OUTPUT_NAME}.pdf"

# Clean auxiliary files if requested
if [ "$ACTION" = "clean" ] || [ "$MODE" = "clean" ]; then
    echo "Cleaning auxiliary files..."
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
    echo "✓ Cleaned"
fi

# Show file size
if [ -f "${OUTPUT_NAME}.pdf" ]; then
    SIZE=$(du -h "${OUTPUT_NAME}.pdf" | cut -f1)
    echo "PDF size: $SIZE"
fi