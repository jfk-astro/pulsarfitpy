#!/bin/bash
# Script to run all tests for pulsarfitpy

echo "========================================"
echo "pulsarfitpy Test Cases"
echo "========================================"
echo ""

# Change to the package directory
cd "$(dirname "$0")"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Pytest was not installed; installing test dependencies..."
    pip install -e ".[test]"
fi

echo "Running all tests..."
echo ""

# Run tests with coverage
pytest --cov=pulsarfitpy --cov-report=term-missing --cov-report=html -v

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "Tests completed!"
echo "Coverage report generated in htmlcov/index.html"
echo "========================================"