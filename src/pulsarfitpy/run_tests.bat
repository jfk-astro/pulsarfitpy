@echo off
REM Script to run all tests for pulsarfitpy on Windows

echo ========================================
echo pulsarfitpy Test Cases
echo ========================================
echo.

REM Change to the package directory
cd /d "%~dp0"

REM Check if pytest is installed
pytest --version >nul 2>&1
if errorlevel 1 (
    echo Pytest was not installed; installing test dependencies...
    pip install -e ".[test]"
)

echo Running all tests...
echo.

REM Run tests with coverage
pytest --cov=pulsarfitpy --cov-report=term-missing --cov-report=html -v

echo.
echo ========================================
echo Test Summary
echo ========================================
echo.
echo Tests completed!
echo Coverage report generated in htmlcov/index.html
echo ========================================
echo.

pause
