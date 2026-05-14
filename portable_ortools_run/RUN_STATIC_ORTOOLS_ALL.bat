@echo off
setlocal
cd /d "%~dp0"

echo ======================================================
echo Static OR-Tools benchmark
echo Folder: %CD%
echo Input : or_instances_uniform_test_30_50
echo Limit : 7200 seconds per static instance
echo ======================================================

python benchmark_ortools.py

echo.
echo ======================================================
echo Finished. Outputs are under:
echo   or_tools_solutions\static_ortools_YYYYMMDD_HHMMSS
echo ======================================================
pause
