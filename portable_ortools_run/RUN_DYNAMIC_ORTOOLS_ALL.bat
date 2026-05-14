@echo off
setlocal
cd /d "%~dp0"

echo ======================================================
echo Dynamic OR-Tools batch
echo Folder: %CD%
echo Input : dynamic_instances\*.json
echo Limit : 7200 seconds per reschedule
echo ======================================================

python run_dynamic_ortools_all.py

echo.
echo ======================================================
echo Finished. Batch summary is under:
echo   or_tools_solutions\dynamic_ortools_batch
echo Detailed outputs are under:
echo   plots\global
echo ======================================================
pause
