@echo off
REM ---- 用 UTF-8，避免中文路徑亂碼 ----
chcp 65001 >NUL

REM ---- 啟動指定的 Conda 環境 ----
CALL "C:\Users\lab643\anaconda3\Scripts\activate.bat" newest_environment
REM 如果上面那行有問題，可以改成下面這行（用完整路徑啟動）
REM CALL "C:\Users\lab643\anaconda3\Scripts\activate.bat" "C:\Users\lab643\anaconda3\envs\newest_environment"

REM ---- 切到你的專案資料夾 ----
cd /d "C:\Users\lab643\Desktop\碩一\PPO_FJSP\FJSP-DRL-main_NO_GNN"

REM ---- 依序跑 yaml 資料夾內所有 .yml ----
for %%f in ("yaml\*.yml") do (
    echo Running config %%f ...
    python train_ddqn.py --config "%%f"
)

echo.
echo All configs done.
pause
