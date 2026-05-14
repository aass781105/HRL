@echo off
REM ---- ??UTF-8嚗?葉?楝敺?蝣?----
chcp 65001 >NUL

REM ---- ??????Conda ?啣? ----
CALL "C:\Users\lab643\anaconda3\Scripts\activate.bat" newest_environment
REM 憒?銝?????憿??臭誑?寞?銝??嚗摰頝臬???嚗?
REM CALL "C:\Users\lab643\anaconda3\Scripts\activate.bat" "C:\Users\lab643\anaconda3\envs\newest_environment"

REM ---- ?雿?撠?鞈?憭?----
cd /d "C:\Users\lab643\Desktop\蝣拐?\PPO_FJSP\FJSP-DRL-main_NO_GNN"

REM ---- 靘?頝?yaml 鞈?憭曉???.yml ----
for %%f in ("yaml\*.yml") do (
    echo Running config %%f ...
    python train_ddqn.py --config "%%f"
)

echo.
echo All configs done.
pause
