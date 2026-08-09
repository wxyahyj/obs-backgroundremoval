@echo off
rem Deploy obs-backgroundremoval.dll to Steam OBS plugins dir
rem Usage: deploy_obs.bat [config]  (default RelWithDebInfo)
setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=RelWithDebInfo
set SRC=E:\obs-heji\obs-backgroundremoval\build_x64_local\%CONFIG%\obs-backgroundremoval.dll
set DST=D:\steam\steamapps\common\OBS Studio\obs-plugins\64bit\obs-backgroundremoval.dll
if not exist "%SRC%" (
    echo [ERROR] build output not found: %SRC%
    exit /b 1
)
copy /y "%SRC%" "%DST%" >nul
echo [OK] deployed %CONFIG% dll to OBS plugins dir
echo Restart OBS to take effect
