@echo off
rem 部署 obs-backgroundremoval.dll 到 Steam OBS 插件目录
rem 用法: deploy_obs.bat [config]  默认 RelWithDebInfo
setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=RelWithDebInfo
set SRC=E:\obs-heji\obs-backgroundremoval\build_x64_local\%CONFIG%\obs-backgroundremoval.dll
set DST=D:\steam\steamapps\common\OBS Studio\obs-plugins\64bit\obs-backgroundremoval.dll
if not exist "%SRC%" (
    echo [ERROR] 未找到 %SRC% ，先编译
    exit /b 1
)
copy /y "%SRC%" "%DST%" >nul
echo [OK] 已部署 %CONFIG% dll 到 OBS 插件目录
echo      重启 OBS 生效
