# 打包 YoloAim standalone → 便携版 ZIP
# 用法: powershell -ExecutionPolicy Bypass -File scripts/package.ps1 [-NoZip]
param(
    [switch]$NoZip
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Build = Join-Path $Root "build\RelWithDebInfo"
$Dist = Join-Path $Root "dist\YoloAim"

if (-not (Test-Path (Join-Path $Build "yolo_host.exe"))) {
    Write-Host "构建产物不存在,先构建:" -ForegroundColor Yellow
    & cmake --build (Join-Path $Root "build") --config RelWithDebInfo --target yolo_host --parallel
}

Write-Host "== 收集产物 ==" -ForegroundColor Cyan
if (Test-Path $Dist) { Remove-Item $Dist -Recurse -Force }
New-Item -ItemType Directory -Path $Dist -Force | Out-Null

# exe + capture dll
Copy-Item (Join-Path $Build "yolo_host.exe") $Dist
Copy-Item (Join-Path $Build "yolo_capture.dll") $Dist

# ORT + CUDA 运行库(全部 dll)
Get-ChildItem $Build -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $Dist }

# webui / config / models
Copy-Item (Join-Path $Build "webui") $Dist -Recurse
Copy-Item (Join-Path $Build "config") $Dist -Recurse
if (Test-Path (Join-Path $Build "models")) {
    Copy-Item (Join-Path $Build "models") $Dist -Recurse
}

# 启动器 README
@"
YoloAim Standalone - 独立版(脱离 OBS)
启动: 双击 YoloAim.exe(当前为 yolo_host.exe)
WebUI: 启动后浏览器打开 http://127.0.0.1:17890
配置: config/user.json(WebUI 保存生成);损坏自动备份恢复
日志: logs/host.log(自动轮转)
模型: models/*.onnx, 支持 YOLOv5/v8/v11 导出
"@ | Set-Content (Join-Path $Dist "README.txt") -Encoding UTF8

if (-not $NoZip) {
    Write-Host "== 打 ZIP ==" -ForegroundColor Cyan
    $Zip = Join-Path $Root "dist\YoloAim-standalone.zip"
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    Compress-Archive -Path $Dist -DestinationPath $Zip -CompressionLevel Optimal
    Write-Host "打包完成: $Zip" -ForegroundColor Green
} else {
    Write-Host "打包完成(目录): $Dist" -ForegroundColor Green
}
