# M1 build script for standalone host
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Gen = "Visual Studio 17 2022"
$BuildDir = Join-Path $Root "build"

Write-Host "== configure ==" -ForegroundColor Cyan
cmake -B $BuildDir -G $Gen -A x64 `
  -DYA_WITH_WGC=ON `
  -DYA_BUILD_CAPTURE_SHARED=ON `
  -DYA_WITH_INFER=OFF `
  -DYA_WITH_AIM=OFF

Write-Host "== build ==" -ForegroundColor Cyan
cmake --build $BuildDir --config RelWithDebInfo --parallel

Write-Host "== done ==" -ForegroundColor Green
Write-Host "Run: $BuildDir\RelWithDebInfo\yolo_host.exe"
