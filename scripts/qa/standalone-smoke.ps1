# YoloAim standalone 产品冒烟(回归门禁)
# 用法: powershell -ExecutionPolicy Bypass -File standalone-smoke.ps1 [exe路径]
$ErrorActionPreference = "Stop"
$HostExe = if ($args[0]) { $args[0] } else { ".\build\RelWithDebInfo\yolo_host.exe" }
$Base = "http://127.0.0.1:17890"
$Failed = 0

function Assert($cond, $name) {
    if ($cond) { Write-Host "  [PASS] $name" -ForegroundColor Green }
    else { Write-Host "  [FAIL] $name" -ForegroundColor Red; $script:Failed++ }
}

Write-Host "== 启动 host: $HostExe" -ForegroundColor Cyan
$p = Start-Process -FilePath $HostExe -PassThru -WindowStyle Hidden

try {
    # 等健康检查(最多 15s)
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $h = Invoke-RestMethod "$Base/api/health" -TimeoutSec 2
            if ($h.ok) { $ready = $true; break }
        } catch { }
    }
    Assert $ready "health ok"

    if ($ready) {
        Start-Sleep -Seconds 4
        $st = Invoke-RestMethod "$Base/api/status"
        Assert ($st.running -eq $true) "engine running"
        Assert ($st.capture_ok -eq $true) "capture ok"
        Assert ($st.frames -gt 0) "frames flowing"
        Assert ($st.aim.controller_ok -eq $true) "controller ok"

        # 配置热更新
        $body = '{"aim":{"fov_radius":150}}' | ConvertTo-Json -Compress
        $r = Invoke-RestMethod "$Base/api/config" -Method Put -ContentType "application/json" -Body '{"aim":{"fov_radius":150}}'
        Assert ($r.ok -eq $true) "config PUT ok"
        $cfg = Invoke-RestMethod "$Base/api/config"
        Assert ($cfg.config.aim.fov_radius -eq 150) "config applied (fov=150)"

        # 预览
        try {
            $bmp = Invoke-WebRequest "$Base/api/preview.bmp" -TimeoutSec 5 -UseBasicParsing
            Assert ($bmp.RawContentStream.Length -gt 1000) "preview.bmp"
        } catch { Assert $false "preview.bmp" }

        # 检测接口(PowerShell 空数组解析为 $null,判 ok 即可)
        $dets = Invoke-RestMethod "$Base/api/detections"
        Assert ($dets.ok -eq $true) "detections endpoint"
    }
} finally {
    Write-Host "== 停止 host ==" -ForegroundColor Cyan
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force }
}

Write-Host ""
if ($Failed -eq 0) { Write-Host "SMOKE: ALL PASS" -ForegroundColor Green; exit 0 }
else { Write-Host "SMOKE: $Failed FAILED" -ForegroundColor Red; exit 1 }
