# module-diff.ps1 - 对比开启插件前后目标进程的模块快照, 定位"第三方软件加载"来源
param(
  [string]$ProcessName,   # 游戏进程名, 如 "game"
  [string]$Mode = "snap", # snap | diff
  [string]$SnapFile = ".\module-snapshot.txt"
)
if ($Mode -eq "snap") {
  $p = Get-Process -Name $ProcessName -ErrorAction Stop
  $m = $p.Modules | ForEach-Object { $_.FileName } | Sort-Object
  $m | Set-Content $SnapFile
  "snapshot $($m.Count) modules -> $SnapFile"
} else {
  if (-not (Test-Path $SnapFile)) { throw "先跑 snap 模式生成基线" }
  $before = Get-Content $SnapFile
  $p = Get-Process -Name $ProcessName -ErrorAction Stop
  $now = $p.Modules | ForEach-Object { $_.FileName } | Sort-Object
  $new = $now | Where-Object { $_ -notin $before }
  if ($new) { "NEW_MODULES:"; $new } else { "无新增模块" }
  "CURRENT_COUNT: $($now.Count)  BEFORE: $($before.Count)"
}
