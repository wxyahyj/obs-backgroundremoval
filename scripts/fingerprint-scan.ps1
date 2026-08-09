# fingerprint-scan.ps1 - 扫描 OBS 插件产物中的暴露指纹
param(
  [string]$Dll = ".\dist-dml-only\obs-plugins\64bit\obs-backgroundremoval.dll",
  [string]$DataDir = ".\dist-dml-only\data\obs-plugins\obs-backgroundremoval"
)
$keywords = @('yolo','backgroundremoval','background','aim','recoil','mousecontroller','logi','makcu','gvinput','syscall','anticeat','ntuser','inject','floatingwindow','piddebug','udp','detector','royshil','roysh','ghibli')
$raw = [IO.File]::ReadAllBytes((Resolve-Path $Dll))
$sb = New-Object Text.StringBuilder
$hits = @{}
for ($i=0; $i -lt $raw.Length; $i++) {
  $c = $raw[$i]
  if ($c -ge 32 -and $c -lt 127) { [void]$sb.Append([char]$c) }
  else {
    if ($sb.Length -ge 8) {
      $s = $sb.ToString()
      foreach ($k in $keywords) { if ($s -match $k) { $hits[$k] = $true } }
    }
    [void]$sb.Clear()
  }
}
"HITS_IN_DLL: " + (($hits.Keys | Sort-Object) -join ', ')
"PDB nearby: " + [bool](Get-ChildItem (Split-Path $Dll) -Filter *.pdb -ErrorAction SilentlyContinue)
if (Test-Path $DataDir) { Get-ChildItem $DataDir -Recurse -File | ForEach-Object { "DATA: $($_.FullName)" } }
