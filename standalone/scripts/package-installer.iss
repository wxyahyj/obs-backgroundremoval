; YoloAim standalone 安装器(Inno Setup 6)
; 构建: iscc package-installer.iss
; 产物: dist/YoloAim-setup.exe

#define MyAppName "YoloAim Standalone"
#define MyAppVersion "0.1.0"
#define MyAppExeName "yolo_host.exe"

[Setup]
AppId={{7E3F2B1A-9C4D-4E5F-B8A2-1D2C3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\YoloAim
DefaultGroupName=YoloAim
OutputDir=dist
OutputBaseFilename=YoloAim-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
; 从便携目录收集(dist\YoloAim 由 package.ps1 生成)
Source: "dist\YoloAim\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\YoloAim"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 YoloAim"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 YoloAim"; Flags: nowait postinstall skipifsilent
