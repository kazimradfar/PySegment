; Inno Setup script: setup wizard for the desktop workstation (built by .github/workflows/build-rsw.yml after
; PyInstaller has produced dist\PySegment). Installs per user (no administrator rights).
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef AppName
  #define AppName "PySegment"
#endif
; file-name form of the display name (spaces -> underscores), used for the setup file
#define AppFile StringChange(AppName, " ", "_")

; Display name ({#AppName}) comes from the program (APP_NAME, passed by the workflow as /DAppName).
; Internal names below (AppId, install folder, data/workspace folder, .exe names) stay fixed on purpose,
; so an update keeps the same installation and data. See docs/EDITING_GUIDE_FA.md (section 11).
[Setup]
AppId={{6C1E0B7A-3D52-4F6B-9A0E-71B2C4D8E9F3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Kazim Radfar
AppCopyright=Copyright (C) 2026 Kazim Radfar
AppComments=Segmentation & Radiomics Workstation (research use only)
DefaultDirName={localappdata}\Programs\PySegment
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=..\rsw\resources\rsw.ico
UninstallDisplayIcon={app}\PySegment.exe
UninstallDisplayName={#AppName}
LicenseFile=..\rsw\LICENSE
Compression=lzma2/max
SolidCompression=yes
OutputDir=..\installer_out
OutputBaseFilename={#AppFile}_Setup_v{#AppVersion}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\PySegment\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\rsw\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\rsw\README_FA.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\PySegment.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\PySegment.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\PySegment.exe"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
