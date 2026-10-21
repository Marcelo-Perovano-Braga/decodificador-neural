[CmdletBinding()]
param(
    [switch]$SkipDependencyInstall,
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot

try {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $PythonExe = $PythonCommand.Source

    $Platform = (& $PythonExe -c "import sys; print(sys.platform)").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Não foi possível consultar o interpretador Python."
    }
    if ($Platform -ne "win32") {
        throw "A geração do .exe deve ser executada no Windows."
    }

    $PointerBits = (& $PythonExe -c "import struct; print(struct.calcsize('P') * 8)").Trim()
    if ($LASTEXITCODE -ne 0 -or $PointerBits -ne "64") {
        throw "Utilize uma instalação Python de 64 bits para gerar o aplicativo."
    }

    if (-not (Test-Path "weights\classifier.pt" -PathType Leaf)) {
        throw (
            "Checkpoint ausente em weights\classifier.pt. " +
            "Execute train_pipeline.py ou copie o checkpoint já treinado."
        )
    }

    if (-not $SkipDependencyInstall) {
        Write-Host "Instalando dependências da aplicação..." -ForegroundColor Cyan
        & $PythonExe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao instalar requirements.txt."
        }

        Write-Host "Instalando dependências de empacotamento..." -ForegroundColor Cyan
        & $PythonExe -m pip install -r requirements-build.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao instalar requirements-build.txt."
        }
    }

    Write-Host "Gerando aplicativo Windows..." -ForegroundColor Cyan
    & $PythonExe -m PyInstaller --clean --noconfirm IzhikevichClassifier.spec
    if ($LASTEXITCODE -ne 0) {
        throw "O PyInstaller não concluiu a geração do aplicativo."
    }

    $DistributionDirectory = Join-Path $ProjectRoot "dist\ClassificadorIzhikevich"
    $ExecutablePath = Join-Path $DistributionDirectory "ClassificadorIzhikevich.exe"
    if (-not (Test-Path $ExecutablePath -PathType Leaf)) {
        throw "Build finalizado sem produzir $ExecutablePath."
    }

    if (-not $SkipArchive) {
        $ArchivePath = Join-Path $ProjectRoot "dist\ClassificadorIzhikevich-Windows-x64.zip"
        if (Test-Path $ArchivePath -PathType Leaf) {
            Remove-Item $ArchivePath -Force
        }
        Write-Host "Compactando a distribuição..." -ForegroundColor Cyan
        Compress-Archive -Path "$DistributionDirectory\*" -DestinationPath $ArchivePath
        Write-Host "Pacote: $ArchivePath" -ForegroundColor Green
    }

    $Hash = (Get-FileHash $ExecutablePath -Algorithm SHA256).Hash
    Write-Host "Executável: $ExecutablePath" -ForegroundColor Green
    Write-Host "SHA-256: $Hash" -ForegroundColor Green
    Write-Host (
        "Mantenha o executável dentro da pasta ClassificadorIzhikevich; " +
        "as dependências ficam no subdiretório _internal."
    ) -ForegroundColor Yellow
}
finally {
    Pop-Location
}
