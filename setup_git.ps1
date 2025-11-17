# Script PowerShell para configurar y subir el repositorio a GitHub
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configurando repositorio Git" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si git está instalado
try {
    $gitVersion = git --version
    Write-Host "✓ Git encontrado: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git no está instalado o no está en el PATH" -ForegroundColor Red
    Write-Host "Por favor instala Git desde https://git-scm.com/" -ForegroundColor Yellow
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Verificar si ya existe un repositorio
if (Test-Path .git) {
    Write-Host "El repositorio Git ya está inicializado." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "Inicializando repositorio Git..." -ForegroundColor Cyan
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo inicializar el repositorio" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "✓ Repositorio inicializado" -ForegroundColor Green
    Write-Host ""
}

# Agregar archivos
Write-Host "Agregando archivos al staging..." -ForegroundColor Cyan
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: No se pudieron agregar los archivos" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}
Write-Host "✓ Archivos agregados" -ForegroundColor Green
Write-Host ""

# Verificar si hay cambios para commitear
$status = git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No hay cambios nuevos para commitear." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "Creando commit inicial..." -ForegroundColor Cyan
    git commit -m "Initial commit: Algoritmo Skatter - Estimación Geoestadística"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo crear el commit" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "✓ Commit creado" -ForegroundColor Green
    Write-Host ""
}

# Configurar remote
Write-Host "Configurando repositorio remoto..." -ForegroundColor Cyan
git remote remove origin 2>$null
git remote add origin https://github.com/Igrojas/Tesina_estimacion_geoestadistica.git
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: No se pudo configurar el remote" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}
Write-Host "✓ Remote configurado" -ForegroundColor Green
Write-Host ""

# Configurar rama main
Write-Host "Configurando rama principal..." -ForegroundColor Cyan
git branch -M main 2>$null
Write-Host "✓ Rama principal configurada" -ForegroundColor Green
Write-Host ""

# Push
Write-Host "Subiendo cambios a GitHub..." -ForegroundColor Cyan
Write-Host "NOTA: Se te pedirá autenticación (usuario y token de acceso personal)" -ForegroundColor Yellow
Write-Host ""
git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: No se pudo subir los cambios" -ForegroundColor Red
    Write-Host ""
    Write-Host "Posibles causas:" -ForegroundColor Yellow
    Write-Host "- No estás autenticado en GitHub"
    Write-Host "- El repositorio remoto no existe o no tienes permisos"
    Write-Host "- Problemas de conexión"
    Write-Host ""
    Write-Host "Para autenticarte, crea un Personal Access Token en GitHub:" -ForegroundColor Yellow
    Write-Host "https://github.com/settings/tokens" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "¡Repositorio configurado y subido exitosamente!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Puedes ver tu repositorio en:" -ForegroundColor Cyan
Write-Host "https://github.com/Igrojas/Tesina_estimacion_geoestadistica" -ForegroundColor Cyan
Write-Host ""
Read-Host "Presiona Enter para salir"

