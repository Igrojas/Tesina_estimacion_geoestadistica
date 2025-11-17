@echo off
REM Script para configurar y subir el repositorio a GitHub
chcp 65001 >nul
echo ========================================
echo Configurando repositorio Git
echo ========================================
echo.

REM Verificar si git está instalado
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git no está instalado o no está en el PATH
    echo Por favor instala Git desde https://git-scm.com/
    pause
    exit /b 1
)

REM Verificar si ya existe un repositorio
if exist .git (
    echo El repositorio Git ya está inicializado.
    echo.
) else (
    echo Inicializando repositorio Git...
    git init
    if %errorlevel% neq 0 (
        echo ERROR: No se pudo inicializar el repositorio
        pause
        exit /b 1
    )
    echo ✓ Repositorio inicializado
    echo.
)

REM Agregar archivos
echo Agregando archivos al staging...
git add .
if %errorlevel% neq 0 (
    echo ERROR: No se pudieron agregar los archivos
    pause
    exit /b 1
)
echo ✓ Archivos agregados
echo.

REM Verificar si hay cambios para commitear
git diff --cached --quiet
if %errorlevel% equ 0 (
    echo No hay cambios nuevos para commitear.
    echo.
) else (
    echo Creando commit inicial...
    git commit -m "Initial commit: Algoritmo Skatter - Estimación Geoestadística"
    if %errorlevel% neq 0 (
        echo ERROR: No se pudo crear el commit
        pause
        exit /b 1
    )
    echo ✓ Commit creado
    echo.
)

REM Configurar remote
echo Configurando repositorio remoto...
git remote remove origin 2>nul
git remote add origin https://github.com/Igrojas/Tesina_estimacion_geoestadistica.git
if %errorlevel% neq 0 (
    echo ERROR: No se pudo configurar el remote
    pause
    exit /b 1
)
echo ✓ Remote configurado
echo.

REM Configurar rama main
echo Configurando rama principal...
git branch -M main 2>nul
echo ✓ Rama principal configurada
echo.

REM Push
echo Subiendo cambios a GitHub...
echo NOTA: Se te pedirá autenticación (usuario y token de acceso personal)
echo.
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo ERROR: No se pudo subir los cambios
    echo.
    echo Posibles causas:
    echo - No estás autenticado en GitHub
    echo - El repositorio remoto no existe o no tienes permisos
    echo - Problemas de conexión
    echo.
    echo Para autenticarte, puedes usar:
    echo   git config --global credential.helper store
    echo   (luego intenta el push nuevamente)
    echo.
    echo O crea un Personal Access Token en GitHub:
    echo   https://github.com/settings/tokens
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ¡Repositorio configurado y subido exitosamente!
echo ========================================
echo.
echo Puedes ver tu repositorio en:
echo https://github.com/Igrojas/Tesina_estimacion_geoestadistica
echo.
pause

