# Instrucciones para Subir el Repositorio a GitHub

## Opción 1: Usar el Script Automático (Recomendado)

### En Windows (CMD):
Haz doble clic en `setup_git.bat` o ejecuta en la terminal:
```cmd
setup_git.bat
```

### En Windows (PowerShell):
Ejecuta en PowerShell:
```powershell
.\setup_git.ps1
```

## Opción 2: Comandos Manuales

Si prefieres ejecutar los comandos manualmente, sigue estos pasos:

1. **Abre una terminal** en el directorio del proyecto

2. **Inicializa Git** (si no está inicializado):
   ```bash
   git init
   ```

3. **Agrega todos los archivos**:
   ```bash
   git add .
   ```

4. **Crea el commit inicial**:
   ```bash
   git commit -m "Initial commit: Algoritmo Skatter - Estimación Geoestadística"
   ```

5. **Configura el repositorio remoto**:
   ```bash
   git remote add origin https://github.com/Igrojas/Tesina_estimacion_geoestadistica.git
   ```

6. **Establece la rama principal**:
   ```bash
   git branch -M main
   ```

7. **Sube los cambios a GitHub**:
   ```bash
   git push -u origin main
   ```

## Autenticación en GitHub

Si es la primera vez que subes código, GitHub te pedirá autenticarte. Tienes dos opciones:

### Opción A: Personal Access Token (Recomendado)
1. Ve a: https://github.com/settings/tokens
2. Crea un nuevo token con permisos `repo`
3. Cuando Git te pida la contraseña, usa el token en lugar de tu contraseña

### Opción B: GitHub CLI
```bash
gh auth login
```

## Verificar el Repositorio

Una vez completado, puedes ver tu repositorio en:
https://github.com/Igrojas/Tesina_estimacion_geoestadistica

## Solución de Problemas

### Error: "Git no está instalado"
- Descarga e instala Git desde: https://git-scm.com/download/win
- Reinicia la terminal después de instalar

### Error: "Permission denied" o "Authentication failed"
- Verifica que tengas permisos de escritura en el repositorio
- Usa un Personal Access Token en lugar de tu contraseña
- Verifica que el token tenga permisos `repo`

### Error: "Repository not found"
- Verifica que el repositorio existe en GitHub
- Verifica que tienes acceso al repositorio
- Verifica que la URL del remote es correcta

