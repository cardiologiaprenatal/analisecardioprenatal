@echo off
cd /d "%~dp0"
echo Fechando necessario: feche data\dataset.csv no editor antes de continuar.
pause
py add_id_gestante.py
if %ERRORLEVEL%==0 (
  echo.
  echo Pronto! Abra data\dataset.csv novamente.
) else (
  echo.
  echo Se falhou, copie manualmente dataset.csv para data\dataset.csv
)
pause
