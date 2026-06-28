@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"
set "NOTEBOOK=Waste_Classifier_Trainer.ipynb"
set "ZIP_FILE=%~dp0waste_type_identification.zip"
set "DOWNLOAD_SCRIPT=%~dp0scarica_dataset.py"

if not exist "%PYTHON_EXE%" (
    echo ERRORE: venv non trovato in "%VENV_DIR%".
    echo Crea prima il venv con:
    echo   python -m venv .venv
    pause
    exit /b 1
)

if not exist "%NOTEBOOK%" (
    echo ERRORE: notebook non trovato: "%NOTEBOOK%".
    pause
    exit /b 1
)

call "%ACTIVATE_BAT%"

rem ── Controlla se il dataset ZIP e' presente ───────────────────────────────
if not exist "%ZIP_FILE%" (
    echo.
    echo [INFO] Dataset ZIP non trovato. Avvio download da Google Drive...
    echo.
    "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"
    if errorlevel 1 (
        echo ERRORE: download del dataset fallito.
        pause
        exit /b 1
    )
)
rem ─────────────────────────────────────────────────────────────────────────

"%PYTHON_EXE%" -m jupyter --version >nul 2>&1
if errorlevel 1 (
    echo Jupyter non e' installato nel venv.
    choice /m "Vuoi installare le dipendenze da requirements.txt"
    if errorlevel 2 (
        echo Installazione annullata.
        pause
        exit /b 1
    )

    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERRORE: installazione dipendenze non riuscita.
        pause
        exit /b 1
    )
)

echo Avvio JupyterLab con il venv attivo...
"%PYTHON_EXE%" -m jupyter lab "%NOTEBOOK%"

endlocal
