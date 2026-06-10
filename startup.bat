@echo off
set current_dir=%~dp0

echo current working directory: %current_dir%

::app.exe

%current_dir%/.venv/Scripts/python app/main.py

