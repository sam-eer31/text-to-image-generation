@echo off
cd /d "%~dp0"

:main
:: Activate virtual environment
call myenv\Scripts\activate



:: Start the Python app in the background
start "" /B python txt_to_img.py

:: Display a loading message
echo Starting the Image-Generator...
echo Please wait...

:: Simple loading animation
:LOADING
setlocal enabledelayedexpansion
set "chars=\|/-"
set /a count=0
:LOOP
:: Calculate the current character position
set /a pos=count %% 4
set "char=!chars:~%pos%,1!"
echo Loading... !char!
timeout /t 1 >nul
curl -s http://127.0.0.1:7860/ >nul
if errorlevel 1 (
    cls
    set /a count+=1
    goto LOOP
)

:: Server is up, open the browser
start http://127.0.0.1:7860/

:: Exit the script
exit