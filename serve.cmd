@echo off
REM ====================================================================
REM  Serve the course materials over HTTP so the live code cells in the
REM  Reveal.js lectures actually run. Double-click this file, then open
REM  the address it prints (http://localhost:8000/session_01/session_01.html).
REM
REM  The live cells use Pyodide, which browsers block on bare file:// URLs.
REM  Serving over HTTP is what makes the "Run" buttons work.
REM
REM  Press Ctrl+C in this window to stop the server when you are done.
REM ====================================================================
cd /d "%~dp0"
echo.
echo   Machine Learning in Finance - local server
echo   ------------------------------------------
echo   Open:  http://localhost:8000/session_01/session_01.html
echo   Stop:  press Ctrl+C in this window
echo.
python -m http.server 8000
