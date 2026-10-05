@echo off
REM Encode every .mkv in the current folder with av1_chunk_encode.py.
REM Finished files go to .\out, temporary files to .\work.
REM Extra arguments are passed through to the script; --era is required, e.g.
REM   encode_chunked.bat --era 2 --target 95
REM Re-running the same command resumes an interrupted encode.
py -3 -u "%~dp0av1_chunk_encode.py" "%CD%\*.mkv" --outdir "%CD%\out" --work "%CD%\work" %*
pause
