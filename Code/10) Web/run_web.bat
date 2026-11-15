@echo off
rem BPR 웹 실행: 이 파일을 더블클릭하면 브라우저에서 http://localhost:8501 이 열립니다. 끄려면 이 창을 닫으세요.
chcp 65001 >nul
set E=C:\Users\user\anaconda3\envs\BreastCancer
set PATH=%E%;%E%\Library\mingw-w64\bin;%E%\Library\usr\bin;%E%\Library\bin;%E%\Scripts;%PATH%
set PYTHONIOENCODING=utf-8
cd /d C:\BreastCancer
python -m streamlit run Web\app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
pause
