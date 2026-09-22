@echo off
REM ── เปิดแอป Telco Churn Predictor (Workshop 11 · ทีมรีเทค) ──
cd /d "%~dp0"
echo กำลังเปิดแอป... เปิดเบราว์เซอร์ที่ http://localhost:8501
echo (ปิดหน้าต่างนี้เพื่อหยุดแอป)
".\venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
pause
