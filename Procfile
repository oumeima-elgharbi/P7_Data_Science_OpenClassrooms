web: sh setup.sh && streamlit run front_end/dashboard.py
server: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}