web: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}
dashboard: sh front_end.setup.sh && streamlit run front_end.dashboard.py