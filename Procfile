web: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}
dashboard: sh -c 'cd ./front_end/ && sh setup.sh && streamlit run dashboard.py'