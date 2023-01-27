api: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}
web: sh -c 'cd ./front_end/ && sh setup.sh && cd .. && streamlit run front_end/dashboard.py'