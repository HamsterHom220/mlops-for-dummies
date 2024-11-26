Run the app locally

1)`cd deploy`\
`python3 api.py`\
2)`cd deploy`\
`streamlit run app.py --server.port=8501`

Run the app via Docker

`docker compose up`

Run the app via Airflow
- `airflow scheduler --daemon --log-file services/airflow/logs/scheduler.log`
-  `airflow triggerer --daemon --log-file services/airflow/logs/triggerer.log`
-  `airflow webserver --daemon --log-file services/airflow/logs/webserver.log`