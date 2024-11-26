from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.sensors.http_sensor import HttpSensor


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['admin@example.org'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
    'execution_timeout': timedelta(days=1),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

with DAG(
    'dag-flip',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval=timedelta(hours=2),
    catchup=False,
) as dag:

    # train_task = PythonOperator(
    #     task_id='train',
    #     python_callable=train,
    #     provide_context=True,
    # )
    train_task = BashOperator(
        task_id='train',
        bash_command='python ~/MLOps/flipped_class/pipeline_jobs/train.py',
    )

    deploy_task = BashOperator(
        task_id='deploy',
        bash_command='cd ~/MLOps/flipped_class && docker compose up -d',
    )

    wait_for_service = HttpSensor(
        task_id='wait_for_service',
        http_conn_id='flip_api',
        endpoint='/',
        method='GET',
        response_check=lambda response: response.status_code == 200,
        poke_interval=5,
        timeout=300,
    )

train_task >> deploy_task >> wait_for_service
