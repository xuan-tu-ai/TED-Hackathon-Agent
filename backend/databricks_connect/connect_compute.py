from databricks import sql
from dotenv import load_dotenv
import os

load_dotenv()

conn = sql.connect(
    server_hostname=os.getenv("DATABRICKS_HOST"),
    http_path=os.getenv("DATABRICKS_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN")
)

with conn.cursor() as cursor:
    # cursor.execute("SELECT current_user();")
    cursor.execute("SELECT * from dbw_ted_hackathon_dev_001.bronze_layer.robot_data_inference limit 10 ;")
    rows = cursor.fetchall()
    print(rows)