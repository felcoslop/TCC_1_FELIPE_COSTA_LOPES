from influxdb import InfluxDBClient
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('code/config.env')
load_dotenv(env_path)

host = os.getenv('INFLUXDB_IP')
port = int(os.getenv('INFLUXDB_PORT', 8086))
database = "aihub"

client = InfluxDBClient(host=host, port=port, database=database)

for mp in ["c_637", "c_640"]:
    query = f'SELECT * FROM "estimated" WHERE "m_point" = \'{mp}\' ORDER BY time DESC LIMIT 1'
    try:
        res = client.query(query)
        points = list(res.get_points())
        if points:
            print(f"Latest for {mp}: {points[0]['time']}")
        else:
            print(f"No data at all for {mp} in 'estimated'")
            
        query_first = f'SELECT * FROM "estimated" WHERE "m_point" = \'{mp}\' ORDER BY time ASC LIMIT 1'
        res_first = client.query(query_first)
        points_first = list(res_first.get_points())
        if points_first:
            print(f"First for {mp}: {points_first[0]['time']}")
    except Exception as e:
        print(f"Error querying {mp}: {e}")
