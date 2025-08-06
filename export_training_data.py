from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://host.docker.internal:27017/")
db = client['ganttDB-replica']

tasks = list(db.tasks.find())
resources = list(db.resources.find())
assignments = list(db.assignments.find())

df_tasks = pd.DataFrame(tasks)
df_resources = pd.DataFrame(resources)
df_assignments = pd.DataFrame(assignments)

df_assignments['task_id'] = df_assignments['event'].apply(lambda x: x.get('$oid') if isinstance(x, dict) else x)
df_assignments['resource_id'] = df_assignments['resource'].apply(lambda x: x.get('$oid') if isinstance(x, dict) else x)

merged = df_assignments.merge(df_tasks, left_on='task_id', right_on='_id', suffixes=('', '_task'))
merged = merged.merge(df_resources, left_on='resource_id', right_on='_id', suffixes=('', '_resource'))

df = merged[[
    'name', 'duration', 'businessUnit', 'profile',
    'requiredCompetences', 'requiredCertifications',
    'requiredFormations', 'resource_id'
]]

df.to_csv('data.csv', index=False)
print("[✔] Training data exported to data.csv")
