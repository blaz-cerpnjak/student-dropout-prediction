import sys
sys.path.append("../../")
import pandas as pd
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
from evidently.tests import *

current_data = pd.read_csv('data/processed/current_data.csv')
reference_data = pd.read_csv('data/reference_dataset.csv')

report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=reference_data, current_data=current_data)
report.save_html('reports/index.html')

result = json.loads(report.json())

if result['metrics'][0]['result']['dataset_drift'] == 'false':
    print("Validation failed!")
    sys.exit(0)
else:
    print("Validation passed!")
    sys.exit(0)