# Student Dropout Prediction

Predicting student dropout ("YES" or "NO") based on different factors like grades, family income, etc.

## Setup Poetry 
Install Poetry
```console
pip install poetry
```

Initialize Poetry
```console
poetry init
```

Install required dependencies with Poetry
```console
poetry add dvc
poetry add pandas
poetry add scikit-learn
poetry add flask
poetry add evidently
poetry add openpyxl
```

## Setup DVC ("Data Version Control")
Add data to .gitignore (because we added it to DVC)

Add data folder to DVC
```console
dvc init
dvc add data
dvc push
```

Add data.dvc to GIT
```console
git add data.dvc
git push
```

## Process Data
```python
python src/data/process_data.py
```

## Validate and Test Data
```python
python src/validation/my_validation.py
```
```python
python src/validation/evidently_test.py
```

## Train and Evaluate Model
```python
python src/models/train_eval.py
```

## Start Prediction API
```python
python src/serve/api.py
```

#### Test API:
```json
[{
    "Residence_city": "LOCAL",
    "Socioeconomic_level": 2,
    "Civil_status": "Single",
    "Age": 25,
    "State": "LOCAL",
    "Province": "LOCAL",
    "Vulnerable_group": 2,
    "Desired_program": "UNSPECIFIED",
    "Family_income": 1500000,
    "Father_level": "PRIMARY SCHOOL",
    "Mother_level": "UNDERGRADUATE",
    "Dropout": "NO",
    "STEM_subjects": 50.8,
    "H_subjects": 56.4,
    "AVG_subject": 53.6
}]
```

API response example:
```json
{
    "prediction": "NO"
}
```

![alt text](api_prediction.png)