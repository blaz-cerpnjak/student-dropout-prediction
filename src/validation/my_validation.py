import sys
sys.path.append("../../")
import pandas as pd

def validate_data(current_data, reference_data):
    """
    Checks if the current data has the same columns, column names and column types as the reference data
    """
    column_count_error = ''
    column_names_errors = []
    column_types_errors = []

    columns_current = current_data.columns
    columns_reference = reference_data.columns

    if len(columns_current) != len(columns_reference):
        column_count_error = f"Column count mismatch: {len(columns_current)} in current data and {len(columns_reference)} in reference data"

    for i in range(len(columns_current)):
        if columns_current[i] != columns_reference[i]:
            column_names_errors.append(columns_current[i])
        
    for i in range(len(current_data)):
        column_type = type(current_data.iloc[i, 0])
        if column_type != type(reference_data.iloc[i, 0]):
            column_types_errors.append(column_type)

    if column_count_error:
        print(column_count_error)

    if len(column_names_errors) > 0:
        print(f"Column names mismatch: {column_names_errors}")

    if len(column_types_errors) > 0:
        print(f"Column types mismatch: {column_types_errors}")

    if not column_count_error and len(column_names_errors) == 0 and len(column_types_errors) == 0:
        print("Data validation passed!")
    else:
        print("Data validation failed!")
        raise Exception("Data validation failed!")

if __name__ == '__main__':
    current_data = pd.read_csv('data/processed/current_data.csv')
    reference_data = pd.read_csv('data/reference_dataset.csv')
    validate_data(current_data=current_data, reference_data=reference_data)