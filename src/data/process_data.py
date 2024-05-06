import sys
sys.path.append("../../")
import pandas as pd

def process_data():
    # Convert excel files to csv
    df = pd.read_excel('data/raw/students.xlsx')
    df.to_csv('data/raw/students_csv.csv', index=False)

    # Open csv files
    students_df = pd.read_csv('data/raw/students_csv.csv')
    students_scores_df = pd.read_csv('data/raw/students_scores.csv')

    # Merge dataframes
    merged_df = students_df.merge(students_scores_df, left_index=True, right_index=True)

    # Add additional column
    merged_df['AVG_subject'] = merged_df[['STEM_subjects', 'H_subjects']].mean(axis=1)

    # Save processed data
    merged_df.to_csv('data/processed/current_data.csv', index=False)
    return

if __name__ == '__main__':
    process_data()