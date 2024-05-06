import sys
sys.path.append("../../")
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train():
    # Open processed data
    df = pd.read_csv('data/processed/current_data.csv')
    print(df.head())

    # Drop unnecessary columns
    df.drop('Unnamed: 0_x', axis=1, inplace=True)
    df.drop('Unnamed: 0_y', axis=1, inplace=True)

    # Save random row (for testing the prediction api)
    random_row = df.sample()
    random_row.to_json('src/serve/random_row.json', orient='records')
    print("Random row:\n", random_row)
    df = df.drop(random_row.index)

    # Define features and target
    X = df[['Socioeconomic_level', 'Age', 'Vulnerable_group', 'Family_income', 'STEM_subjects', 
            'H_subjects', 'AVG_subject', 'Residence_city', 'Civil_status', 'State', 'Province',
            'Desired_program', 'Father_level', 'Mother_level']]
    y = df['Dropout']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessor and classifier
    numeric_features = ['Socioeconomic_level', 'Age', 'Vulnerable_group', 'Family_income', 'STEM_subjects',
                        'H_subjects', 'AVG_subject']
    categorical_features = ['Residence_city', 'Civil_status', 'State', 'Province', 'Desired_program',
                            'Father_level', 'Mother_level']

    # Fill missing values and scale numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Fill missing values and one-hot encode categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    classifier = MLPClassifier(max_iter=1000, random_state=1234)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save model
    joblib.dump(pipeline, 'models/model.joblib')

    # Evaluate model
    predictions = pipeline.predict(X_test)
    print("Predictions:\n", predictions)

    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)
    
    return

if __name__ == "__main__":
    train()