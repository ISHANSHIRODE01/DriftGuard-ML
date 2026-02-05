import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

pipeline = joblib.load('model/model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

print("Pipeline steps:", pipeline.steps)
print("Preprocessor:", preprocessor)

# Load data
train_df = pd.read_csv('data/train.csv')
print("Train columns:", train_df.columns.tolist())

# Try a sub-transform
X_sample = train_df.drop(columns=['G3']).head(1)

# Hand-apply binary conversion to match
binary_features = ['schoolsup', 'famsup', 'paid', 'activities', 
                  'nursery', 'higher', 'internet', 'romantic']
for col in binary_features:
    X_sample[col] = (X_sample[col] == 'yes').astype(int)

print("X_sample snippet:\n", X_sample.iloc[:, :5])

try:
    transformed = preprocessor.transform(X_sample)
    print("Preprocessor transform success. Shape:", transformed.shape)
    
    pred = pipeline.predict(X_sample)
    print("Pipeline predict success. Value:", pred)
except Exception as e:
    print("Error during transform/predict:", e)
    import traceback
    traceback.print_exc()
