# ------------------------- DATA COLLECTION --------------------------------------------------------------------------------
import pandas as pd

# ------------------------- Load the dataset --------------------------
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ------------------------- Explore the first few rows ----------------
print(df.head())

# ------------------------- Check the columns and their data types ---
print(df.info())


# ------------------------- PREPROCESSING WHOLE PROCESS --------------------------------------------------------------------

# ------------------------- 1. Remove Unnecessary Columns ---------------
df = df.drop(columns=['customerID'])

print(df.isnull().sum())  # Check for missing values

# ------------------------- 2. Handle Missing Values ---------------------
# Fill missing numerical values with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)


# ------------------------- 3. Label Encoding -----------------------------
from sklearn.preprocessing import LabelEncoder

# Columns with binary categories
binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                  'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

le = LabelEncoder()
for col in binary_columns:
    df[col] = le.fit_transform(df[col])

# You can also convert other columns like 'InternetService', 'Contract', 'PaymentMethod' to one-hot encoded.
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])


# ------------------------- 4. Feature Scaling and Normlizing --------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])


# ------------------------- 5. Handle Class Imbalance(SMOTE) ---------------
from imblearn.over_sampling import SMOTE

X = df.drop(columns=['Churn'])
y = df['Churn']

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)


# ------------------------- 6. save Preprocessed data -----------------------
df.to_csv('data/cleaned_telco_churn.csv', index=False)