import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def process_data():
    df=pd.read_csv('/opt/airflow/Data/churn-data.csv')
    # drop the id column
    df.drop('customerID',axis=1,inplace=True)

    # check for missing values which are not numbers in totalCharges column
    df[df['TotalCharges']==' ']

    # fill the missing values in Totalcharges with the median
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce') # convert it to numbers
    df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

    # check the missing values
    df[df['TotalCharges']==' ']

    # ### Encode the categorical features
    df.columns[df.dtypes=='object']

    # binary categorical features
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if df[col].dtype == 'object':
            df[col]=df[col].str.lower()
            df[col]=df[col].map({'yes':1,"no":0})

    # Encode gender
    if df['gender'].dtype == 'object':
        df['gender']=df['gender'].str.lower()
        df['gender']=df['gender'].map({'male':1,'female':0})
    # ecode the remaining categorical columns using one hot
    df=pd.get_dummies(
        df,columns=df.columns[df.dtypes=='object'] 
    )


    df.columns[df.dtypes=='object']
    # ### scale the numerical features
    scaler=StandardScaler() # for numerical values
    df[['tenure', 'MonthlyCharges', 'TotalCharges']]=scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    joblib.dump(scaler, '/opt/airflow/models/scaler.pkl')

    joblib.dump(df.drop('Churn', axis=1).columns.tolist(), '/opt/airflow/models/columns.pkl')
    return df

