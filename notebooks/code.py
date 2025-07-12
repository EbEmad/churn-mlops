import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

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


# ### train test split
from sklearn.model_selection import train_test_split
x=df.drop('Churn', axis=1)
y=df['Churn']
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)


# ### Data visulaization

import seaborn as sns
import matplotlib.pyplot as plt
# Churn Distribution
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

df['tenure'].unique()  # these values after standscalar

# Customer Tenure Vs Churn
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title("Customer Tenure vs. Churn")
plt.show()


# monthly Charges by Churn Status
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges by Churn Status")
plt.show()


# ### Correlatin HeatMap

numerical_cols= df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols


plt.figure(figsize=(10,8))
sns.heatmap(df[numerical_cols].corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# ### modeling step

def evaluate_model(model,x_test,y_test,model_name):
    y_pred=model.predict(x_test)
    y_proba=model.predict_proba(x_test)[:,1] if hasattr(model,'predict_proba') else None
    print(f"\n Evaluation for {model_name}")
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ### Logistic Regression

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
evaluate_model(lr,x_test,y_test,"Logistic Regression")


# ### Random Forest


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
evaluate_model(rf, x_test, y_test, "Random Forest")



# ### KNN
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
evaluate_model(knn, x_test, y_test, "KNN")


# ### Support Vector Classifier


svm = SVC(probability=True)
svm.fit(x_train,y_train)
evaluate_model(svm, x_test, y_test, "SVM")


# ### Drawing the ROC-AUC Curve for the trained models


models={
    "Logistic Regression": lr,
    "Random Forest": rf,
    "KNN": knn,
    "SVM": svm 
}


from sklearn.metrics import roc_curve, auc
best_model = None
best_score = 0.0
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(x_test)
    else:
        continue  # skip models that can't return probability or score

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.title('ROC Curve Comparison of Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()
# Save the model and vectorizer
joblib.dump(best_model, '/opt/airflow/models/best_model.pkl')






