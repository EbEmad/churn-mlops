import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_data import process_data
def train_model():
    df = process_data()
    # ### train test split
    x=df.drop('Churn', axis=1)
    y=df['Churn']
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=42,stratify=y
    )


    # ### Data visulaization
    # replace that by using mlflow


    # ### modeling step
    # use mlflow to help you


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


