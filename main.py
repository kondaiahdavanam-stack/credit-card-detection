import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,mean_squared_error
import joblib

data=pd.read_csv("credit_card_fraud.csv")
transaction_ids = data["transaction_id"]
data = data.drop("transaction_id", axis=1)

#Future Engeneering
data["high_amount"]=(data["amount"]>data["amount"].median()).astype(int)
data["night_transcation"]=((data["transaction_hour"]>=22) | (data["transaction_hour"]<=5)).astype(int)
data["high_velocity"]=(data["velocity_last_24h"]>5).astype(int)
data["geo_risk"]=(data["foreign_transaction"]+data["location_mismatch"])
data["low_trust_device"]=(data["device_trust_score"]<50).astype(int)


le=LabelEncoder()
data["merchant_category"]=le.fit_transform(data["merchant_category"])

scaler=StandardScaler()
col=["amount","transaction_hour","device_trust_score","velocity_last_24h","cardholder_age"]
data[col]=scaler.fit_transform(data[col])

x=data.drop("is_fraud",axis=1)
y=data["is_fraud"]

#splitting data
x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(x,y,transaction_ids,test_size=0.2,random_state=42,stratify=y)


#SMOTE
smote=SMOTE(random_state=42)
x_train_resampled,y_train_resampled=smote.fit_resample(x_train,y_train)

#xgboost
model=XGBClassifier(random_state=42,eval_metric='logloss')
model.fit(x_train_resampled,y_train_resampled)

y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]

accuracy=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

#probabilities
auc=roc_auc_score(y_test,y_prob)
rmse=np.sqrt(mean_squared_error(y_test,y_prob))

print("Accuracy :", accuracy)
print("F1 Score :", f1)
print("AUC-ROC  :", auc)
print("RMSE     :", rmse)

print("\n================ FRAUD ALERTS ================\n")

for i in range(len(y_prob)):

    if y_prob[i] >= 0.80:

        print("="*60)
        print("⚠ FRAUD ALERT")
        print("="*60)

        print(f"Transaction ID      : {id_test.iloc[i]}")
        print(f"Fraud Probability   : {y_prob[i]*100:.2f}%")

        print(f"Amount              : {x_test.iloc[i]['amount']:.2f}")
        print(f"Transaction Hour    : {x_test.iloc[i]['transaction_hour']:.2f}")
        print(f"Merchant Category   : {x_test.iloc[i]['merchant_category']}")
        print(f"Foreign Transaction : {x_test.iloc[i]['foreign_transaction']}")
        print(f"Location Mismatch   : {x_test.iloc[i]['location_mismatch']}")

        if y_prob[i] >= 0.90:
            print("Risk Level          : HIGH 🔴")
        elif y_prob[i] >= 0.70:
            print("Risk Level          : MEDIUM 🟠")
        else:
            print("Risk Level          : LOW 🟡")

        print("Status              : Fraudulent Transaction")
        print("="*60)

#xgboost model
joblib.dump(model, "fraud_model.pkl")

#standard scaler
joblib.dump(scaler, "scaler.pkl")

# Label Encoder
joblib.dump(le, "label_encoder.pkl")