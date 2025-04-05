from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
from scipy.stats import mode


# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle XGBoost pré-entraîné

model_path1 = os.path.join("Model","model1.pkl")
model_path2 = os.path.join("Model","model2.pkl")
model_path3 = os.path.join("Model","model3.pkl")

xgb_model = joblib.load(model_path1)
lgb_model = joblib.load(model_path2 )
cat_model = joblib.load(model_path3)

# Définir la fonction pour calculer les nouvelles variables
def preprocess_input_data(input_data):
    """
    Prétraiter les données d'entrée pour correspondre aux caractéristiques utilisées dans le modèle.
    """
    # Convertir les dates en objets datetime
    input_data['disbursement_date'] = pd.to_datetime(input_data['disbursement_date'], errors='coerce')
    input_data['due_date'] = pd.to_datetime(input_data['due_date'], errors='coerce')

    # Calculer les nouvelles variables temporelles
    input_data['loan_term_days'] = (input_data['due_date'] - input_data['disbursement_date']).dt.days
    input_data['disbursement_weekday'] = input_data['disbursement_date'].dt.weekday
    input_data['due_weekday'] = input_data['due_date'].dt.weekday
    input_data['disbursement_date_month'] = input_data['disbursement_date'].dt.month
    input_data['disbursement_date_day'] = input_data['disbursement_date'].dt.day
    input_data['disbursement_date_year'] = input_data['disbursement_date'].dt.year
    input_data['due_date_month'] = input_data['due_date'].dt.month
    input_data['due_date_day'] = input_data['due_date'].dt.day
    input_data['due_date_year'] = input_data['due_date'].dt.year

    # Calculer les ratios financiers
    input_data['repayment_ratio'] = input_data['Total_Amount_to_Repay'] / input_data['Total_Amount']
    input_data['amount_due_per_day'] = input_data['Total_Amount_to_Repay'] / input_data['duration']

    # Appliquer des transformations logarithmiques
    input_data['log_Total_Amount'] = np.log1p(input_data['Total_Amount'])
    input_data['log_Total_Amount_to_Repay'] = np.log1p(input_data['Total_Amount_to_Repay'])
    input_data['log_Amount_Funded_By_Lender'] = np.log1p(input_data['Amount_Funded_By_Lender'])
    input_data['log_Lender_portion_to_be_repaid'] = np.log1p(input_data['Lender_portion_to_be_repaid'])

    # Calculer si le montant à rembourser est supérieur à la moyenne
    aggregates = input_data.groupby('customer_id')['Total_Amount_to_Repay'].agg(['mean', 'median']).reset_index()
    aggregates.rename(columns={'mean': 'Mean_Total_Amount', 'median': 'Median_Total_Amount'}, inplace=True)
    input_data = input_data.merge(aggregates, on='customer_id', how='left')
    input_data['amount_to_repay_greater_than_average'] = input_data['Mean_Total_Amount'] - input_data['Total_Amount_to_Repay']

    # Gérer les valeurs aberrantes
    q = 0.9
    input_data['Total_Amount_to_Repay'] = np.where(
        input_data['Total_Amount_to_Repay'] >= input_data['Total_Amount_to_Repay'].quantile(q),
        input_data['Total_Amount_to_Repay'].quantile(q),
        input_data['Total_Amount_to_Repay']
    )
    input_data['Total_Amount'] = np.where(
        input_data['Total_Amount'] >= input_data['Total_Amount'].quantile(q),
        input_data['Total_Amount'].quantile(q),
        input_data['Total_Amount']
    )

    # Define the mapping
    mapping = {
        "Repeat Loan": 0,
        "New Loan": 1
    }

    # Apply the mapping to the column
    input_data['New_versus_Repeat'] = input_data['New_versus_Repeat'].map(mapping)

    # Retourner uniquement les caractéristiques nécessaires pour le modèle
    features_for_modelling = [
        'customer_id', 'tbl_loan_id', 'lender_id', 'Total_Amount', 'Total_Amount_to_Repay', 'duration',
        'New_versus_Repeat', 'Amount_Funded_By_Lender', 'Lender_portion_Funded', 'Lender_portion_to_be_repaid',
        'Mean_Total_Amount', 'Median_Total_Amount', 'disbursement_date_month', 'disbursement_date_day',
        'disbursement_date_year', 'loan_term_days', 'disbursement_weekday', 'due_weekday', 'due_date_month',
        'due_date_day', 'due_date_year', 'repayment_ratio', 'amount_due_per_day', 'log_Total_Amount',
        'log_Total_Amount_to_Repay', 'log_Amount_Funded_By_Lender', 'log_Lender_portion_to_be_repaid',
        'amount_to_repay_greater_than_average'
    ]
    return input_data[features_for_modelling]


@app.route("/", methods=["GET"])
def accueil():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de Credit scoring"})


# Définir la route pour l'API
@app.route('/predict', methods=['POST'])

def predict():
    
    try:
        input_json = request.json
        input_data = pd.DataFrame([input_json])
        processed_data = preprocess_input_data(input_data)
        
        # Prédictions des trois modèles
        pred_xgb = xgb_model.predict(processed_data)
        pred_lgb = lgb_model.predict(processed_data)
        pred_cat = cat_model.predict(processed_data)
        
        prob_xgb = xgb_model.predict_proba(processed_data)[:, 1]
        prob_lgb = lgb_model.predict_proba(processed_data)[:, 1]
        prob_cat = cat_model.predict_proba(processed_data)[:, 1]
        
        predictions=[]
        pred_prob=[]
        #Append predictions
        predictions.append(pred_xgb)
        predictions.append(pred_lgb)
        predictions.append(pred_cat)
        pred_prob.append(prob_xgb)
        pred_prob.append(prob_lgb)
        pred_prob.append(prob_cat)
        
        # Agrégation des prédictions (vote majoritaire)
        
        final_prediction = mode(predictions, axis=0).mode.flatten()
        
        # Moyenne des probabilités
        final_probability = np.mean(pred_prob,axis=0)
        
        result = {
            'prediction': int(final_prediction[0]),
            'probability': float(final_probability[0])
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Lancer l'application Flask

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=7860)