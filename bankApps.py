import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import joblib
import shap
import matplotlib.pyplot as plt
from app import preprocess_input_data

# Configuration de la page
st.set_page_config(layout="wide", page_icon="💳", page_title="Portail Client - Analyse Risque")

# Configuration CSS personnalisée
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Open+Sans:wght@300;500&display=swap');
    :root {
        --primary-color: #264653;
        --secondary-color: #2A9D8F;
        --accent-color: #E9C46A;
        --background-color: #F8F9FA;
    }
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        color: var(--primary-color) !important;
    }
    .styled-container {
        border: 1px solid var(--primary-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        background: white;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    input {
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    /* Styles pour le datepicker */
    .stDateInput div[data-baseweb="input"] {
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .stDateInput svg {
        color: var(--primary-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Chargement des données et modèle

@st.cache_resource
def load_model():
    model_path = os.path.join("Model","model3.pkl")
    return joblib.load(model_path)

model1 = load_model()
dataTest = pd.read_csv("data/Test.csv")
dataTrain = pd.read_csv("data/Train.csv")

# Initialisation du session state
if 'client_id' not in st.session_state:
    st.session_state.client_id = ''
if 'total_amount_repay' not in st.session_state:
    st.session_state.total_amount_repay = None

# Callback pour génération aléatoire
def generate_random_id():
    if not dataTest.empty:
        st.session_state.client_id = str(dataTest['customer_id'].sample().values[0])
    else:
        st.error("Aucun client disponible")

# Interface utilisateur
with st.container():
    col_title, _ = st.columns([9, 6])
    with col_title:
        st.markdown("# 💼 Portail Client - Analyse Risque credit")

# Formulaire de recherche
with st.form("client_search"):
    cols = st.columns([3, 1, 8])
    
    with cols[0]:
        st.text_input(
            "🔍 Identifiant Client",
            placeholder="Saisissez l'ID client...",
            key='client_id'
        )
        
    with cols[1]:
        search_submitted = st.form_submit_button("Rechercher", type="primary")
        
    with cols[2]:
        st.form_submit_button(
            "Génération aléatoire",
            type="secondary",
            on_click=generate_random_id
        )

# Logique principale après soumission du formulaire
if search_submitted or st.session_state.client_id:
    try:
        current_client_id = int(st.session_state.client_id)
        
        # Vérification de changement de client
        if 'last_client_id' not in st.session_state:
            st.session_state.last_client_id = None
            
        if st.session_state.last_client_id != current_client_id:
            st.session_state.last_client_id = current_client_id
            st.session_state.total_amount_repay = None  # Réinitialisation
            
        filtered_dataTest = dataTest[dataTest["customer_id"] == current_client_id].copy()
        filtered_dataTrain = dataTrain[dataTrain["customer_id"] == current_client_id].copy()
        
        if not filtered_dataTest.empty:
            first_transaction = filtered_dataTest.iloc[0]
            client_credit_score = 700  # Valeur par défaut pour le test
            plafond = client_credit_score * 10
            
            # Si le montant n'est pas initialisé, le définir
            if st.session_state.total_amount_repay is None:
                st.session_state.total_amount_repay = first_transaction['Total_Amount_to_Repay']

            # En-tête des métriques
            with st.container():
                # Calcul des indicateurs
                nombre_pret = len(filtered_dataTrain)
                nombre_defaut = filtered_dataTrain['target'].sum() if not filtered_dataTrain.empty else 0
                montant_max = filtered_dataTrain['Total_Amount'].max() if not filtered_dataTrain.empty else 0
                
                # Première rangée de métriques
                row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
                with row1_col1:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">SCORE CREDIT</div>
                            <div class="metric-value" style="color: var(--secondary-color);">{client_credit_score}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row1_col2:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">NOMBRE DE PRÊTS</div>
                            <div class="metric-value">{nombre_pret}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with row1_col3:    
                    st.markdown(f"""
                            <div class="styled-container">
                                <div class="metric-label">NOMBRE DE DÉFAUTS</div>
                                <div class="metric-value" style="color: {'#E76F51' if nombre_defaut > 0 else '#2A9D8F'};">
                                    {nombre_defaut}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                with row1_col4:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">MONTANT MAX</div>
                            <div class="metric-value">{montant_max:.2f} $</div>
                        </div>
                    """, unsafe_allow_html=True)

                

            # Section principale
            main_col1, main_col2 = st.columns([1, 1], gap="large")
            
            with main_col1:
                st.markdown("### 📝 Détails Transaction")
                interestRate = int((first_transaction["Total_Amount_to_Repay"] - first_transaction["Total_Amount"]) * 100 / (first_transaction["Total_Amount"] + 1e-6))
                
                # Grid layout
                grid_col1, grid_col2 = st.columns(2)
                with grid_col1:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">ID TRANSACTION</div>
                            <div class="metric-value">{first_transaction['ID']}</div>
                        </div>
                        <div class="styled-container">
                            <div class="metric-label">MONTANT INITIAL</div>
                            <div class="metric-value">{first_transaction['Total_Amount']} $</div>
                        </div>
                        <div class="styled-container">
                            <div class="metric-label">DATE DE CONTRACTION</div>
                            <div class="metric-value">{first_transaction["disbursement_date"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with grid_col2:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">STATUT</div>
                            <div class="metric-value" style="color: {'#2A9D8F' if first_transaction['loan_type'] == "Repeat Loan" else '#E76F51'};">{first_transaction['loan_type']}</div>
                        </div>
                        <div class="styled-container">
                            <div class="metric-label">MONTANT A REMBOURSER</div>
                            <div class="metric-value">{st.session_state.total_amount_repay:.2f} $</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    new_date = st.date_input(
                        "Date d'écheance :",
                        pd.to_datetime(first_transaction["due_date"]).date(),
                        key=f"date_{first_transaction['due_date']}",
                        format="DD/MM/YYYY",
                        help="Modifier la date de la transaction"
                    )

                # Callback pour le slider
                def update_total_amount():
                    initial = first_transaction['Total_Amount']
                    rate = st.session_state.interestRate
                    st.session_state.total_amount_repay = initial * (1 + rate / 100)

                # Contrôles interactifs
                interestRate = st.slider(
                    "💰 TAUX D'INTERET",
                    0,
                    70,
                    value=interestRate,
                    key="interestRate",
                    on_change=update_total_amount
                )

                if st.button("Simulation", key="evaluer_button"):
                    disbursement_date = datetime.strptime(first_transaction["disbursement_date"], "%Y-%m-%d")
                    duration = (new_date - disbursement_date.date()).days

                    payload = {
                        "customer_id": int(first_transaction["customer_id"]),
                        "tbl_loan_id": int(first_transaction["tbl_loan_id"]),
                        "lender_id": int(first_transaction["lender_id"]),
                        "Total_Amount": float(first_transaction["Total_Amount"]),
                        "Total_Amount_to_Repay": float(st.session_state.total_amount_repay),
                        "duration": duration,
                        "New_versus_Repeat": first_transaction["New_versus_Repeat"],
                        "Amount_Funded_By_Lender": float(first_transaction["Amount_Funded_By_Lender"]),
                        "Lender_portion_Funded": float(first_transaction["Lender_portion_Funded"]),
                        "Lender_portion_to_be_repaid": float(st.session_state.total_amount_repay) * float(first_transaction["Lender_portion_Funded"]),
                        "disbursement_date": first_transaction["disbursement_date"],
                        "due_date": new_date.strftime("%Y-%m-%d")
                    }

                    load_dotenv()
                    API_TOKEN = os.getenv("HF_API_TOKEN")
                    headers = {
                        "Authorization": f"Bearer {API_TOKEN}",
                        "Content-Type": "application/json"
                    }

                    try:
                        response = requests.post("https://AmedBah-CreditSoring.hf.space/predict", headers=headers, json=payload)
                        if response.status_code == 200:
                            res = response.json()
                            probability = res.get("probability", 0) * 100
                            st.metric("Probabilité de défaut", f"{probability:.1f}%")
                        else:
                            st.error(f"Erreur API : {response.status_code}")
                    except Exception as e:
                        st.error(f"Exception : {e}")

                    # Calcul SHAP
                    with st.spinner("Calcul SHAP en cours..."):
                        input_data = pd.DataFrame([payload])
                        processed_data = preprocess_input_data(input_data)
                        explainer = shap.Explainer(model1)
                        shap_values = explainer(processed_data)
                        st.session_state.shap_values = shap_values

            with main_col2:
                st.markdown("### Explication SHAP")
                if "shap_values" in st.session_state:
                    fig, ax = plt.subplots(figsize=(14, 9))
                    shap.plots.waterfall(st.session_state.shap_values[0], show=False)
                    st.pyplot(fig)
                else:
                    st.info("Cliquez sur 'Simuler' pour afficher l'analyse SHAP")

            # Historique des transactions
            st.markdown("### 📅 Historique des Transactions")
            styled_data = filtered_dataTrain.style \
                .background_gradient(subset=["Total_Amount"], cmap='Blues') \
                .format({'Total_Amount': "{:,.2f} €"})
            st.dataframe(styled_data, use_container_width=True, height=400)

        else:
            st.warning("Aucune transaction trouvée pour ce client")

    except ValueError:
        st.error("Veuillez saisir un identifiant client numérique valide")