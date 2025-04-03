import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
import os


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

# Chargement des données et scores de crédit (Données fictives)
@st.cache_data
def load_data():
    return pd.DataFrame({
        'id_transaction': [1, 2, 3, 4, 5, 6],
        'client_id': [1, 2, 1, 3, 2, 1],
        'montant': [250, 150, 400, 300, 500, 600],
        'date': ['2024-03-01', '2024-03-05', '2024-03-10', '2024-03-15', '2024-03-20', '2024-03-20'],
        'statut': ['Validé', 'En attente', 'Refusé', 'Validé', 'Validé', 'Validé']
    })

credit_scores = {
    1: 720,
    2: 650,
    3: 690
}

data = load_data()
dataTest = pd.read_csv("data/Test.csv")
dataTrain = pd.read_csv("data/Train.csv")



# Interface utilisateur
with st.container():
    col_title, _ = st.columns([9, 6])
    with col_title:
        st.markdown("# 💼 Portail Client - Analyse Risque credit")

# Formulaire de recherche
with st.form("client_search"):
    cols = st.columns([3, 1, 8])
    with cols[0]:
        client_id = st.text_input("🔍 Identifiant Client", placeholder="Saisissez l'ID client...")
    with cols[1]:
        st.form_submit_button("Rechercher", type="primary")
    with cols[2]:
        st.form_submit_button("Génération aléatoire", type="secondary")

if client_id:
    try:
        client_id = int(client_id)

        if "client_id" not in st.session_state or st.session_state.client_id != client_id:
            # Réinitialisation du state pour le nouveau client
            st.session_state.client_id = client_id
            st.session_state.total_amount_repay = None

        
        filtered_dataTest = dataTest[dataTest["customer_id"] == client_id].copy()
        filtered_dataTrain = dataTrain[dataTrain["customer_id"] == client_id].copy()

        if not filtered_dataTest.empty:
            #client_credit_score = credit_scores.get(client_id, 600)
            client_credit_score = 700  # Valeur par défaut pour le test
            plafond = client_credit_score * 10 

            first_transaction = filtered_dataTest.iloc[0]

            # Si le montant n'est pas encore initialisé, le définir
            if st.session_state.total_amount_repay is None:
                st.session_state.total_amount_repay = first_transaction['Total_Amount_to_Repay']

            # En-tête des métriques
            with st.container():
                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">SCORE CREDIT</div>
                            <div class="metric-value" style="color: var(--secondary-color);">{client_credit_score}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"""
                        <div class="styled-container">
                            <div class="metric-label">PLAFOND AUTORISÉ</div>
                            <div class="metric-value">{plafond} $</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Section principale
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                with st.container():
                    st.markdown("### 📝 Détails Transaction")
                    first_transaction = filtered_dataTest.iloc[0]
                    interestRate = int((first_transaction["Total_Amount_to_Repay"]-first_transaction["Total_Amount"])* 100 / (first_transaction["Total_Amount"]+1e-6) )
                    # Montant à rembourser
                    total_amount_repay = first_transaction["Total_Amount"]* (1 + interestRate / 100)

                    # Grid layout avec date modifiable
                    col_grid1, col_grid2 = st.columns(2)
                    
                    with col_grid1:
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
                    if "total_amount_repay" not in st.session_state:
                        st.session_state.total_amount_repay = first_transaction['Total_Amount']

                    with col_grid2:
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
                        
                        # Sélecteur de date
                        new_date = st.date_input(
                            "Date d'écheance :",
                            pd.to_datetime(first_transaction["due_date"]).date(),
                            key=f"date_{first_transaction["due_date"]}",
                            format="DD/MM/YYYY",
                            help="Modifier la date de la transaction"
                            )
                        # filtered_dataTest.loc[filtered_dataTest['id_transaction'] == first_transaction['ID'], 'date'] = new_date.strftime('%Y-%m-%d')

                    # Définition du callback qui met à jour le montant à rembourser
                    def update_total_amount_repay():
                        current_rate = st.session_state.interestRate  # Récupération de la valeur mise à jour du slider
                        st.session_state.total_amount_repay = first_transaction['Total_Amount'] * (1 + current_rate / 100)

                    # Contrôles interactifs (ajustement du montant et probabilité)
                    with st.container():
                        # Contrôle interactif : le slider pour le taux d'intérêt
                        # Slider avec le callback
                        interestRate = st.slider(
                            "💰 TAUX D'INTERET",
                            0,
                            70,
                            value=int((first_transaction["Total_Amount_to_Repay"] - first_transaction["Total_Amount"]) * 100 / (first_transaction["Total_Amount"] + 1e-6)),
                            key="interestRate",
                            on_change=update_total_amount_repay
                        )

                        # Bouton Evaluer positionné juste en dessous du slider
                        if st.button("Simulation", key="evaluer_button"):
                            # Calcul de la durée en jours entre la date de déblocage et la nouvelle date (due_date modifiée)
                            disbursement_date = datetime.strptime(first_transaction["disbursement_date"], "%Y-%m-%d")
                            # new_date est obtenu via le date_input précédemment
                            duration = (new_date - disbursement_date.date()).days
                            
                            # Construction du payload à envoyer à l'API
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
                            
                            # Appel à l'API
                            url = "https://AmedBah-CreditSoring.hf.space/predict"  # l'URL Space
                            load_dotenv()  # Charge les variables d'environnement du fichier .env
                            API_TOKEN = os.getenv("HF_API_TOKEN")

                            headers = {
                                "Authorization": f"Bearer {API_TOKEN}",
                                "Content-Type": "application/json"
                            }

                            try:
                                response = requests.post(url, headers=headers, json=payload)
                                if response.status_code == 200:
                                    res = response.json()
                                    # On attend un retour de la forme : {'prediction': 0, 'probability': 0.00059}
                                    probability = res.get("probability")
                                    if probability is not None:
                                        probabilite = probability * 100  # Conversion en pourcentage
                                        st.metric("Probabilité de défaut", f"{probabilite:.1f}%")
                                    else:
                                        st.error("Réponse de l'API invalide : clé 'probability' manquante")
                                else:
                                    st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
                            except Exception as e:
                                st.error(f"Exception lors de l'appel à l'API : {e}")

                    def get_client_data(client_id):
                        """Récupère les données du client à partir de son ID"""
                        return dataTest[dataTest["customer_id"] == client_id].iloc[0].to_dict()
                        
                    # Recalcul du montant à rembourser à chaque modification du slider
                    def update_total_amount_repay(initial_amount, rate):
                        st.session_state.total_amount_repay = initial_amount * (1 + rate / 100)
                        
                    
                    def update_total_amount_on_client_change():
                        current_client_data = get_client_data(client_id)
                        current_rate = st.session_state.get("interestRate", 0)
                        st.session_state.total_amount_repay = current_client_data["Total_Amount"] * (1 + current_rate / 100)
                        
                        

            with col2:
                with st.container():
                    st.markdown("### 📊 Analyse performance")
                    categories = ['Catégorie A', 'Catégorie B', 'Catégorie C', 'Catégorie D']
                    values = np.random.randint(10, 100, size=len(categories))

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=categories,
                        orientation='h',
                        marker_color=['#264653' if x < 50 else '#2A9D8F' for x in values]
                    ))

                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_showgrid=False,
                        yaxis_showgrid=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Section tableau historique
            with st.container():
                st.markdown("### 📅 Historique des Transactions")
                styled_dataTest = filtered_dataTrain.style \
                    .background_gradient(subset=["Total_Amount"], cmap='Blues') \
                    .format({'montant': "{:} €"})
                st.dataframe(styled_dataTest, use_container_width=True, height=400)

        else:
            st.warning("Aucune transaction trouvée pour ce client")

    except ValueError:
        st.error("Veuillez saisir un identifiant numérique valide")