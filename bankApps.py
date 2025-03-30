import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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

if client_id:
    try:
        client_id = int(client_id)
        filtered_data = data[data['client_id'] == client_id].copy()

        if not filtered_data.empty:
            client_credit_score = credit_scores.get(client_id, 600)
            plafond = client_credit_score * 10

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
                            <div class="metric-value">{plafond} €</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Section principale
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                with st.container():
                    st.markdown("### 📝 Détails Transaction")
                    first_transaction = filtered_data.iloc[0]

                    # Grid layout avec date modifiable
                    col_grid1, col_grid2 = st.columns(2)
                    
                    with col_grid1:
                        st.markdown(f"""
                            <div class="styled-container">
                                <div class="metric-label">ID TRANSACTION</div>
                                <div class="metric-value">{first_transaction['id_transaction']}</div>
                            </div>
                            <div class="styled-container">
                                <div class="metric-label">MONTANT INITIAL</div>
                                <div class="metric-value">{first_transaction['montant']} €</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_grid2:
                        st.markdown(f"""
                            <div class="styled-container">
                                <div class="metric-label">STATUT</div>
                                <div class="metric-value" style="color: {'#2A9D8F' if first_transaction['statut'] == 'Validé' else '#E76F51'};">{first_transaction['statut']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Sélecteur de date
                        new_date = st.date_input(
                            "Date transaction :",
                            pd.to_datetime(first_transaction['date']).date(),
                            key=f"date_{first_transaction['id_transaction']}",
                            format="DD/MM/YYYY",
                            help="Modifier la date de la transaction"
                            )
                        filtered_data.loc[filtered_data['id_transaction'] == first_transaction['id_transaction'], 'date'] = new_date.strftime('%Y-%m-%d')

                    # Contrôles interactifs (ajustement du montant et probabilité)
                    with st.container():
                        montant_saisi = st.slider("💰 Ajustement du montant", 
                                                0, 
                                                plafond, 
                                                first_transaction['montant'])
                        probabilite = (1 - (montant_saisi / plafond)) * 100
                        st.metric("Probabilité de Remboursement",
                                f"{probabilite:.1f}%",
                                delta_color="off",
                                help="Probabilité estimée basée sur le ratio montant/plafond")

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
                styled_df = filtered_data.style \
                    .background_gradient(subset=['montant'], cmap='Blues') \
                    .format({'montant': "{:} €"})
                st.dataframe(styled_df, use_container_width=True, height=400)

        else:
            st.warning("Aucune transaction trouvée pour ce client")

    except ValueError:
        st.error("Veuillez saisir un identifiant numérique valide")