import streamlit as st
import pandas as pd

# Exemple de DataFrame
df = pd.DataFrame({
    'customer_id': [1001, 1002, 1003, 1004, 1005]
})

# Initialisation du session state
if 'client_id' not in st.session_state:
    st.session_state.client_id = ''

# Callback pour génération aléatoire
def generate_random_id():
    if not df.empty:
        st.session_state.client_id = str(df['customer_id'].sample().values[0])
    else:
        st.error("Aucun client disponible")

with st.form("client_search"):
    cols = st.columns([3, 1, 8])
    
    with cols[0]:
        # Liaison directe avec le session state via key
        st.text_input(
            "🔍 Identifiant Client",
            placeholder="Saisissez l'ID client...",
            key='client_id'  # Binding direct avec st.session_state.client_id
        )
        
    with cols[1]:
        search_button = st.form_submit_button("Rechercher", type="primary")
        
    with cols[2]:
        st.form_submit_button(
            "Génération aléatoire",
            type="secondary",
            on_click=generate_random_id
        )

# Logique de recherche
if search_button:
    st.write(f"Recherche lancée pour : {st.session_state.client_id}")
    # Ajoutez ici votre logique de recherche