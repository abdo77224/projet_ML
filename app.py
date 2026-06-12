import streamlit as st
import pickle
import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features.molecular_descriptors import generate_features_df
from src.models.predict_model import predict_solubility

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# تحميل النموذج
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# الواجهة
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title('🧪 Molecular Solubility Predictor')
st.markdown('Predict aqueous solubility **(LogS)** of molecules from their SMILES representation.')
st.divider()

# Input
st.subheader('🔬 Enter Molecule')

example_smiles = {
    'Aspirin':       'CC(=O)OC1=CC=CC=C1C(=O)O',
    'Caffeine':      'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'Ibuprofen':     'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'Phenobarbital': 'CCC1(C(=O)NC(=O)NC1=O)c2ccccc2',
}

col1, col2 = st.columns([2, 1])

with col1:
    smiles_input = st.text_input('SMILES String:', 'CC(=O)OC1=CC=CC=C1C(=O)O')

with col2:
    example = st.selectbox('Or pick an example:', [''] + list(example_smiles.keys()))
    if example:
        smiles_input = example_smiles[example]

# Predict
if st.button('🚀 Predict Solubility', use_container_width=True):

    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error('❌ Invalid SMILES — please check your input!')
    else:
        result = predict_solubility(model, [smiles_input], scaler=scaler)
        logS   = result['Predicted_Solubility'].values[0]

        st.divider()
        st.subheader('📊 Results')

        if logS > 0:
            label = '🟢 Highly Soluble'
        elif logS > -2:
            label = '🟡 Soluble'
        elif logS > -4:
            label = '🟠 Moderately Soluble'
        else:
            label = '🔴 Poorly Soluble'

        col1, col2 = st.columns(2)

        with col1:
            st.metric('Predicted LogS', f'{logS:.3f}')
            st.markdown(f'**Solubility Class:** {label}')

        with col2:
            img = Draw.MolToImage(mol, size=(250, 200))
            st.image(img, caption='Molecule Structure')

        st.divider()
        st.subheader('🔍 Molecular Descriptors')
        features = generate_features_df([smiles_input]).drop('SMILES', axis=1)
        st.dataframe(features.T.rename(columns={0: 'Value'}).style.format('{:.3f}'))