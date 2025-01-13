import streamlit as st 
import numpy as np
import joblib

st.title("Prédiction du prix d'une voiture en fonction de ses caractéristiques")
st.subheader("Application réalisée par Nicolas")
st.markdown("Cette application utilise un modèle de Machine Learning")

# Chargement du modèle


import os

model_path = os.path.join(os.path.dirname(__file__), "final_model.joblib")
model = joblib.load(model_path)

#model = joblib.load(filename="final_model.joblib")

# Définission d'un fonction d'inférence

def inference(symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg):
    new_data = np.array([symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg])
    pred = model.predict(new_data.reshape(1,-1)) 
    return pred

# np.array([...]) crée un tableau unidimensionnel de forme (14,), ce qui représente une seule ligne 
# sans spécifier le nombre de "n_samples". Pour rendre ce tableau compatible avec le modèle, 
# on doit le convertir en une forme explicite de (1, 14) en utilisant .reshape(1, -1)

# L'utilisateur saisie les valeurs des caractéristiques de la voiture

symboling = st.number_input(label='symboling', min_value=0, value=3)
wheelbase = st.number_input("wheelbase", value=90)
carlength = st.number_input("carlength", value=150)
carwidth = st.number_input("carwidth", value=65)
carheight = st.number_input("carheight", value=50)
curbweight = st.number_input("curbweight", value=200)
enginesize = st.number_input("enginesize", value=120)
boreratio = st.number_input("boreratio", value=3.0)
stroke = st.number_input("stroke", value=3.0)
compressionratio = st.number_input("compressionratio", value=9.0)
horsepower = st.number_input("horsepower", value=110)
peakrpm = st.number_input("peakrpm", value=5000)
citympg = st.number_input("citympg", value=20)
highwaympg = st.number_input("highwaympg", value=30)

# Création du bouton "Predict" qui retourne la prédiction du modèle
if st.button("Predict"):
    prediction = inference(
        symboling, wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, 
        stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg
    )
    resultat = "Le prix de cette voiture ($) est de :" + str(round(prediction[0], 2))
    st.success(resultat)