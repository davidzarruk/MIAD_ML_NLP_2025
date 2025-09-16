#!/usr/bin/env python3
import os
import sys
import joblib

# 1) Definición de las columnas de salida
COLS = [
    'p_Action', 'p_Adventure', 'p_Animation', 'p_Biography',
    'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
    'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music',
    'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
    'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western'
]

# 2) Rutas absolutas a los artefactos (vectorizador y clasificador)
BASE_DIR  = os.path.dirname(__file__)
VECT_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
CLF_PATH  = os.path.join(BASE_DIR, 'clf_genres.pkl')

# 3) Carga única de los modelos al importar el módulo
try:
    vect = joblib.load(VECT_PATH)
    clf  = joblib.load(CLF_PATH)
except Exception as e:
    sys.exit(f"Error cargando modelos: {e}")

def predict_genre_proba(text:str)-> dict:
    """
    Recibe un texto (string) y devuelve un dict {género: probabilidad}.
    """
    docs = [text]
    X    = vect.transform(docs)           # transform, no fit_transform
    proba = clf.predict_proba(X)[0]       # array de forma (24,)
    return dict(zip(COLS, proba))
