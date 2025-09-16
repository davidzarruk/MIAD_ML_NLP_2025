#!/usr/bin/python  # Indica que este archivo puede ejecutarse como script con Python

import pandas as pd       # Importa pandas para manejo de datos en DataFrames
import joblib             # Importa joblib para cargar el modelo entrenado
import os                 # Importa os para manejar rutas de archivos



# Función principal que recibe una URL y devuelve la probabilidad de que sea phishing
def predict_phishing_proba(url):
    # Construye la ruta al archivo del modelo 'phishing_clf.pkl' basado en la ubicación actual del script
    model_path = os.path.join(os.path.dirname(__file__), 'phishing_clf.pkl')
    
    # Carga el modelo previamente entrenado
    clf = joblib.load(model_path)
    
    # Crea un DataFrame de una sola fila con la URL como texto
    url_ = pd.DataFrame([url], columns=['url'])

    # Extrae características binarizando si ciertas palabras clave están presentes en la URL
    keywords = ['https', 'login', '.php', '.html', '@', 'sign']
    for keyword in keywords:
        url_['keyword_' + keyword] = url_.url.str.contains(keyword).astype(int)

    # Calcula la longitud de la URL (menos 2 caracteres, no está claro por qué se resta 2)
    url_['lenght'] = url_.url.str.len() - 2

    # Separa la URL por '/' y toma la tercera parte (índice 2) que normalmente es el dominio
    split_url = url_.url.str.split('/', expand=True)
    if split_url.shape[1] > 2:
        domain = split_url.iloc[:, 2]  # Extrae el dominio si existe
    else:
        domain = pd.Series([""], index=url_.index)  # Si no hay suficiente estructura, usa string vacío

    # Calcula la longitud del dominio
    url_['lenght_domain'] = domain.str.len()

    # Detecta si el dominio parece ser una IP (todos dígitos)
    url_['isIP'] = url_.url.str.replace('.', '', regex=False).str.isnumeric().astype(int)

    # Cuenta cuántas veces aparece "com" en la URL
    url_['count_com'] = url_.url.str.count('com')

    # Elimina la columna original 'url' y deja solo las características para predicción
    X = url_.drop('url', axis=1)

    # Realiza la predicción y extrae la probabilidad de la clase positiva (phishing)
    p1 = clf.predict_proba(X)[0, 1]

    # Devuelve esa probabilidad
    return p1

# Bloque que permite ejecutar el archivo desde la terminal
if __name__ == "__main__":
    import sys  # Permite acceder a argumentos pasados por la terminal

    # Verifica que el usuario haya proporcionado una URL
    if len(sys.argv) < 2:
        print("Please provide a URL")  # Muestra un mensaje si no hay URL
    else:
        url = sys.argv[1]              # Toma la URL desde los argumentos del sistema
        p1 = predict_phishing_proba(url)        # Llama a la función para predecir la probabilidad
        print(url)                     # Muestra la URL ingresada
        print("Probability of Phishing:", p1)  # Muestra la probabilidad calculada
