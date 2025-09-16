# 🚀 Despliegue de API Flask en AWS EC2 (Phishing Detection)

Guía paso a paso para desplegar una API Flask que predice si una URL es phishing, utilizando `scikit-learn`, en una instancia EC2 de AWS. Incluye cómo mantener la API activa incluso después de cerrar la terminal.

---

## 🖥️ Paso 1: Crear e Iniciar Instancia EC2

1. **Lanzar instancia EC2:**

   * **SO:** Ubuntu Server 22.04 LTS
   * **Tipo de instancia:** t2.micro (Free Tier)

2. **Configuración de Seguridad:**

   * Agrega una **regla de entrada**:

     * Tipo: `Custom TCP`
     * Puerto: `5000`
     * Origen: `0.0.0.0/0` *(acceso público)*

---

## ⚙️ Paso 2: Configurar el Entorno y Clonar el Proyecto

Conéctate a la instancia EC2 vía SSH y ejecuta:

```bash
# Actualizar e instalar dependencias
sudo apt update
sudo apt install -y python3-pip python3-venv git tmux unzip

# Crear y activar entorno virtual
python3 -m venv myenv
source myenv/bin/activate

# Instalar dependencias
pip install flask flask-restx pandas joblib scikit-learn

# Clonar el repositorio
git clone https://github.com/sijaramillogo/MIAD_ML_NLP_2025.git
cd MIAD_ML_NLP_2025
```

---

## 🚀 Paso 3: Ejecutar la API Flask

```bash
source ~/myenv/bin/activate
cd ~/MIAD_ML_NLP_2025/model_deployment
python api.py
```

Verás un mensaje como:

```bash
Running on http://0.0.0.0:5000/
```

Accede a la API desde cualquier navegador:

```bash
http://<TU_IP_PUBLICA>:5000
```

---

## 🧱 Paso 4: Mantener la API Activa con `tmux`

1. Iniciar una sesión de `tmux`:

```bash
tmux new -s api_server
```

2. Dentro de `tmux`, activa el entorno y lanza la API:

```bash
source ~/myenv/bin/activate
cd ~/MIAD_ML_NLP_2025/model_deployment
python api.py
```

3. Desvincular la sesión sin detener la API:

Presiona `Ctrl + B`, luego suelta y presiona `D`.

4. Para reanudar la sesión luego:

```bash
tmux attach -t api_server
```

5. Para detener la API y cerrar `tmux`:

```bash
Ctrl + C
exit
```

---

## ✅ Paso 5: Comandos ÚTiles

* Activar entorno virtual:

  ```bash
  source ~/myenv/bin/activate
  ```

* Desactivar entorno virtual:

  ```bash
  deactivate
  ```

* Ver IP pública desde EC2:

  ```bash
  curl ifconfig.me
  ```

* Reanudar sesión de tmux:

  ```bash
  tmux attach -t api_server
  ```

* Salir de tmux sin cerrar la API:
  `Ctrl + B`, luego suelta y presiona `D`

* Cerrar sesión de tmux completamente:

  ```bash
  exit
  ```

---

## 🧩 Paso 6: Explicación del Código de la API

### 🔍 Inicio del Script

```python
#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_simon.prueba_despliegue import predict_phishing_proba
```

* `#!/usr/bin/python`: permite ejecutar el script como programa.
* `Flask`: framework para apps web.
* `flask_restx`: estructura y documentación Swagger para APIs.
* `joblib`: no se usa directamente aquí, puede eliminarse.
* `predict_phishing_proba`: función de predicción importada.

### ⚖️ Inicialización

```python
app = Flask(__name__)
api = Api(app, version='1.0', title='Phishing Prediction API', description='Phishing Prediction API')
```

### 🔹 Namespace

```python
ns = api.namespace('predict', description='Phishing Classifier')
```

### ⚙️ Parámetros de Entrada

```python
parser = api.parser()
parser.add_argument('URL', type=str, required=True, help='URL to be analyzed', location='args')
```

### 🔄 Modelo de Respuesta para Swagger

```python
resource_fields = api.model('Resource', {
    'result': fields.String,
})
```

### 🤖 Clase del Recurso y Método GET

```python
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        return {"result": predict_phishing_proba(args['URL'])}, 200
```

### 🚀 Ejecución del Servidor

```python
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
```

* `host='0.0.0.0'`: permite acceso desde cualquier IP.
* `port=5000`: expone la API en ese puerto.
* `debug=True`: solo para desarrollo, muestra errores detallados.
