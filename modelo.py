import os
import tempfile
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
import joblib
from io import BytesIO
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, 'DEMALE-HSJM_2025_data.xlsx')
MODEL_DIR = os.path.join(APP_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ['male', 'female', 'age', 'body_temperature', 'hemoglobin',
            'headache', 'vomiting', 'platelets']
TARGET = 'diagnosis'

LOG_PATH = os.path.join(MODEL_DIR, 'logistic.pkl')
NN_PATH = os.path.join(MODEL_DIR, 'nn.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
CLASS_MAP_PATH = os.path.join(MODEL_DIR, 'class_map.pkl')

app = Flask(__name__, static_folder='static', template_folder='templates')


USER_CODE_TO_NAME = {
    '1': 'Dengue',
    '2': 'Malaria',
    '3': 'Leptospirosis'
}

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")
    return pd.read_excel(DATA_PATH, engine='openpyxl')

def prepare_dataframe(df):
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    df = df[~df[TARGET].isnull()]

    for col in ['headache', 'vomiting']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({
                'yes': 1, 'Yes': 1, 'Si': 1, 'SÍ': 1, 'Sí': 1, 'sí': 1,
                'no': 0, 'No': 0, 'nan': np.nan, 'None': np.nan
            })
            df[col] = pd.to_numeric(df[col], errors='coerce')

   
    num_cols = ['age', 'body_temperature', 'hemoglobin', 'platelets']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['male'] = pd.to_numeric(df.get('male', 0), errors='coerce').fillna(0).astype(int)
    df['female'] = pd.to_numeric(df.get('female', 0), errors='coerce').fillna(0).astype(int)

    return df

def train_and_save_models(force_retrain=False):
    if (os.path.exists(LOG_PATH) and os.path.exists(NN_PATH)
            and os.path.exists(SCALER_PATH) and os.path.exists(CLASS_MAP_PATH)
            and not force_retrain):
        print("Modelos cargados desde disco.")
        log = joblib.load(LOG_PATH)
        nn = joblib.load(NN_PATH)
        scaler = joblib.load(SCALER_PATH)
        class_info = joblib.load(CLASS_MAP_PATH)
        return log, nn, scaler, class_info

    print("Entrenando modelos (esto puede tardar)...")
    df = prepare_dataframe(load_data())

    X = df[FEATURES].copy()
    y_raw = df[TARGET].astype(str).str.strip()  

    for col in X.columns:
        if X[col].dtype.kind in 'biufc':
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

   
    le_classes = le.classes_.tolist()  
    class_map = {}
    for cls in le_classes:
       
        if str(cls) in USER_CODE_TO_NAME:
            class_map[str(cls)] = USER_CODE_TO_NAME[str(cls)]
        else:
           
            class_map[str(cls)] = str(cls)

  
    stratify_arg = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=stratify_arg)


    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)

    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    nn.fit(X_train, y_train)


    joblib.dump(log, LOG_PATH)
    joblib.dump(nn, NN_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({'le_classes': le_classes, 'class_map': class_map}, CLASS_MAP_PATH)

    print("Entrenamiento finalizado.")
    return log, nn, scaler, {'le_classes': le_classes, 'class_map': class_map}


model_log, model_nn, model_scaler, class_info = train_and_save_models()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lotes')
def lotes():
    return render_template('lotes.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'No data received'}), 400

   
    sex = payload.get('sex', '').lower()
    male = 1 if sex in ['male', 'masculino', 'm'] else 0
    female = 1 if sex in ['female', 'femenino', 'f'] else 0

    def parse_bool(v):
        if v is None:
            return 0
        s = str(v).strip().lower()
        return 1 if s in ['sí', 'si', 'yes', 'y', '1', 'true'] else 0

    try:
        row = pd.DataFrame([{
            'male': male,
            'female': female,
            'age': float(payload.get('age', 0)),
            'body_temperature': float(payload.get('body_temperature', 37.0)),
            'hemoglobin': float(payload.get('hemoglobin', 14.0)),
            'headache': parse_bool(payload.get('headache')),
            'vomiting': parse_bool(payload.get('nausea')), 
            'platelets': float(payload.get('platelets', 250000))
        }])
    except Exception as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    df_orig = load_data()
    medians = {}
    for c in ['age', 'body_temperature', 'hemoglobin', 'platelets']:
        medians[c] = df_orig[c].median() if c in df_orig.columns else 0
    for c in ['age', 'body_temperature', 'hemoglobin', 'platelets']:
        if pd.isna(row.at[0, c]):
            row.at[0, c] = medians[c]

    X_row = model_scaler.transform(row[FEATURES])

    model_choice = payload.get('model', 'logistica')
    if model_choice in ['nn', 'red', 'red_neuronal']:
        probs = model_nn.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(probs))
        confidence = round(float(probs[pred_idx]) * 100, 2)
        model_name = 'Red Neuronal'
    else:
        probs = model_log.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(probs))
        confidence = round(float(probs[pred_idx]) * 100, 2)
        model_name = 'Regresión Logística'

    le_classes = class_info.get('le_classes', [])
    class_map = class_info.get('class_map', {})

    pred_label_original = le_classes[pred_idx] if pred_idx < len(le_classes) else str(pred_idx)

    pred_readable = class_map.get(str(pred_label_original), str(pred_label_original))

    return jsonify({
        'model': model_name,
        'diagnosis': str(pred_readable),
        'confidence': confidence
    })
    
# 🧩 Ruta 1: subir dataset y listar columnas
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    file = request.files.get('file')
    if not file:
        return render_template('lotes.html', error="No se ha subido ningún archivo")

    # Guardar temporalmente
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    # Leer el dataset
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return render_template('lotes.html', error=f"Error al leer el archivo: {e}")

    columns = df.columns.tolist()
    return render_template('lotes.html', columns=columns, file_path=file_path)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE   


@app.route('/train_batch', methods=['POST'])
def train_batch():
    file_path = request.form.get('file_path')
    target = request.form.get('target')
    model_type = request.form.get('model_type')

    if not file_path or not os.path.exists(file_path):
        return render_template('lotes.html', error="No se encontró el dataset")

    # Leer dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if target not in df.columns:
        return render_template('lotes.html', error="La columna objetivo no existe")

    # Separar variables
    X = df.drop(columns=[target])
    y = df[target]

    # Codificar variables no numéricas
    X = pd.get_dummies(X)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # 🔹 Aplicar balanceo de clases con SMOTE
    try:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print("✅ Dataset balanceado con SMOTE")
    except Exception as e:
        print(f"⚠️ No se pudo aplicar SMOTE: {e}")

    # 🔹 Seleccionar modelo
    if model_type == "logistic":
        base_model = LogisticRegression(max_iter=1000)
        model_name = "Regresión Logística"
    elif model_type == "nn":
        base_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=300)
        model_name = "Red Neuronal"
    else:
        return render_template('lotes.html', error="Modelo no reconocido")

    # Crear pipeline con estandarización
    pipeline = make_pipeline(StandardScaler(), base_model)

    # 🔹 Cross-validation para obtener predicciones “out-of-fold”
    y_pred = cross_val_predict(pipeline, X, y, cv=5)

    # 📊 Matriz de confusión y reporte
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    # 🖼️ Graficar la matriz
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.colorbar(im, ax=ax)

    # Etiquetas en las celdas
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # Convertir la figura a base64
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    return render_template(
        'lotes.html',
        cm_img=plot_url,
        report=report,
        model_name=model_name,
        n_samples=len(y)
    )



@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    import matplotlib.pyplot as plt
    import io, base64
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

    file = request.files.get('file')
    model_choice = request.form.get('model', 'logistica')

    if not file:
        return render_template('lotes.html', error="No se ha subido ningún archivo")

    # 📥 Leer el archivo (CSV o Excel)
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return render_template('lotes.html', error="Formato no soportado. Usa .csv o .xlsx")

    # 📋 Verificar columnas requeridas
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        return render_template(
            'lotes.html',
            error=f"Faltan las siguientes columnas en el dataset: {', '.join(missing_cols)}"
        )

    # 🧹 Preprocesar los datos
    df = prepare_dataframe(df)
    X = df[FEATURES].copy()
    y_true = df[TARGET].astype(str).str.strip()

    # Escalar los datos
    X_scaled = model_scaler.transform(X)

    le_classes = class_info.get('le_classes', [])
    class_map = class_info.get('class_map', {})

    # 🧠 Selección del modelo
    if model_choice in ['nn', 'red', 'red_neuronal']:
        y_pred = model_nn.predict(X_scaled)
        model_name = 'Red Neuronal'
    else:
        y_pred = model_log.predict(X_scaled)
        model_name = 'Regresión Logística'

    # Convertir índices predichos a etiquetas (por si el modelo entrega índices)
    y_pred_labels = [le_classes[i] if i < len(le_classes) else i for i in y_pred]

    # 📊 Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_labels, labels=le_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_classes)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 📈 Exactitud del modelo
    try:
        accuracy = round(accuracy_score(y_true, y_pred_labels) * 100, 2)
    except Exception:
        accuracy = None

    return render_template(
        'lotes.html',
        cm=img_base64,
        model_name=model_name,
        accuracy=accuracy
    )


from flask import send_file

@app.route('/download_pdf')
def download_pdf():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Archivo no encontrado", 404



if __name__ == '__main__':
    print("Servidor iniciado en http://127.0.0.1:5000")
    app.run(debug=True)
