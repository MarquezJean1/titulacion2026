from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from collections import Counter
import math
import re
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import send_file
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'gff3', 'gff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# CARGAR MODELO, SCALER E INFO (XGBOOST)
# ============================================================================
try:
    print("iniciando carga...")
    with open('modelo_xgboost.pkl', 'rb') as archivo:
        modelo = pickle.load(archivo)
    print("Modelo cargado")
    with open('scaler.pkl', 'rb') as archivo:
        scaler = pickle.load(archivo)
    print("Scaler cargado")
    with open('info_modelo_xgboost.pkl', 'rb') as archivo:
        info_modelo = pickle.load(archivo)
    print("Info del modelo cargado")
    print("✅ Modelo XGBoost, scaler e info_modelo cargados exitosamente")

    # (Opcional) imprimir hiperparámetros en consola al iniciar
    if isinstance(info_modelo, dict) and "hiperparametros" in info_modelo:
        print("\n📌 Hiperparámetros XGBoost cargados:")
        for k, v in info_modelo["hiperparametros"].items():
            print(f" - {k}: {v}")

except Exception as e:
    print(f"⚠️  Advertencia: No se pudieron cargar los PKL: {e}")
    modelo = None
    scaler = None
    info_modelo = None

# ============================================================================
# FUNCIONES DE EXTRACCIÓN (del código original)
# ============================================================================

def tiene_ambiguedades(secuencia):
    """Detecta si una secuencia contiene códigos IUPAC de ambigüedad."""
    bases_validas = set('ATCG')
    codigos_iupac = set('WRYSKMNBDHV')
    bases_ambiguas = set()
    for base in secuencia.upper():
        if base not in bases_validas and base in codigos_iupac:
            bases_ambiguas.add(base)
    return len(bases_ambiguas) > 0, bases_ambiguas

def calcular_frecuencias_nucleotidos(secuencia):
    total = len(secuencia)
    if total == 0:
        return 0, 0, 0, 0
    contador = Counter(secuencia.upper())
    return (
        round((contador['A'] / total) * 100, 2),
        round((contador['T'] / total) * 100, 2),
        round((contador['C'] / total) * 100, 2),
        round((contador['G'] / total) * 100, 2)
    )

def calcular_contenido_gc(secuencia):
    secuencia = secuencia.upper()
    gc = secuencia.count('G') + secuencia.count('C')
    return round((gc / len(secuencia)) * 100, 2) if len(secuencia) > 0 else 0

def calcular_gc_skew(secuencia):
    secuencia = secuencia.upper()
    g = secuencia.count('G')
    c = secuencia.count('C')
    resultado = (g - c) / (g + c) if (g + c) > 0 else 0
    return round(resultado, 4)

def calcular_at_skew(secuencia):
    secuencia = secuencia.upper()
    a = secuencia.count('A')
    t = secuencia.count('T')
    resultado = (a - t) / (a + t) if (a + t) > 0 else 0
    return round(resultado, 4)

def calcular_entropia_shannon(secuencia):
    secuencia = secuencia.upper()
    contador = Counter(secuencia)
    total = len(secuencia)
    if total == 0:
        return 0
    entropia = 0
    for count in contador.values():
        probabilidad = count / total
        if probabilidad > 0:
            entropia -= probabilidad * math.log2(probabilidad)
    return round(entropia, 4)

def calcular_frecuencias_dinucleotidos(secuencia):
    secuencia = secuencia.upper()
    total_dinuc = len(secuencia) - 1
    if total_dinuc <= 0:
        return {dinuc: 0 for dinuc in ['AA', 'AT', 'CG', 'GC', 'TA', 'TT']}
    dinucleotidos = [secuencia[i:i+2] for i in range(len(secuencia)-1)]
    contador = Counter(dinucleotidos)
    dinuc_importantes = ['AA', 'AT', 'CG', 'GC', 'TA', 'TT']
    frecuencias = {}
    for dinuc in dinuc_importantes:
        valor = (contador.get(dinuc, 0) / total_dinuc) * 100
        frecuencias[f'dinuc_{dinuc}'] = round(valor, 2)
    return frecuencias

def calcular_conservacion_posicional(ventana_actual, todas_secuencias, inicio, tamano_ventana):
    if len(todas_secuencias) <= 1:
        return 100.0
    identidades = []
    for otra_seq in todas_secuencias:
        fin = min(inicio + tamano_ventana, len(otra_seq))
        otra_ventana = otra_seq[inicio:fin]
        longitud_min = min(len(ventana_actual), len(otra_ventana))
        if longitud_min == 0:
            continue
        matches = sum(1 for i in range(longitud_min)
                     if ventana_actual[i].upper() == otra_ventana[i].upper())
        identidad = (matches / longitud_min) * 100
        identidades.append(identidad)
    resultado = np.mean(identidades) if identidades else 100.0
    return round(resultado, 2)

def calcular_posicion_relativa(inicio, longitud_total):
    resultado = inicio / longitud_total if longitud_total > 0 else 0
    return round(resultado, 4)

def leer_gff3_completo(archivo_gff3):
    genes = []
    secuencia_partes = []
    leyendo_fasta = False
    nombre_variante = None

    with open(archivo_gff3, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()

            if linea.startswith('##FASTA'):
                leyendo_fasta = True
                continue

            if leyendo_fasta and linea.startswith('>'):
                nombre_variante = linea[1:].split()[0]
                continue

            if leyendo_fasta:
                secuencia_partes.append(linea)
                continue

            if linea.startswith('#') or not linea:
                continue

            campos = linea.split('\t')
            if len(campos) < 9:
                continue

            tipo_feature = campos[2]
            inicio = int(campos[3])
            fin = int(campos[4])
            atributos = campos[8]

            if tipo_feature == 'gene':
                match_nombre = re.search(r'Nombre=([^;]+)', atributos)
                match_id = re.search(r'ID=([^;]+)', atributos)
                nombre = match_nombre.group(1) if match_nombre else match_id.group(1)

                genes.append({
                    'nombre': nombre,
                    'inicio': inicio,
                    'fin': fin,
                    'longitud': fin - inicio + 1,
                    'tipo': 'gene'
                })

            elif tipo_feature == 'regulatory_region' or 'URR' in atributos or 'LCR' in atributos:
                match_nombre = re.search(r'Nombre=([^;]+)', atributos)
                nombre = match_nombre.group(1) if match_nombre else 'URR'

                genes.append({
                    'nombre': nombre,
                    'inicio': inicio,
                    'fin': fin,
                    'longitud': fin - inicio + 1,
                    'tipo': 'regulatory'
                })

    secuencia_completa = ''.join(secuencia_partes).upper()
    return genes, secuencia_completa, nombre_variante

def clasificar_ventana(inicio, fin, genes):
    for gen in genes:
        if inicio < gen['fin'] and fin > gen['inicio']:
            return 'C', gen['nombre']
    return 'V', 'Intergénica'

# ============================================================================
# FUNCIÓN DE LIMPIEZA
# ============================================================================

def limpiar_archivos(lista_archivos):
    """
    Elimina archivos del servidor después de procesarlos.
    """
    for archivo in lista_archivos:
        try:
            if os.path.exists(archivo):
                os.remove(archivo)
                print(f"  ✓ Eliminado: {os.path.basename(archivo)}")
        except Exception as e:
            print(f"  ⚠️ No se pudo eliminar {os.path.basename(archivo)}: {e}")

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================
# ============================================================================
# FUNCIÓN DE PREDICCIÓN (ADAPTADA A XGBOOST con clase invertida)
# ============================================================================

def predecir_region_genomica(datos_ventana):
    """
    Predice usando XGBoost entrenado con re-etiquetado:
      - En entrenamiento XGB: Variables (original 0) -> 1
                           Conservadas (original 1) -> 0
    Por tanto:
      prob_var = predict_proba(X)[:,1]  # P(Variables)
      pred_xgb = prob_var >= umbral
      pred_original = 0 si pred_xgb==1 else 1
    """
    if modelo is None or scaler is None:
        return {
            'prediccion': 'N/A',
            'prob_conservada': 0.0,
            'prob_variable': 0.0
        }

    try:
        # 1) Definir variables esperadas
        #    Si info_modelo trae variables_usadas, se usa eso (recomendado)
        if isinstance(info_modelo, dict) and info_modelo.get("variables_usadas"):
            variables_modelo = info_modelo["variables_usadas"]
        else:
            # fallback: lista fija (la misma que tú usabas, SIN posicion_relativa)
            variables_modelo = [
                'freq_A', 'freq_T', 'freq_C', 'freq_G',
                'contenido_GC',
                'GC_skew', 'AT_skew',
                'entropia_shannon',
                'dinuc_AA', 'dinuc_AT', 'dinuc_CG', 'dinuc_GC', 'dinuc_TA', 'dinuc_TT',
                'conservacion_posicional'
            ]

        # 2) Umbral (si está guardado), si no, 0.5
        umbral = 0.5
        if isinstance(info_modelo, dict) and "umbral_prob_var" in info_modelo:
            umbral = float(info_modelo["umbral_prob_var"])

        # 3) Filtrar solo variables esperadas
        datos_filtrados = {k: datos_ventana.get(k, None) for k in variables_modelo}

        # Validar faltantes
        faltantes = [k for k, v in datos_filtrados.items() if v is None]
        if faltantes:
            print(f"⚠️ Faltan variables: {faltantes}")
            return {
                'prediccion': 'Error',
                'prob_conservada': 0.0,
                'prob_variable': 0.0
            }

        # 4) DataFrame en orden correcto
        df_input = pd.DataFrame([datos_filtrados])[variables_modelo]

        # 5) Escalar (porque tú lo hiciste en preprocesamiento)
        X_scaled = scaler.transform(df_input)

        # 6) Probabilidad de Variables en el esquema XGB
        prob_var = float(modelo.predict_proba(X_scaled)[0][1])  # P(Variables)

        # 7) Predicción en esquema XGB (1=Variables, 0=Conservadas)
        pred_xgb = 1 if prob_var >= umbral else 0

        # 8) Convertir a esquema original:
        #    si pred_xgb=1 => original 0 (Variables)
        #    si pred_xgb=0 => original 1 (Conservadas)
        pred_original = 0 if pred_xgb == 1 else 1

        # 9) Retornar en tu formato
        resultado = {
            'prediccion': 'Conservada' if pred_original == 1 else 'Variable',
            'prob_conservada': round(1.0 - prob_var, 4),
            'prob_variable': round(prob_var, 4),
        }

        return resultado

    except Exception as e:
        print(f"Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return {
            'prediccion': 'Error',
            'prob_conservada': 0.0,
            'prob_variable': 0.0
        }


def procesar_multiples_gff3(archivos_gff3, tamano_ventana=100, paso=50):
    """Procesa múltiples archivos GFF3 y retorna DataFrame con variables"""
    
    print(f"Procesando {len(archivos_gff3)} archivos GFF3...")
    
    todas_secuencias_raw = []
    nombres_variantes_raw = []
    genes_referencia = None

    # Leer todos los archivos
    for idx, archivo in enumerate(archivos_gff3):
        genes, secuencia, nombre = leer_gff3_completo(archivo)

        if idx == 0:
            genes_referencia = genes

        todas_secuencias_raw.append(secuencia)
        nombres_variantes_raw.append(nombre if nombre else f"Variante_{idx+1}")

    # Filtrar secuencias con códigos IUPAC
    secuencias_validas = []
    nombres_validos = []
    secuencias_excluidas = []

    for nombre, secuencia in zip(nombres_variantes_raw, todas_secuencias_raw):
        tiene_amb, codigos = tiene_ambiguedades(secuencia)

        if tiene_amb:
            conteos_codigos = {}
            for codigo in codigos:
                conteos_codigos[codigo] = secuencia.count(codigo)

            secuencias_excluidas.append({
                'Nombre_Variante': nombre,
                'Codigos_IUPAC': ', '.join(sorted(codigos)),
                'Total_Bases_Ambiguas': sum(conteos_codigos.values())
            })
        else:
            secuencias_validas.append(secuencia)
            nombres_validos.append(nombre)

    if len(secuencias_validas) == 0:
        return None, secuencias_excluidas

    todas_secuencias = secuencias_validas
    nombres_variantes = nombres_validos

    # Extraer variables
    datos = []
    for idx_var, (secuencia, nombre_var) in enumerate(zip(todas_secuencias, nombres_variantes)):
        longitud_total = len(secuencia)

        for inicio in range(0, longitud_total - tamano_ventana + 1, paso):
            fin = inicio + tamano_ventana
            ventana = secuencia[inicio:fin]

            freq_A, freq_T, freq_C, freq_G = calcular_frecuencias_nucleotidos(ventana)
            dinucleotidos = calcular_frecuencias_dinucleotidos(ventana)
            etiqueta, region_genomica = clasificar_ventana(inicio, fin, genes_referencia)
            conservacion = calcular_conservacion_posicional(
                ventana, todas_secuencias, inicio, tamano_ventana
            )

            fila = {
                'variante': nombre_var,
                'inicio_ventana': inicio,
                'fin_ventana': fin,
                'longitud_ventana': len(ventana),
                'freq_A': freq_A,
                'freq_T': freq_T,
                'freq_C': freq_C,
                'freq_G': freq_G,
                'contenido_GC': calcular_contenido_gc(ventana),
                'GC_skew': calcular_gc_skew(ventana),
                'AT_skew': calcular_at_skew(ventana),
                'entropia_shannon': calcular_entropia_shannon(ventana),
                **dinucleotidos,
                'conservacion_posicional': conservacion,
                'posicion_relativa': calcular_posicion_relativa(inicio, longitud_total),
                'etiqueta': etiqueta,
                'region_genomica': region_genomica
            }

            datos.append(fila)

    df = pd.DataFrame(datos)
    return df, secuencias_excluidas

# ============================================================================
# RUTAS FLASK
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Variable global para almacenar el dataset procesado
dataset_global = None
dataset_completo_global = None  # Incluye predicciones
excluidas_global = None

@app.route('/')
def index():
    return render_template('index.html', datos=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset_global, dataset_completo_global, excluidas_global
    
    if 'files[]' not in request.files:
        flash('No se seleccionaron archivos')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No se seleccionaron archivos')
        return redirect(url_for('index'))
    
    # Guardar archivos temporalmente
    archivos_guardados = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            archivos_guardados.append(filepath)
    
    if not archivos_guardados:
        flash('No se cargaron archivos GFF3 válidos')
        return redirect(url_for('index'))
    
    # Procesar archivos GFF3
    try:
        df, excluidas = procesar_multiples_gff3(archivos_guardados, tamano_ventana=100, paso=50)
        
        if df is None:
            flash('Error: Todas las secuencias contienen códigos IUPAC')
            # Limpiar archivos antes de retornar
            limpiar_archivos(archivos_guardados)
            return redirect(url_for('index'))
        
        # Extraer variables de entrada
        columnas_excluir = ['etiqueta', 'region_genomica', 'variante', 
                           'inicio_ventana', 'fin_ventana', 'longitud_ventana']
        
        columnas_entrada = [col for col in df.columns if col not in columnas_excluir]
        
        # Guardar dataset de entrada CON posicion_relativa (para mostrar en frontend)
        dataset_global = df[columnas_entrada].copy()
        
        print(f"📊 Columnas extraídas: {columnas_entrada}")
        
        # HACER PREDICCIONES
        print("🤖 Realizando predicciones...")
        predicciones = []
        for idx, row in dataset_global.iterrows():
            pred = predecir_region_genomica(row.to_dict())
            predicciones.append(pred)
        
        # Crear DataFrame con predicciones
        df_predicciones = pd.DataFrame(predicciones)
        
        # Combinar variables de entrada + predicciones
        dataset_completo_global = pd.concat([dataset_global, df_predicciones], axis=1)
        
        excluidas_global = excluidas
        
        # ============================================================
        # LIMPIAR ARCHIVOS DESPUÉS DE PROCESAR EXITOSAMENTE
        # ============================================================
        limpiar_archivos(archivos_guardados)
        print(f"🧹 {len(archivos_guardados)} archivos eliminados del servidor")
        
        flash(f'✅ {len(archivos_guardados)} archivos procesados y {len(dataset_completo_global)} predicciones realizadas')
        
        # Preparar datos para la vista
        datos = {
            'total_filas': len(dataset_completo_global),
            'total_columnas': len(columnas_entrada),
            'total_archivos': len(archivos_guardados) - len(excluidas),
            'archivos_excluidos': len(excluidas),
            'modelo_cargado': modelo is not None
        }
        
        return render_template('index.html', datos=datos)
        
    except Exception as e:
        # Limpiar archivos en caso de error también
        limpiar_archivos(archivos_guardados)
        flash(f'Error al procesar archivos: {str(e)}')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))

@app.route('/get_data')
def get_data():
    """Endpoint para obtener datos paginados con predicciones"""
    global dataset_completo_global
    
    if dataset_completo_global is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    total_rows = len(dataset_completo_global)
    total_pages = (total_rows + per_page - 1) // per_page
    
    # Obtener datos de la página
    page_data = dataset_completo_global.iloc[start_idx:end_idx]
    
    # Separar columnas de variables y predicciones
    columnas_prediccion = ['prediccion', 'prob_conservada', 'prob_variable']
    columnas_variables = [col for col in dataset_completo_global.columns if col not in columnas_prediccion]
    
    # Convertir a formato JSON
    data = {
        'rows': page_data.to_dict('records'),
        'columnas_variables': columnas_variables,
        'columnas_prediccion': columnas_prediccion,
        'page': page,
        'per_page': per_page,
        'total_rows': total_rows,
        'total_pages': total_pages
    }
    
    return jsonify(data)

@app.route('/visualizar_circular')
def visualizar_circular():
    """Genera visualización circular del genoma con predicciones"""
    global dataset_completo_global
    
    if dataset_completo_global is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    try:
        # Obtener datos necesarios
        df = dataset_completo_global.copy()
        
        # Crear una columna de posición angular (0-360 grados)
        total_filas = len(df)
        df['angulo'] = [(i / total_filas) * 360 for i in range(total_filas)]
        
        def clamp(x, lo=0, hi=255):
            return max(lo, min(hi, int(x)))
        # Crear color basado en probabilidad de conservada
        # 0-30%: Rojo, 30-70%: Amarillo, 70-100%: Verde
        def obtener_color(prob):
            # Asegurar que prob esté en 0..1
            try:
                prob = float(prob)
            except:
                prob = 0.0
            prob = max(0.0, min(1.0, prob))

            if prob >= 0.70:
                # Verde (más intenso mientras más prob)
                t = (prob - 0.70) / 0.30  # 0..1
                r = clamp(255 * (1 - t))      # 255 -> 0
                g = clamp(180 + 75 * t)       # 180 -> 255
                b = clamp(80)                 # fijo
                return f"rgb({r},{g},{b})"

            elif prob >= 0.30:
                # Amarillo / naranja suave (intermedio)
                t = (prob - 0.30) / 0.40      # 0..1
                r = clamp(255)
                g = clamp(200 - 20 * t)       # 200 -> 180
                b = clamp(60)                 # fijo
                return f"rgb({r},{g},{b})"

            else:
                # Rojo (más intenso mientras más baja prob)
                t = prob / 0.30               # 0..1
                r = clamp(255)
                g = clamp(50 + 100 * t)       # 50 -> 150
                b = clamp(50 + 50 * t)        # 50 -> 100
                return f"rgb({r},{g},{b})"
        
        df['color'] = df['prob_conservada'].apply(obtener_color)
        
        # Crear texto para hover (tooltip)
        df['hover_text'] = df.apply(lambda row: 
            f"<b>Ventana #{df.index.get_loc(row.name) + 1}</b><br>" +
            f"Predicción: <b>{row['prediccion']}</b><br>" +
            f"Prob. Conservada: <b>{row['prob_conservada']:.2%}</b><br>" +
            f"Prob. Variable: <b>{row['prob_variable']:.2%}</b><br>" +
            f"GC: {row.get('contenido_GC', 0):.2f}%<br>" +
            f"Entropía: {row.get('entropia_shannon', 0):.4f}<br>" +
            f"Conservación: {row.get('conservacion_posicional', 0):.2f}%",
            axis=1
        )
        
        # Crear gráfico polar
        fig = go.Figure()
        
        # Agregar las barras en coordenadas polares
        fig.add_trace(go.Barpolar(
            r=[1] * len(df),  # Todas en el mismo radio
            theta=df['angulo'],
            width=[360/total_filas] * len(df),  # Ancho de cada barra
            marker=dict(
                color=df['color'],
                line=dict(color='white', width=0.5)
            ),
            hovertext=df['hover_text'],
            hoverinfo='text',
            name='Ventanas genómicas'
        ))
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': '🧬 Visualización Circular del Genoma HPV<br><sub>Rojo: Variable | Amarilla: Intermedia | Verde: Conservada</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333'}
            },
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, 1.2]
                ),
                angularaxis=dict(
                    visible=True,
                    tickmode='linear',
                    tick0=0,
                    dtick=30,
                    direction='clockwise',
                    rotation=90
                ),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            showlegend=False,
            height=700,
            paper_bgcolor='#f8f9fa',
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Agregar anotación en el centro
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"<b>HPV</b><br>{total_filas} ventanas",
            showarrow=False,
            font=dict(size=16, color='#667eea'),
            xref="paper",
            yref="paper"
        )
        
        # Convertir a HTML
        graph_html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={
                'responsive': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': [
                    'zoom2d',
                    'pan2d',
                    'select2d',
                    'lasso2d',
                    'zoomIn2d',
                    'zoomOut2d',
                    'autoScale2d',
                    'resetScale2d'
                ],
                'modeBarButtonsToAdd': ['toImage']
            }
        )
        
        return render_template('visualizacion_circular.html', graph_html=graph_html, total_ventanas=total_filas)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/descargar_excel')
def descargar_excel():
    global dataset_completo_global
    
    if dataset_completo_global is None or len(dataset_completo_global) == 0:
        return jsonify({'error': 'No hay datos para descargar. Primero carga archivos GFF3.'}), 400

    try:
        # Copia para no tocar el global
        df = dataset_completo_global.copy()

        # (Opcional) ordenar si existen columnas
        if 'variante' in df.columns and 'inicio_ventana' in df.columns:
            df = df.sort_values(['variante', 'inicio_ventana']).reset_index(drop=True)

        # Nombre archivo
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resultados_predicciones_HPV_{stamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Exportar a Excel
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Resultados")

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)