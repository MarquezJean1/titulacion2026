from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from collections import Counter
import math
import re

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'gff3', 'gff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
excluidas_global = None

@app.route('/')
def index():
    return render_template('index.html', datos=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset_global, excluidas_global
    
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
            return redirect(url_for('index'))
        
        # Extraer solo variables de entrada
        columnas_excluir = ['etiqueta', 'region_genomica', 'variante', 
                           'inicio_ventana', 'fin_ventana', 'longitud_ventana']
        
        columnas_entrada = [col for col in df.columns if col not in columnas_excluir]
        
        # Guardar en variables globales
        dataset_global = df[columnas_entrada].copy()
        excluidas_global = excluidas
        
        flash(f'✅ {len(archivos_guardados)} archivos procesados exitosamente')
        
        # Preparar datos para la vista
        datos = {
            'total_filas': len(dataset_global),
            'total_columnas': len(columnas_entrada),
            'columnas': columnas_entrada,
            'total_archivos': len(archivos_guardados),
            'archivos_excluidos': len(excluidas)
        }
        
        return render_template('index.html', datos=datos)
        
    except Exception as e:
        flash(f'Error al procesar archivos: {str(e)}')
        return redirect(url_for('index'))

@app.route('/get_data')
def get_data():
    """Endpoint para obtener datos paginados"""
    global dataset_global
    
    if dataset_global is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    total_rows = len(dataset_global)
    total_pages = (total_rows + per_page - 1) // per_page
    
    # Obtener datos de la página
    page_data = dataset_global.iloc[start_idx:end_idx]
    
    # Convertir a formato JSON
    data = {
        'rows': page_data.to_dict('records'),
        'columns': list(dataset_global.columns),
        'page': page,
        'per_page': per_page,
        'total_rows': total_rows,
        'total_pages': total_pages
    }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)