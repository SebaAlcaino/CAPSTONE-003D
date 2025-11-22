import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from langchain_ollama import ChatOllama
import json
import os
import tempfile
import io
import plotly.io as pio
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# Configuración general

st.set_page_config(page_title="Dashboard inteligente", layout="wide")
st.title("InformIA")

# Carga de base de datos
st.sidebar.header("Configuración")

archivos_locales = [f for f in os.listdir(".") if f.endswith(".db")]
opcion = st.sidebar.selectbox(
    "Selecciona una base de datos o sube una nueva:",
    ["Subir archivo..."] + archivos_locales
)

if opcion == "Subir archivo...":
    archivo_subido = st.sidebar.file_uploader("Sube una base de datos SQLite (.db)", type=["db"])
    if archivo_subido:
        temp_dir = tempfile.gettempdir()
        path_db = os.path.join(temp_dir, archivo_subido.name)
        with open(path_db, "wb") as f:
            f.write(archivo_subido.getbuffer())
        st.sidebar.success(f"Base de datos cargada: {path_db}")
    else:
        st.stop()
else:
    path_db = opcion


# Obtener estructura de la base de datos

def obtener_estructura_y_ejemplos(path_db):
    conn = sqlite3.connect(path_db)
    cursor = conn.cursor()
    
    # Obtener tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tablas = [t[0] for t in cursor.fetchall()]

    estructura = {}
    ejemplos = {}

    for tabla in tablas:
        # Obtener columnas
        cursor.execute(f"PRAGMA table_info({tabla});")
        columnas = [col[1] for col in cursor.fetchall()]
        estructura[tabla] = columnas

        # Obtener 1 registro de ejemplo
        try:
            df_ej = pd.read_sql_query(f"SELECT * FROM {tabla} LIMIT 1;", conn)
            ejemplos[tabla] = df_ej.to_dict(orient="records")[0] if not df_ej.empty else {}
        except:
            ejemplos[tabla] = {}

    conn.close()
    return estructura, ejemplos

try:
    estructura_db, ejemplos_db = obtener_estructura_y_ejemplos(path_db)

    st.sidebar.write("**Tablas encontradas:**")
    st.sidebar.json(estructura_db)

except Exception as e:
    st.error(f"No se pudo leer la base de datos seleccionada: {e}")
    st.stop()


# Inicializar Ollama
try:
    # llm = ChatOllama(model="llama3", temperature=0, base_url="http://host.docker.internal:11434")
    llm = ChatOllama(model="llama3", temperature=0)
except Exception as e:
    st.error(f"No se pudo inicializar el modelo Ollama: {e}")
    st.stop()


# Entrada del usuario

consulta_usuario = st.text_input("Escribe tu pregunta:").strip()

# Lista para guardar imágenes de gráficos (bytes en memoria)
imagenes_bytes = []

if consulta_usuario:
    try:
        
        # Generar SQL con IA
        prompt_sql = f"""
Eres un experto en SQL para SQLite.

Reglas estrictas:

1. La base de datos puede tener fechas en formato "dd-mm-yyyy" o "yyyy-mm-dd".
2. Nunca uses funciones que no existan en SQLite (como MONTH(), YEAR(), DATEPART(), etc.).
3. Para extraer el año de la fecha, detecta el formato:
   - Si el carácter en la posición 5 es "-", la fecha está en formato "yyyy-mm-dd".
   - Si no, está en formato "dd-mm-yyyy".

   Año:
   CASE
       WHEN substr(fecha, 5, 1) = '-' THEN substr(fecha, 1, 4)
       ELSE substr(fecha, 7, 4)
   END

4. Para extraer el mes según el formato:

   Mes:
   CASE
       WHEN substr(fecha, 5, 1) = '-' THEN substr(fecha, 6, 2)
       ELSE substr(fecha, 4, 2)
   END

5. Para extraer el día según el formato:

   Día:
   CASE
       WHEN substr(fecha, 5, 1) = '-' THEN substr(fecha, 9, 2)
       ELSE substr(fecha, 1, 2)
   END
   
REGLA EXTRA — FILTROS DE MES/AÑO
- SOLO aplique filtros por año/mes/día si el usuario menciona explícitamente una referencia temporal:
  (ej.: '2024', 'enero', 'mes 01', 'entre 2023 y 2024', 'ventas de marzo 2024', '2024-05', etc.)
- No uses LIKE con patrones de fecha (ej. LIKE '____-01-%') a menos que el usuario pida específicamente un mes o un patrón textual.
- Para año/mes use la lógica CASE (ver reglas anteriores) para soportar dd-mm-yyyy y yyyy-mm-dd.

REGLA: Uso de lógica de fechas SOLO si el usuario lo solicita
La consulta SQL debe usar substr(fecha, ...) y cualquier lógica de fecha UNICAMENTE si el usuario menciona explícitamente elementos temporales, como:
-un año (ej: “2024”, “2023”)
-un mes (ej: “enero”, “03”, “marzo”)
-una fecha específica (ej: “01-04-2024”)
-rangos (ej: “entre 2023 y 2024”)
-términos temporales (ej: “mensual”, “semanal”, “anual”)
-expresiones como “ventas de 2024”, “productos en mayo”, “diario”, “por fecha”

REGLA CRÍTICA SOBRE IDENTIFICADORES:

- Las columnas que contengan "id", "ID", "_id", "Id", "identificador" o variaciones similares
  NO representan cantidades, totales ni métricas.
- Son SOLO identificadores únicos.
- Nunca deben interpretarse como:
  * número de productos vendidos,
  * volumen de ventas,
  * montos,
  * cantidades,
  * frecuencia,
  * ranking,
  * demanda,
  * popularidad.
- No pueden usarse para análisis, sumas, totales ni conclusiones.
- Si aparece un valor numérico en una columna identificadora (como 300),
  se debe interpretar EXCLUSIVAMENTE como un ID, nunca como un total.
- Si la consulta pide "cuántos se vendieron", entonces:
    → usar COUNT(*) después de aplicar filtros
    → o usar una columna explícita de cantidad SI existe.
  Pero JAMÁS usar valores del ID.

Si el usuario NO menciona ninguna referencia temporal, entonces:
-NO uses substr(fecha, ...)
-NO filtres por año o mes
-NO crees columnas temporales
-NO interpretes que debe haber filtro temporal

En esos casos, la consulta debe trabajar SOLO con las columnas que el usuario mencionó.

6. Genera SOLO SQL válido para SQLite, sin comentarios ni explicaciones.
7. Devuelve únicamente la consulta final terminada en punto y coma.
8. No inventes columnas, no inventes tablas, no inventes datos.
9. Si usas funciones agregadas (COUNT, SUM, AVG, etc.), la columna debe tener alias descriptivo.
10. Si el usuario solicita filtrar por mes o año, usa los CASE anteriores.
11. Si el usuario solicita ordenamiento por conteos o sumas, usa ORDER BY con el alias generado.

Estructura de la base de datos (tablas y columnas):

{json.dumps(estructura_db, indent=2)}

Ejemplos de registros de cada tabla:

{json.dumps(ejemplos_db, indent=2)}

Pregunta del usuario:
"{consulta_usuario}"

Si el usuario solicita un año específico (por ejemplo “2024”), la consulta debe filtrar únicamente ese año.
No está permitido agregar condiciones adicionales del tipo:

OR ...

BETWEEN ...

>=, <=

Rangos de meses

Años distintos al solicitado

No incluyas texto adicional, explicaciones ni comentarios.  
SOLO devuelve SQL válido terminado en punto y coma.
"""
        sql_generado = llm.invoke(prompt_sql).content.strip()
        st.subheader("Consulta SQL generada:")
        st.code(sql_generado, language="sql")


        # Ejecutar la consulta

        conn = sqlite3.connect(path_db)
        df = pd.read_sql_query(sql_generado, conn)
        conn.close()
        
        
        
        # Conversión fechas
        
        def detectar_columnas_fecha(df):
            columnas_fecha = []
            for col in df.columns:
                muestra = df[col].astype(str).head(5)

                # formato dd-mm-yyyy
                if muestra.str.match(r"^\d{2}-\d{2}-\d{4}$").all():
                    columnas_fecha.append(col)
                    continue    

                # formato yyyy-mm-dd
                if muestra.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
                    columnas_fecha.append(col)
                    continue

            return columnas_fecha

        def convertir_fechas(df):
            columnas_fecha = detectar_columnas_fecha(df)

            for col in columnas_fecha:
                try:
                    # intenta dd-mm-yyyy
                    df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="ignore")
                    # intenta yyyy-mm-dd
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="ignore")

                    # dejar solo fecha sin hora
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.date

                except:
                    pass

            return df

        df = convertir_fechas(df)

        # Ordenar si hay cualquier columna datetime
        columnas_datetime = df.select_dtypes(include=["datetime64[ns]"]).columns
        if len(columnas_datetime) > 0:
            df = df.sort_values(by=columnas_datetime[0])

        if df.empty:
            st.info("La consulta no devolvió resultados.")
        else:
            st.subheader("Resultados")
            st.dataframe(df)

            # Gráficos sugeridos Ollama
            
            st.subheader("Gráficos sugeridos:")

            prompt_graficos = f"""
Tienes un DataFrame con columnas: {list(df.columns)}.
La consulta del usuario fue: "{consulta_usuario}".

Genera 2 o 3 gráficos en formato JSON con los campos:
- tipo (linea, barra, scatter, torta)
- x
- y (opcional en torta)

REGLAS ESTRICTAS (DEBEN CUMPLIRSE AL 100%):

1. JAMÁS selecciones columnas cuyo nombre contenga:
- "id"
- "ID"
- "_id"
- "Id"
- "identificador"
- cualquier variación de estas (usa coincidencia por substring).
Si una columna tiene esas letras en cualquier parte, NO LA USES.

2. Para gráficos de torta:
- SOLO usa columnas categóricas.
- NO uses columnas numéricas en "x".
- NO uses columnas prohibidas (punto 1).

3. Para gráficos de línea o barra:
- "x" debe ser una columna categórica o fecha.
- "y" debe ser una columna numérica.
- Ninguna columna de "x" o "y" puede ser una columna prohibida (punto 1).

4. Si no existen columnas válidas para un tipo de gráfico, NO inventes columnas.
Simplemente omite ese tipo y genera otro que sí cumpla todas las reglas.

5. Responde **solo JSON válido**, sin texto adicional.
Responde *solo* en formato JSON, por ejemplo:
[{{"tipo":"linea","x":"mes","y":"ventas"}},{{"tipo":"torta","x":"region"}}]
"""

            figuras = []  # guardaremos objetos plotly aquí

            try:
                raw = llm.invoke(prompt_graficos).content.strip()

                # Extraer JSON de la respuesta
                inicio = raw.find('[')
                fin = raw.rfind(']') + 1
                graficos_sugeridos_json = raw[inicio:fin]
                graficos_sugeridos = json.loads(graficos_sugeridos_json)

                if isinstance(graficos_sugeridos, dict):
                    graficos_sugeridos = [graficos_sugeridos]
                if not isinstance(graficos_sugeridos, list):
                    raise ValueError("Formato de gráficos no válido")

                for g in graficos_sugeridos:
                    tipo = g.get("tipo", "barra").lower()
                    x_col = g.get("x", df.columns[0])
                    y_col = g.get("y", None)

                    # Validar que columnas existan
                    if x_col not in df.columns:
                        st.warning(f"Gráfico omitido: '{x_col}' no es una columna válida.")
                        continue

                    if tipo != "torta":
                        numeric_cols = df.select_dtypes(include="number").columns
                        if y_col not in df.columns or y_col is None:
                            if len(numeric_cols) == 0:
                                st.warning(f"Gráfico omitido: no hay columnas numéricas para 'y'.")
                                continue
                            y_col = numeric_cols[0]
                        if y_col not in df.columns:
                            st.warning(f"Gráfico omitido: '{y_col}' no es una columna válida.")
                            continue
                        
                    if tipo == "linea":
                        if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
                            st.warning(f"Gráfico de línea omitido: '{x_col}' no es una fecha válida.")
                            continue

                        # Crear columna sin hora
                        df["_fecha"] = df[x_col].dt.date

                        fig = px.line(df, x="_fecha", y=y_col, title=f"{y_col} por {x_col}")

                        df.drop(columns=["_fecha"], inplace=True)

                    elif tipo == "barra":
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} por {x_col}")
                        text_auto=True

                    elif tipo == "scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")

                    elif tipo == "torta":
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) == 0:
                            st.warning("No hay columnas numéricas para gráfico de torta; se omitirá.")
                            continue
                        y_col = numeric_cols[0]  # usar primera columna numérica
                        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} por {x_col}")

                    else:
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} por {x_col}")
                        text_auto=True

                    st.plotly_chart(fig, width="stretch")

                    # Guardar imagen para PDF
                    try:
                        img_bytes = pio.to_image(fig, format="png")
                        imagenes_bytes.append(img_bytes)
                        figuras.append(fig)
                    except Exception as e:
                        st.warning(f"No se pudo exportar la figura a imagen: {e}")

            except Exception as e:
                st.warning(f"No se pudieron generar gráficos inteligentes: {e}")


            # Explicación automática
            st.subheader("Interpretación del resultado:")
            try:
                prompt_explicacion = f"""
Eres un analista de datos de ventas extremadamente estricto.  
Tu tarea es responder EXACTAMENTE la consulta del usuario ({consulta_usuario}) usando solo los datos que aparecen en el JSON.

REGLAS OBLIGATORIAS:

1. Responde únicamente a la consulta del usuario.
2. Usa exclusivamente los valores del JSON.
3. No inventes totales, porcentajes, rankings, tendencias ni conclusiones que no puedan calcularse directamente.
4. No inventes fechas con hora, motivaciones, ni comportamiento de regiones.
5. Si algo NO aparece, responde: "No aparece en los datos".
6. No incluyas saludos, despedidas, recomendaciones ni texto de relleno.
7. Solo se permiten cálculos directos del JSON:
   - Conteos
   - Sumas
   - Valores únicos
   - Agrupaciones exactas
8. Puedes mencionar patrones o información útil estrictamente presentes en los datos, pero **sin encabezados adicionales** ni texto que no sea parte de los resultados.

REGLA ABSOLUTA SOBRE IDENTIFICADORES:

- Las columnas que contengan "id", "ID", "_id", "Id" o "identificador" en cualquier parte del nombre:
    * NO deben aparecer en la respuesta.
    * NO deben ser listadas entre paréntesis.
    * NO deben usarse para enumerar elementos.
    * NO deben ser mostradas como detalles, ejemplos, metadatos ni información adicional.
- Los identificadores JAMÁS deben mostrarse, ni siquiera como listas explicativas.
- Si la consulta requiere contar ventas, productos o registros:
    → usar solamente COUNT(*) o agrupaciones, nunca los valores del ID.
- Toda respuesta debe omitir completamente cualquier columna de identificación,
  incluso si el modelo cree que son útiles.
  
- Nunca incluyas IDs entre paréntesis.
- Nunca muestres listas de identificadores.
- Solo muestra categorías reales y valores calculados (conteos, sumas, etc.).

FORMATO DE RESPUESTA OBLIGATORIO:

- Usa listas con guiones o viñetas para todos los resultados que tengan múltiples elementos.
- Cada elemento debe mostrar claramente la categoría y su valor.
- Evita tablas tipo DataFrame; todo debe ser texto plano legible.
- Si hay solo un valor, indícalo de forma clara con frase corta.
- Presenta patrones o información útil directamente como parte de la lista, sin texto extra.

DATOS ENTREGADOS:
DATAFRAME_JSON:
{df.to_json(orient="records")}

GENERAR:
Un informe estructurado que responda directamente a "{consulta_usuario}" siguiendo todas las reglas y formato de lista descrito arriba.
"""
                explicacion = llm.invoke(prompt_explicacion).content
                st.write(explicacion)
            except Exception as e:
                explicacion = f"No se pudo generar explicación automática: {e}"
                st.warning(explicacion)
                
            # EXPORTAR A PDF

            st.subheader("Exportar reporte a PDF")

            if st.button("Generar y descargar PDF"):
                # buffer en memoria
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                # Título y consulta
                story.append(Paragraph("Dashboard Inteligente - Reporte", styles["Title"]))
                story.append(Spacer(1, 8))
                story.append(Paragraph(f"<b>Consulta:</b> {consulta_usuario}", styles["Normal"]))
                story.append(Spacer(1, 8))


                # Tabla resumen (primeras filas)
                story.append(Paragraph("<b>Resumen (10 filas):</b>", styles["Heading2"]))
                # convertir tabla a texto simple para reportlab
                tabla_text = df.head(10).to_string(index=False)
                story.append(Preformatted(tabla_text, styles["Code"]))
                story.append(Spacer(1, 12))

                # Explicación
                story.append(Paragraph("<b>Interpretación / Conclusiones:</b>", styles["Heading2"]))
                story.append(Paragraph(explicacion.replace("\n", "<br/>"), styles["Normal"]))
                story.append(Spacer(1, 12))

                # Gráficos (si hay)
                if imagenes_bytes:
                    story.append(Paragraph("<b>Gráficos:</b>", styles["Heading2"]))
                    for img_b in imagenes_bytes:
                        img_buf = io.BytesIO(img_b)
                        # Image acepta file-like objects
                        story.append(Image(img_buf, width=6*inch, height=3.5*inch))
                        story.append(Spacer(1, 12))
                else:
                    story.append(Paragraph("No hay gráficos disponibles para exportar.", styles["Normal"]))

                # construir PDF en memoria
                doc.build(story)
                pdf_buffer.seek(0)

                # Descargar
                st.download_button(
                    label="Descargar PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=consulta_usuario.replace(" ", "_") + ".pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error general: {e}")
else:
    st.info("Escribe una consulta para generar SQL, gráficos y reportes.")
