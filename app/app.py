import streamlit as st
import pandas as pd
import sklearn.tree  # Importar explícitamente para que joblib reconozca las clases
import joblib
import numpy as np
import os

# Forma del sombrero
map_cap_shape = {
    "Campana": "b",
    "Cónica": "c",
    "Convexa": "x",
    "Plana": "f",
    "Nudosa": "k",
    "Hundida": "s"
}

# Superficie del sombrero
map_cap_surface = {
    "Fibrosa": "f",
    "Con ranuras": "g",
    "Escamosa": "y",
    "Lisa": "s"
}

# Color del sombrero
map_cap_color = {
    "Marrón": "n",
    "Beige": "b",
    "Canela": "c",
    "Gris": "g",
    "Verde": "r",
    "Rosa": "p",
    "Púrpura": "u",
    "Rojo": "e",
    "Blanco": "w",
    "Amarillo": "y"
}

# Magulladuras
map_bruises = {
    "Con magulladuras": "t",
    "Sin magulladuras": "f"
}

# Color de las láminas
map_gill_color = {
    "Negro": "k",
    "Marrón": "n",
    "Beige": "b",
    "Chocolate": "h",
    "Gris": "g",
    "Verde": "r",
    "Naranja": "o",
    "Rosa": "p",
    "Púrpura": "u",
    "Rojo": "e",
    "Blanco": "w",
    "Amarillo": "y"
}

# Forma del tallo
map_stalk_shape = {
    "Ensanchado hacia la base": "e",
    "Afilándose hacia la base": "t"
}

# Superficie del tallo por encima del anillo
map_stalk_surface_above_ring = {
    "Fibrosa": "f",
    "Escamosa": "y",
    "Sedosa": "k",
    "Lisa": "s"
}

# Superficie del tallo por debajo del anillo
map_stalk_surface_below_ring = {
    "Fibrosa": "f",
    "Escamosa": "y",
    "Sedosa": "k",
    "Lisa": "s"
}

# Color del tallo por encima del anillo (coincide con el de abajo, así que creamos uno genérico)
map_stalk_color = { # Usamos un mapa genérico si las opciones son las mismas para "above" y "below"
    "Marrón": "n",
    "Beige": "b",
    "Canela": "c",
    "Gris": "g",
    "Naranja": "o",
    "Rosa": "p",
    "Rojo": "e",
    "Blanco": "w",
    "Amarillo": "y"
}

# Color del velo
map_veil_color = {
    "Marrón": "n",
    "Naranja": "o",
    "Blanco": "w",
    "Amarillo": "y"
}

# Número de anillos
map_ring_number = {
    "Ninguno": "n",
    "Uno": "o",
    "Dos": "t"
}

# Tipo de anillo
map_ring_type = {
    "Telaraña": "c",
    "Evanescente": "e",
    "Acampanado": "f",
    "Grande": "l",
    "Ninguno": "n",
    "Colgante": "p",
    "Enfundado": "s",
    "Zonal": "z"
}

# Población
map_population = {
    "Abundante": "a",
    "Agrupado": "c",
    "Numeroso": "n",
    "Disperso": "s",
    "Varios": "v",
    "Solitario": "y"
}

# Hábitat
map_habitat = {
    "Pastizales": "g",
    "Hojas": "l",
    "Prados": "m",
    "Senderos": "p",
    "Urbano": "u",
    "Desechos": "w",
    "Bosques": "d"
}

# --- Carga del modelo y utilidades ---
# Definimos la ruta a la carpeta 'models'.
models_folder = os.path.join(os.path.dirname(__file__), '..', 'models')

# Puedes descomentar estas líneas si quieres ver la ruta y los archivos al inicio para depurar
# print("Ruta models_folder:", models_folder)
# print("Archivos en models_folder:", os.listdir(models_folder))

try:
    model = joblib.load(os.path.join(models_folder, 'best_decision_tree_model_streamlit.pkl'))
    label_encoder = joblib.load(os.path.join(models_folder, 'label_encoder_y.pkl'))
    ohe_columns = joblib.load(os.path.join(models_folder, 'ohe_columns_for_streamlit.pkl'))
except FileNotFoundError:
    st.error(f"Error al cargar los archivos del modelo. Asegúrate de que los archivos .pkl estén en la carpeta '{models_folder}'.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar los archivos del modelo: {e}")
    st.stop()

# --- Definir las características que usará el modelo (¡las mismas que usaste para entrenar!) ---
# Es crucial que esta lista y el orden sean IDÉNTICOS a 'streamlit_features' de tu entrenamiento.
streamlit_features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises',
    'gill-color', 'stalk-shape', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-color', 'ring-number',
    'ring-type', 'population', 'habitat'
]


# --- Menú lateral ---
tabs = st.tabs([
    "🏠 Inicio", 
    "🔍 Predicción Interactiva", 
    "📂 Predicción por Archivo", 
    "📄 Contenido Descargable", 
    "🚀 Próximos Pasos"
])


# --- PÁGINA 1: Introducción ---
with tabs[0]:
    st.title("🍄 FungiScan: ¿Comestible o Venenosa? 🍄")
    st.markdown("""
    ¡Bienvenido a **FungiScan**, tu guía personal para identificar setas!
    Introduce las características que observes en tu seta y te ayudaremos a determinar si es segura.

    Esta app fue diseñada con un enfoque en **seguridad y transparencia**, utilizando modelos de aprendizaje automático entrenados con características visuales accesibles para personas no expertas.

    
    """)
    st.warning("¡Recuerda! Esta aplicación es una herramienta de ayuda. Ante la menor duda sobre una seta, NUNCA la consumas y consulta siempre a un experto micólogo.")

    st.subheader("📚 ¿Cómo identificar las caracteristicas de una seta?")
    st.markdown(
        """
        Para realizar una predicción precisa, es fundamental entender las características
        que el modelo utiliza. Esta guía te ayudará a interpretar las opciones en el
        formulario de predicción.

        **Recomendamos leer esta guía antes de introducir los datos de la seta.**
        """
    )

    st.subheader("Características del Sombrero")
    with st.expander("Forma del Sombrero"):
        st.markdown("""
        Se refiere a la silueta general del sombrero de la seta.
        * **Campanulado (b):** Forma de campana.
        * **Cónico (c):** Forma de cono o puntiaguda.
        * **Convexo (x):** Curvado hacia arriba, como una cúpula.
        * **Plano (f):** Completamente horizontal o casi.
        * **Perilla (k):** Tiene un bulto o protuberancia en el centro del sombrero.
        * **Hundido (s):** El centro del sombrero está ligeramente deprimido.
        """)
    with st.expander("Superficie del Sombrero"):
        st.markdown("""
        Describe la textura de la parte superior del sombrero.
        * **Fibroso (f):** Con fibras finas o pelos.
        * **Con surcos (g):** Presenta estrías o ranuras.
        * **Escamoso (y):** Cubierto de pequeñas escamas o "copos".
        * **Liso (s):** Sin textura notable, uniforme.
        """)
    with st.expander("Color del Sombrero"):
        st.markdown("""
        Color predominante en la parte superior del sombrero.
        * **Marrón (n)**
        * **Beige (b)**
        * **Canela (c)**
        * **Gris (g)**
        * **Verde (r)**
        * **Rosa (p)**
        * **Púrpura (u)**
        * **Rojo (e)**
        * **Blanco (w)**
        * **Amarillo (y)**
        """)

    st.subheader("Otras Características Clave")
    with st.expander("¿Se forman magulladuras al tocarla?"):
        st.markdown("""
        Si la carne de la seta cambia de color (por ejemplo, a azul, rojo o marrón) al ser tocada o cortada.
        * **Sí (t):** Se magulla.
        * **No (f):** No se magulla.
        """)
    with st.expander("Color de las Láminas"):
        st.markdown("""
        El color de las láminas en la parte inferior del sombrero.
        * **Negro (k)**
        * **Marrón (n)**
        * **Beige (b)**
        * **Chocolate (h)**
        * **Gris (g)**
        * **Verde (r)**
        * **Naranja (o)**
        * **Rosa (p)**
        * **Púrpura (u)**
        * **Rojo (e)**
        * **Blanco (w)**
        * **Amarillo (y)**
        """)

    st.subheader("Características del Tallo")
    with st.expander("Forma del Tallo"):
        st.markdown("""
        La forma general del tallo.
        * **Ensanchado hacia la base (e):** El tallo se hace más grueso hacia el suelo.
        * **Estrecho hacia la base (t):** El tallo se hace más delgado hacia el suelo.
        """)
    with st.expander("Superficie del Tallo (arriba del anillo)"):
        st.markdown("""
        La textura de la parte superior del tallo, justo debajo del sombrero.
        * **Fibroso (f):** Con fibras o hilos.
        * **Sedoso (k):** Con una apariencia suave y brillante.
        * **Liso (s):** Sin textura notable.
        * **Escamoso (y):** Con pequeñas escamas.
        """)
    with st.expander("Superficie del Tallo (debajo del anillo)"):
        st.markdown("""
        La textura de la parte inferior del tallo, debajo del anillo.
        * **Fibroso (f)**
        * **Sedoso (k)**
        * **Liso (s)**
        * **Escamoso (y)**
        """)
    with st.expander("Color del Tallo (arriba del anillo)"):
        st.markdown("""
        Color predominante de la parte superior del tallo.
        * **Marrón (n)**
        * **Beige (b)**
        * **Canela (c)**
        * **Gris (g)**
        * **Naranja (o)**
        * **Rosa (p)**
        * **Rojo (e)**
        * **Blanco (w)**
        * **Amarillo (y)**
        """)
    with st.expander("Color del Tallo (debajo del anillo)"):
        st.markdown("""
        Color predominante de la parte inferior del tallo.
        * **Marrón (n)**
        * **Beige (b)**
        * **Canela (c)**
        * **Gris (g)**
        * **Naranja (o)**
        * **Rosa (p)**
        * **Rojo (e)**
        * **Blanco (w)**
        * **Amarillo (y)**
        """)

    st.subheader("Características del Velo y Anillo")
    with st.expander("Color del Velo"):
        st.markdown("""
        El color del velo (membrana que cubre las láminas cuando la seta es joven).
        * **Marrón (n)**
        * **Naranja (o)**
        * **Blanco (w)**
        * **Amarillo (y)**
        """)
    with st.expander("Número de Anillos"):
        st.markdown("""
        Cantidad de anillos presentes en el tallo.
        * **Ninguno (n):** Sin anillo.
        * **Uno (o):** Un solo anillo.
        * **Dos (t):** Dos anillos.
        """)
    with st.expander("Tipo de Anillo"):
        st.markdown("""
        La forma o característica del anillo en el tallo.
        * **Telaraña (c):** Fino, como una telaraña.
        * **Evanescente (e):** Que desaparece rápidamente.
        * **Acampanado (f):** Con forma de campana.
        * **Grande (l):** Anillo prominente.
        * **Ninguno (n):** Sin anillo.
        * **Colgante (p):** Cuelga del tallo.
        * **Envainador (s):** Envuelve el tallo como una vaina.
        * **Zona (z):** Una banda en el tallo sin ser un anillo definido.
        """)

    st.subheader("Otras Características del Entorno")
    with st.expander("Población"):
        st.markdown("""
        Cómo crece la seta en su entorno.
        * **Abundante (a):** Crece en grandes cantidades.
        * **Agrupado (c):** Crece en pequeños grupos.
        * **Numeroso (n):** Varias setas dispersas.
        * **Disperso (s):** Pocas setas separadas.
        * **Varios (v):** Un número moderado de setas.
        * **Solitario (y):** Crece individualmente.
        """)
    with st.expander("Hábitat"):
        st.markdown("""
        El tipo de entorno donde se encuentra la seta.
        * **Hierba (g):** En zonas de césped.
        * **Hojas (l):** Entre hojarasca o restos vegetales.
        * **Prados (m):** En campos abiertos.
        * **Caminos (p):** Cerca o en caminos.
        * **Urbano (u):** En zonas urbanas.
        * **Desperdicios (w):** En zonas de desechos o residuos.
        * **Bosques (d):** En áreas boscosas.
        """)

# --- PÁGINA 2: Predicción Interactiva ---
with tabs[1]:
    st.header("🔍 Predicción por características observadas")
    st.markdown("Selecciona las características de la seta que deseas clasificar:")

    # Diccionario para almacenar las selecciones del usuario
    user_selections = {}

    # Campos del formulario usando los mapeos definidos
    user_selections['cap-shape'] = st.selectbox("Forma del sombrero", list(map_cap_shape.keys()))
    user_selections['cap-surface'] = st.selectbox("Superficie del sombrero", list(map_cap_surface.keys()))
    user_selections['cap-color'] = st.selectbox("Color del sombrero", list(map_cap_color.keys()))
    user_selections['bruises'] = st.selectbox("¿Se forman magulladuras al tocarla?", list(map_bruises.keys()))
    user_selections['gill-color'] = st.selectbox("Color de las láminas", list(map_gill_color.keys()))
    user_selections['stalk-shape'] = st.selectbox("Forma del tallo", list(map_stalk_shape.keys()))
    user_selections['stalk-surface-above-ring'] = st.selectbox("Superficie del tallo arriba del anillo", list(map_stalk_surface_above_ring.keys()))
    user_selections['stalk-surface-below-ring'] = st.selectbox("Superficie del tallo debajo del anillo", list(map_stalk_surface_below_ring.keys()))
    user_selections['stalk-color-above-ring'] = st.selectbox("Color del tallo arriba del anillo", list(map_stalk_color.keys())) # Usamos map_stalk_color genérico
    user_selections['stalk-color-below-ring'] = st.selectbox("Color del tallo debajo del anillo", list(map_stalk_color.keys())) # Usamos map_stalk_color genérico
    user_selections['veil-color'] = st.selectbox("Color del velo", list(map_veil_color.keys()))
    user_selections['ring-number'] = st.selectbox("Número de anillos", list(map_ring_number.keys()))
    user_selections['ring-type'] = st.selectbox("Tipo de anillo", list(map_ring_type.keys()))
    user_selections['population'] = st.selectbox("Población", list(map_population.keys()))
    user_selections['habitat'] = st.selectbox("Hábitat", list(map_habitat.keys()))

    if st.button("Clasificar Seta"):
        # Convertimos la selección del usuario (descripciones) a los valores codificados (letras)
        input_data_codes = {}
        input_data_codes['cap-shape'] = map_cap_shape[user_selections['cap-shape']]
        input_data_codes['cap-surface'] = map_cap_surface[user_selections['cap-surface']]
        input_data_codes['cap-color'] = map_cap_color[user_selections['cap-color']]
        input_data_codes['bruises'] = map_bruises[user_selections['bruises']]
        input_data_codes['gill-color'] = map_gill_color[user_selections['gill-color']]
        input_data_codes['stalk-shape'] = map_stalk_shape[user_selections['stalk-shape']]
        input_data_codes['stalk-surface-above-ring'] = map_stalk_surface_above_ring[user_selections['stalk-surface-above-ring']]
        input_data_codes['stalk-surface-below-ring'] = map_stalk_surface_below_ring[user_selections['stalk-surface-below-ring']]
        input_data_codes['stalk-color-above-ring'] = map_stalk_color[user_selections['stalk-color-above-ring']]
        input_data_codes['stalk-color-below-ring'] = map_stalk_color[user_selections['stalk-color-below-ring']]
        input_data_codes['veil-color'] = map_veil_color[user_selections['veil-color']]
        input_data_codes['ring-number'] = map_ring_number[user_selections['ring-number']]
        input_data_codes['ring-type'] = map_ring_type[user_selections['ring-type']]
        input_data_codes['population'] = map_population[user_selections['population']]
        input_data_codes['habitat'] = map_habitat[user_selections['habitat']]

        print("DEBUG: Contenido de input_data_codes justo antes de crear el DataFrame:", input_data_codes) 

        # Crear un DataFrame con la entrada del usuario (usando los códigos)
        input_df = pd.DataFrame([input_data_codes])

        # Aplicar One-Hot Encoding a la entrada del usuario.
        # drop_first=True debe coincidir con cómo se entrenó el modelo.
        input_encoded = pd.get_dummies(input_df, columns=streamlit_features)

        # Crear un DataFrame final con todas las columnas OHE esperadas por el modelo,
        # y rellenar con los valores de la entrada del usuario.
        final_input_df = pd.DataFrame(0, index=[0], columns=ohe_columns)

        for col in input_encoded.columns:
            if col in final_input_df.columns:
                final_input_df[col] = input_encoded[col].iloc[0]

        # Asegurarse de que el orden de las columnas sea el que el modelo espera
        final_input_df = final_input_df[ohe_columns]

        # --- Realizar la Predicción ---
        try:
            prediction_encoded = model.predict(final_input_df)
            prediction_label = label_encoder.inverse_transform(prediction_encoded)

            # --- Mostrar el Resultado ---
            st.subheader("Resultado de la Clasificación:")
            if prediction_label[0] == 'p':
                st.error("¡CUIDADO! Esta seta es muy probablemente **VENENOSA**.")
            else:
                st.success("¡Buenas noticias! Esta seta es muy probablemente **COMESTIBLE**.")

            st.markdown("---")
            st.subheader("Más Información:")
            st.markdown("""
            Nuestro modelo de Árbol de Decisión analizó las características que proporcionaste y siguió una serie de pasos lógicos para llegar a esta predicción. Está diseñado para ser transparente y seguro.

            **¡Recuerda siempre!** La identificación de setas silvestres para consumo debe ser realizada **SIEMPRE por un experto micólogo**. Esta aplicación es una herramienta educativa y de apoyo, no una sustitución del juicio profesional.
            """)
            st.info("Para cualquier duda, no consumas la seta.")

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
            st.error("Asegúrate de que los datos de entrada son válidos y de que las columnas OHE coinciden con las del modelo.")

# --- PÁGINA 3: Predicción por Archivo ---
with tabs[2]:
    st.header("📂 Subir Archivo para Predicción en Lote")
    st.markdown("Necesitas que el archivo contenga las siguientes columnas, escritas exactamente igual (respetando minúsculas y guiones), en el encabezado de tu CSV: "
            "`cap-shape`, `cap-surface`, `cap-color`, `bruises`, `gill-color`, "
            "`stalk-shape`, `stalk-surface-above-ring`, `stalk-surface-below-ring`, "
            "`stalk-color-above-ring`, `stalk-color-below-ring`, `veil-color`, "
            "`ring-number`, `ring-type`, `population`, `habitat`."
            "\n\nAdemás, recuerda que los valores dentro de estas columnas deben ser los códigos de una sola letra "
            "(como `x`, `s`, `n`, `t`, `k`, etc.), no las descripciones largas (como 'Convexo', 'Liso', 'Marrón').")

    # Definir las columnas esperadas para el CSV de entrada
    expected_csv_columns = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'gill-color',
        'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
        'ring-number', 'ring-type', 'population', 'habitat'
    ]

    example = pd.DataFrame(columns=expected_csv_columns)
    st.download_button("Descargar plantilla CSV", example.to_csv(index=False).encode('utf-8'), "plantilla_setas.csv", mime="text/csv")

    file = st.file_uploader("Carga tu archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # Verificar que las columnas del CSV subido coincidan con las esperadas
        if not all(col in df.columns for col in expected_csv_columns):
            st.error("El archivo CSV subido no contiene todas las columnas esperadas o sus nombres no coinciden. Por favor, usa la plantilla.")
            st.stop()

        # Asegurarse de que el DataFrame de entrada solo contenga las columnas relevantes y en el orden correcto
        df_input = df[expected_csv_columns].copy()

        # Aplicar los mapeos a las columnas categóricas
        for col, map_dict in {
            'cap-shape': map_cap_shape,
            'cap-surface': map_cap_surface,
            'cap-color': map_cap_color,
            'bruises': map_bruises,
            'gill-color': map_gill_color,
            'stalk-shape': map_stalk_shape,
            'stalk-surface-above-ring': map_stalk_surface_above_ring,
            'stalk-surface-below-ring': map_stalk_surface_below_ring,
            'stalk-color-above-ring': map_stalk_color, # Usamos map_stalk_color genérico
            'stalk-color-below-ring': map_stalk_color, # Usamos map_stalk_color genérico
            'veil-color': map_veil_color,
            'ring-number': map_ring_number,
            'ring-type': map_ring_type,
            'population': map_population,
            'habitat': map_habitat
        }.items():
            # Invertir el mapeo para ir de descripción a código
            reversed_map = {v: k for k, v in map_dict.items()}
            # Verificar si los valores en el CSV subido son descripciones completas
            # y mapearlos a los códigos si es necesario.
            # Asumimos que el CSV de entrada podría tener los códigos directamente o las descripciones.
            # Para mayor robustez, se recomienda que el CSV use los códigos directamente.
            # Si el CSV usa descripciones completas, necesitaríamos un paso extra de mapeo.
            # Por simplicidad aquí, asumimos que el CSV de plantilla se llenará con los códigos.
            # Si el usuario introduce las descripciones completas en la plantilla,
            # esta parte necesitaría un mapeo inverso antes del get_dummies.
            # Por ahora, la plantilla invita a usar los códigos directamente si el modelo los espera.
            # Nota: La plantilla que descargas genera headers pero no los valores,
            # así que el usuario debería saber qué códigos usar.
            # Para hacerlo más fácil, podríamos pedir que la plantilla se llene con descripciones y hacer el mapeo inverso.
            pass # No necesitamos mapeo aquí si pd.get_dummies trabaja directamente con las columnas.


        df_encoded = pd.get_dummies(df_input, columns=expected_csv_columns, drop_first=True)

        # Alinear las columnas con las que el modelo fue entrenado
        for col in ohe_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[ohe_columns] # ordenar columnas

        # Predicción
        predicciones = model.predict(df_encoded)
        resultados = label_encoder.inverse_transform(predicciones)
        df['predicción'] = resultados # Añadir la columna de predicción al DataFrame original
        st.success("Predicciones realizadas:")
        st.dataframe(df)

        st.download_button("Descargar resultados", df.to_csv(index=False).encode('utf-8'), "setas_con_predicciones.csv", mime="text/csv")

# --- PÁGINA 4: Contenido Descargable ---
with tabs[3]:
    st.header("📄 Descarga de Informe del Proyecto")
    st.markdown("Aquí puedes descargar el informe detallado de este proyecto de Machine Learning aplicado a la identificación de setas.")

    # Asegúrate de que la ruta a tu PDF es correcta
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'Informe Identificación de Setas.pdf')
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Descargar informe del proyecto",
                data=file,
                file_name="Informe Identificación de Setas.pdf",
                mime="application/pdf"
            )
    else:
        st.warning(f"Informe del proyecto no encontrado en la ruta: {pdf_path}")

# --- PÁGINA 5: Próximos Pasos ---
with tabs[4]:
    st.header("🚀 Próximos Pasos")
    st.write("---")
    st.markdown("""
## 🔮 El Futuro de la Identificación de Setas

Nuestro proyecto no se detiene aquí. Lo que habéis visto hoy es solo el comienzo de una herramienta que aspira a convertirse en un referente dentro del mundo de la micología digital. Estos son los próximos pasos que guiarán la evolución de **FungiScan**:

### 🧪 1. Aplicaciones Médicas y Terapéuticas
Queremos explorar en profundidad el potencial medicinal de las setas, investigando sus propiedades curativas, inmunológicas y terapéuticas. El objetivo es integrar una base de datos científica que permita identificar especies con posibles beneficios para la salud humana y su uso en medicina alternativa o farmacología.

### 📸 2. Identificación Automática por Imagen
Una de nuestras mayores ambiciones es permitir la identificación de setas simplemente a partir de una fotografía. Estamos trabajando para incorporar modelos de visión por computadora que analicen la imagen de una seta y, en segundos, indiquen su especie, toxicidad y otras características clave. Esto hará la herramienta mucho más accesible para usuarios sin conocimientos técnicos.

### 🍽️ 3. Información Ampliada y Personalizada
No nos limitaremos a decirte si una seta es comestible o venenosa. En futuras versiones de la app también podrás saber:
- Si es alucinógena o tóxica.
- Cómo cocinarla o prepararla (en caso de ser apta para consumo).
- Qué otras aplicaciones culturales, medicinales o ecológicas puede tener.
- Curiosidades e historia detrás de cada especie.

### 📱 4. Una Herramienta de Bolsillo
**FungiScan** aspira a ser una app móvil que puedas llevar contigo al bosque. Ya sea para excursionistas, micólogos, amantes de la naturaleza o curiosos, el objetivo es proporcionar una experiencia rica, útil y educativa desde cualquier lugar.
""")

    st.write("---")
    st.markdown ("""
                Si quereis saber más sobre el proceso de creación de la aplicación podeis hecharle un vistado a la presentación:
                🚀 https://prezi.com/view/gh5B808tN8uloI4wx4Ng/
                """)


    st.write("---")
    st.subheader("💡 ¿Tienes ideas o sugerencias?")

    feedback = st.text_area("Cuéntanos cómo mejorar la app o qué te gustaría ver en el futuro:")

    if st.button("Enviar comentario"):
        if feedback.strip():
            try:
                data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

                feedback_file_path = os.path.join(data_folder, "comentarios.txt")
                with open(feedback_file_path, "a", encoding="utf-8") as f:
                    # Añade el separador "---" más explícitamente y con saltos de línea para facilitar la lectura
                    f.write(feedback.strip() + "\n\n---\n\n") # Añadir saltos de línea extra
                st.success("¡Gracias por tu sugerencia! 🍄 La tendremos muy en cuenta.")
            except PermissionError:
                st.error("No tengo permiso para guardar los comentarios. Revisa los permisos de la carpeta 'data'.")
            except Exception as e:
                st.error(f"Ocurrió un error al guardar el comentario: {e}")
        else:
            st.warning("Por favor, escribe algo antes de enviar.")

    comentarios_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'comentarios.txt')
    if os.path.exists(comentarios_path):
        st.write("### 📬 Comentarios recibidos:")
        try:
            with open(comentarios_path, "r", encoding="utf-8") as f:
                comentarios_raw = f.read()

            comentarios_list = [c.strip() for c in comentarios_raw.split("---") if c.strip()]

            if comentarios_list:
                for i, comentario in enumerate(comentarios_list):
                    # Usamos st.info para cada comentario para un estilo más agradable
                    st.info(f"**Comentario {i+1}:**\n\n{comentario}")
            else:
                st.info("Aún no hay comentarios. ¡Sé el primero en dejar una sugerencia!")
            

        except Exception as e:
            st.error(f"No se pudieron cargar los comentarios: {e}")
