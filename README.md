# 🍄 FungiScan – Identificación de Setas Comestibles y Venenosas

## 🧩 Contexto del Proyecto

> FungiScan es una aplicación desarrollada para ayudar a identificar setas como comestibles o venenosas usando un modelo de Machine Learning basado en un Árbol de Decisión.  
La herramienta está pensada para usuarios sin conocimientos avanzados de micología, enfocándose en características fácilmente observables para minimizar riesgos.

> **Objetivo:**  
> Crear un clasificador confiable y explicable que ayude a la toma de decisiones segura sobre la recolección y consumo de setas.

> [!NOTE]  
> Este proyecto fue desarrollado como parte de un Bootcamp de Análisis de Datos, replicando un caso real de análisis y modelado.

---

<details>
<summary>📦 <strong>Resumen del Dataset</strong></summary>

El dataset utilizado es un conjunto clásico para identificación de hongos con características como:

- Forma y color del sombrero.
- Color y forma de las láminas.
- Características del tallo (anillos, superficie).
- Hábitat donde se encontró la seta.

Se eliminaron variables con “fuga de datos” (olor, color de esporas) para que el modelo sea más seguro y realista.

</details>

---

<details>
<summary>🧹 <strong>Limpieza y Preparación de Datos</strong></summary>

Para asegurar la calidad del modelo se aplicaron los siguientes pasos:

- Eliminación de variables poco fiables o difíciles de observar por usuarios.
- Codificación One-Hot de variables categóricas.
- Revisión y tratamiento de datos faltantes o inconsistentes.

> Se priorizó la usabilidad y seguridad, evitando características que requieran conocimientos técnicos avanzados.

</details>

---

<details>
<summary>📊 <strong>Modelo y Entrenamiento</strong></summary>

- Algoritmo: Árbol de Decisión, por su transparencia y capacidad explicativa.
- Objetivo: Minimizar falsos negativos (clasificar venenosas como comestibles).
- Optimización con GridSearchCV para mejorar hiperparámetros.
- Validación con métricas de precisión, recall y matriz de confusión.

</details>

---

<details>
<summary>📈 <strong>Resultados Clave</strong></summary>

- Alta precisión en clasificación con enfoque en seguridad.
- El modelo evita usar características subjetivas o poco fiables.
- Proporciona explicaciones claras para cada predicción.

</details>

---

<details>
<summary>🧭 <strong>Recomendaciones y Uso</strong></summary>

- Usar la app como herramienta de apoyo, **no como sustituto de un experto micólogo**.
- Ingresar características observables con cuidado para evitar errores.
- Consultar siempre con especialistas ante dudas o setas desconocidas.
- Ampliar el dataset y seguir refinando el modelo para mayor robustez.

</details>

---

## 🚀 Aplicación Web con Streamlit

Se desarrolló una aplicación interactiva con **Streamlit** donde el usuario puede:

- Ingresar características manualmente para obtener una predicción.
- Visualizar la explicación del resultado y la seguridad del modelo.
  
👉 [Prueba la app online aquí](https://fungiscan-eqh8bxu2ysfwrxoximq38a.streamlit.app/)
---

<details>
  <summary>
    <h2>👤 Autora</h2>
  </summary>

[![Rocío](https://img.shields.io/badge/@JimenezRoDA-GitHub-181717?logo=github&style=flat-square)](https://github.com/JimenezRoDA)  

---

![Python](https://img.shields.io/badge/Python-3.12.7-blue?logo=python)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Status](https://img.shields.io/badge/Status-Finished-brightgreen)

[🔝 Volver arriba](#-fungiscan--identificación-de-setas-comestibles-y-venenosas)
</details>
