import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


def optimize_model_with_gridsearch(model_base, param_grid, X_train, y_train, model_name="Modelo", cv=5, 
scoring='f1_macro', n_jobs=-1, verbose=1):
    # Configurar y Ejecutar GridSearchCV
    grid_search = GridSearchCV(model_base, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)

    # Entrenar el modelo con los datos de entrenamiento
    grid_search.fit(X_train, y_train)

    # Mostrar los mejores hiperparámetros encontrados
    best_params = grid_search.best_params_
    print(f"\nMejores hiperparámetros encontrados para {model_name}: {best_params}")

    # Accedemos al modelo mejor entrenado (el mejor estimador de Grid Search)
    best_model = grid_search.best_estimator_

    return best_model, best_params

def evaluate_and_report_model(model, X_train, y_train, X_test, y_test, label_encoder, model_name="Modelo"):
    print(f"\n--- Evaluación del {model_name} ---")

    # Predicciones en el conjunto de ENTRENAMIENTO
    y_train_pred = model.predict(X_train)
    print(f"\nRendimiento en el Conjunto de ENTRENAMIENTO ({model_name}):")
    print("Reporte de Clasificación (Entrenamiento):\n",
          classification_report(y_train, y_train_pred, target_names=label_encoder.classes_))

    # Predicciones en el conjunto de PRUEBA
    y_test_pred = model.predict(X_test)
    print(f"\nRendimiento en el Conjunto de PRUEBA ({model_name}):")
    print("Reporte de Clasificación (Prueba):\n",
          classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))


    # Matriz de Confusión para el conjunto de prueba
    print(f"\nMatriz de Confusión ({model_name} - Conjunto de Prueba):")
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Matriz de Confusión para {model_name}')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Verdadera')
    plt.show()
