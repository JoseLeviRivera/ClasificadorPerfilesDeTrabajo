import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


"""
Esta función evalúa el rendimiento de un modelo mediante métricas como precisión, matriz de confusión 
y un informe de clasificación. Los resultados se imprimen en la consola y se guardan en un archivo JSON
especificado por el parámetro save_path.
Parámetros:
    - algorithm (str): El nombre del algoritmo o modelo utilizado para generar el informe.
    - y_test (similar a matriz): Las etiquetas reales del conjunto de prueba.
    - y_pred (similar a matriz): Las etiquetas predichas por el modelo para el conjunto de prueba.
    - save_path (str): La ruta del archivo para guardar los resultados del informe en formato JSON.
Resultados:
    Imprime en la consola la precisión del modelo, la matriz de confusión y el informe de clasificación.
    Guarda los resultados en un archivo JSON con la estructura especificada.
"""
def generate_report(algoritm, y_test, y_pred, save_path):
    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Guardar los resultados en un archivo JSON
    results = {
        "algoritm": algoritm,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": classification_rep
    }

    # Intentar cargar el archivo JSON existente
    try:
        with open(save_path, "r") as json_file:
            existing_results = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Si el archivo no existe o no es un JSON válido, iniciar una lista vacía
        existing_results = []

    # Agregar el nuevo resultado a la lista existente
    existing_results.append(results)

    # Escribir la lista actualizada en el archivo JSON
    with open(save_path, "w") as json_file:
        json.dump(existing_results, json_file, indent=2)
