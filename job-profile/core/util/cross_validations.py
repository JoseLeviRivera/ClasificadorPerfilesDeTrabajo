import json
from sklearn.model_selection import KFold, cross_val_score


"""
Esta función realiza validación cruzada k-fold utilizando el algoritmo y el modelo proporcionados. Calcula el 
rendimiento para cada fold, imprime los resultados y guarda la información en un archivo JSON especificado por el parámetro save_path.
Parámetros:

    algorithm (str): El nombre del algoritmo o modelo utilizado en la validación cruzada.
    model (modelo de scikit-learn): El modelo a evaluar mediante validación cruzada.
    - X (similar a matriz): La matriz de características de los datos.
    - y (similar a matriz): Las etiquetas de destino de los datos.
    - k_folds (int): Número de divisiones (folds) para la validación cruzada.
    save_path (str): La ruta del archivo para guardar los resultados de la validación cruzada en formato JSON.

Resultados:
La función imprime en la consola el tipo de algoritmo, el rendimiento por fold y el rendimiento promedio. Además, guarda 
los resultados en un archivo JSON con la estructura especificada.

# Example
# get_cross_validation("GaussianNB", model, X_train, y_train, 4, "reports/cross_validations.json")

"""
def get_cross_validation(algorithm, model, X, y, k_folds, save_path):
    k_folds = KFold(n_splits=k_folds)
    scoresK = cross_val_score(model, X, y, cv=k_folds)

    # Convertir el array de NumPy a una lista antes de serializar
    new_result = {
        "Algorithm": algorithm,
        "Fold Performance": scoresK.tolist(),
        "Average performance": scoresK.mean().tolist()
    }

    # Intentar cargar el archivo JSON existente
    try:
        with open(save_path, "r") as json_file:
            existing_results = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Si el archivo no existe o no es un JSON válido, iniciar una lista vacía
        existing_results = []

    # Agregar el nuevo resultado a la lista existente
    existing_results.append(new_result)

    # Escribir la lista actualizada en el archivo JSON
    with open(save_path, "w") as json_file:
        json.dump(existing_results, json_file, indent=2)




"""
Esta función realiza validación cruzada k-fold utilizando el modelo proporcionado y calcula el rendimiento promedio y la desviación estándar. 
Imprime los resultados en la consola.
Parámetros:
    - modelo (modelo de scikit-learn): El modelo a evaluar mediante validación cruzada.
    - X (similar a matriz): La matriz de características de los datos.
    - y (similar a matriz): Las etiquetas de destino de los datos.
    - k_folds (int): Número de divisiones (folds) para la validación cruzada.
Resultados:
La función imprime en la consola el rendimiento por fold, el rendimiento promedio y la desviación estándar del rendimiento.
"""
def get_avg_models(modelo, X, y, k_folds):
    scoresS = cross_val_score(modelo, X, y, cv=k_folds)
    print("Desempeño por fold: ", scoresS)
    print("Promedio de desempeño: ", scoresS.mean())
    print("Con desviación estandar: ", scoresS.std())