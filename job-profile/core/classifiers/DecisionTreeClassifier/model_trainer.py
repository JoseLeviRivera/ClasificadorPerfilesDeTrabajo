from sklearn.tree import DecisionTreeClassifier
from core.util.models_reports import generate_report

"""
Esta función realiza el entrenamiento de un clasificador DecisionTreeClassifier
utilizando los datos de entrenamiento proporcionados (X_train, y_train). Luego evalúa 
el modelo entrenado en los datos de prueba (X_test, y_test) y genera un informe de clasificación.
Los resultados del entrenamiento y la evaluación del modelo se guardan en un archivo JSON 
especificado por el parámetro save_path.
Parámetros:
    - X_train (similar a matriz): La matriz de características de los datos de entrenamiento.
    - X_test (similar a matriz): La matriz de características de los datos de prueba.
    - y_train (similar a matriz): Las etiquetas de destino de los datos de entrenamiento.
    - y_test (similar a matriz): Las etiquetas de destino de los datos de prueba.
    - save_path (str, opcional): La ruta del archivo para guardar los resultados de la evaluación
     del modelo en formato JSON. El valor predeterminado es "model_results.json".
Devoluciones:
    gnb (GaussianNB): El clasificador Naive Bayes Gaussiano entrenado.
"""

def run_model_training(X_train, X_test, y_train, y_test, save_path="model_results.json"):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree.score(X_test, y_test)
    y_pred = tree.predict(X_test)
    generate_report("DecisionTreeClassifier", y_test, y_pred, save_path)
    return tree
