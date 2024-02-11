from sklearn.model_selection import train_test_split
import pandas as pd

"""
Esta función carga un conjunto de datos desde un archivo CSV ubicado en la ruta especificada (path_to_data). 
Luego, realiza un mapeo de habilidades específicas a números según el diccionario skills_mapping. 
La función devuelve el conjunto de datos modificado.
Parámetros:
    - path_to_data (str): La ruta al archivo CSV que contiene los datos a cargar.
Resultados:
La función devuelve un DataFrame de pandas que representa el conjunto de datos, con las habilidades mapeadas a números según el diccionario proporcionado.
"""
def prepare_data(path_to_data):
    data = pd.read_csv(path_to_data)
    skills_mapping = {
        "Javascript": 1, "HTML/CSS": 2, "Photoshop": 3, "GitHub": 4, "Figma": 5, "Node.js": 6,
        "Angular": 7, "React": 8, "Python": 9, "R": 10, "Tensorflow": 11, "Deep Learning": 12, "Ansible": 13,
        "Pytorch": 14, "Machine Learning": 15, "C/C++": 16, "Java": 17, "MYSQL": 18, "Oracle": 19,
        "Linux": 20, "BASH/SHELL": 21, "Cisco Packet tracer": 22, "Wire Shark": 23,
    }
    data.replace(skills_mapping, inplace=True)
    return data

"""
Esta función toma matrices de características (X) y etiquetas de destino (y) y realiza una división en conjuntos de entrenamiento
y prueba utilizando la función train_test_split de scikit-learn. Devuelve un diccionario con las siguientes
claves: 'x_train', 'x_test', 'y_train', 'y_test', que contienen las matrices de características y etiquetas de entrenamiento y 
prueba, respectivamente.
Parámetros:

    - X (similar a matriz): La matriz de características.
    - y (similar a matriz): Las etiquetas de destino.
    test_size (float o int): Representa la proporción del conjunto de datos que se asignará al conjunto de prueba 
    o el número absoluto de muestras en el conjunto de prueba.
    random_state (int o None): Semilla para garantizar reproducibilidad en la división de datos.

Resultados:

La función devuelve un diccionario que contiene las matrices de características y etiquetas de entrenamiento y prueba.
"""
def create_train_test_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return {'x_train': X_train, 'x_test': X_test,
            'y_train': y_train, 'y_test': y_test}
