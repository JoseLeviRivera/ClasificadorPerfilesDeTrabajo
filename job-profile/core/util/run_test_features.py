import os

import joblib

# Get customized functions from library
import core.util.data_processor as dp
import core.classifiers.Bayes.model_trainer as model_bayes
import core.classifiers.KNeighborsClassifier.model_trainer as model_knn
import core.classifiers.DecisionTreeClassifier.model_trainer as model_tree
import core.classifiers.SupportVectorMachine.model_trainer as model_svm

from core.util.cross_validations import get_cross_validation
from core.util.selector_features import generate_features_selector

os.makedirs('../../', exist_ok=True)

# 1.Prepare the data
data = dp.prepare_data("data/Student Placement.csv")


# 2.Create train - test split
train_test_data = dp.create_train_test_data(data.drop('Profile', axis=1), data['Profile'], 0.33, 2021)

# 3.Run training
model_bayes = model_bayes.run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                                             train_test_data['y_train'], train_test_data['y_test'],
                                             "reports/model_results.json"
                                             )
model_knn = model_knn.run_model_training(3, train_test_data['x_train'], train_test_data['x_test'],
                                         train_test_data['y_train'], train_test_data['y_test'],
                                         "reports/model_results.json"
                                         )

model_tree = model_tree.run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                                           train_test_data['y_train'], train_test_data['y_test'],
                                           "reports/model_results.json"
                                           )

model_svm = model_svm.run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                                         train_test_data['y_train'], train_test_data['y_test'],
                                         "reports/model_results.json"
                                         )

# Cross Validations
get_cross_validation("GaussianNB", model_bayes, train_test_data['x_train'], train_test_data['y_train'], 4,
                     "reports/cross_validations.json")
get_cross_validation("KNeighborsClassifier", model_knn, train_test_data['x_train'], train_test_data['y_train'], 4,
                     "reports/cross_validations.json")
get_cross_validation("DecisionTreeClassifier", model_tree, train_test_data['x_train'], train_test_data['y_train'], 4,
                     "reports/cross_validations.json")
get_cross_validation("SupportVectorMachine", model_svm, train_test_data['x_train'], train_test_data['y_train'], 4,
                     "reports/cross_validations.json")

# Definir las características
features = ["DSA","DBMS","OS","CN","Mathmetics","Aptitute","Comm","Problem Solving","Creative","Hackathons","Skill 1","Skill 2"]

print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 1,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 2,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 3,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 4,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 5,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 6,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 7,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 8,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 9,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 10,"forward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 11,"forward"))

print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 1,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 2,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 3,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 4,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 5,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 6,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 7,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 8,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 9,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 10,"backward"))
print("GaussianNB ",generate_features_selector(model_bayes, train_test_data['x_train'], train_test_data['y_train'], 11,"backward"))



print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 1,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 2,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 3,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 4,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 5,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 6,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 7,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 8,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 9,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 10,"forward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 11,"forward"))

print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 1,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 2,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 3,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 4,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 5,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 6,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 7,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 8,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 9,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 10,"backward"))
print("knn ",generate_features_selector(model_knn, train_test_data['x_train'], train_test_data['y_train'], 11,"backward"))


print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 1,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 2,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 3,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 4,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 5,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 6,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 7,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 8,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 9,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 10,"forward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 11,"forward"))

print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 1,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 2,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 3,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 4,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 5,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 6,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 7,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 8,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 9,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 10,"backward"))
print("DecisionTreeClassifier ",generate_features_selector(model_tree, train_test_data['x_train'], train_test_data['y_train'], 11,"backward"))


print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 1,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 2,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 3,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 4,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 5,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 6,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 7,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 8,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 9,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 10,"forward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 11,"forward"))

print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 1,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 2,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 3,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 4,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 5,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 6,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 7,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 8,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 9,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 10,"backward"))
print("SVM ",generate_features_selector(model_svm, train_test_data['x_train'], train_test_data['y_train'], 11,"backward"))


