# Packages
import os
import joblib

from core.classifiers.Bayes.model_trainer import run_model_training
from core.util.cross_validations import get_cross_validation
from core.util.selector_features import generate_features_selector
import core.util.data_processor as dp

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# 1.Prepare the data
data = dp.prepare_data("data/Student Placement.csv")

# 2.Create train - test split
train_test_data = dp.create_train_test_data(data.drop('Profile', axis=1), data['Profile'], 0.33, 2021, )

# 3.Run training
model = run_model_training(train_test_data['x_train'], train_test_data['x_test'],
                           train_test_data['y_train'], train_test_data['y_test'],
                           "reports/model_results.json"
                           )
# 4. Cross Validations
get_cross_validation("GaussianNB", model, train_test_data['x_train'], train_test_data['y_train'], 4,
                     "reports/cross_validations.json")

# 5. Features Selections
# print(generate_features_selector(model, train_test_data['x_train'], train_test_data['y_train'], 10, "forward"))

# 6. Save model machine learning
joblib.dump(model, 'regression/model.pkl')
