import pickle
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA_SOURCE = "data\Training.csv"
DATA_TEST = "data\Testing.csv"
data = pd.read_csv(DATA_SOURCE).dropna(axis = 1)

disease_counts = data["prognosis"].value_counts()
df = pd.DataFrame({
    "Disease": disease_counts.index, "Counts": disease_counts.values})

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

svm_model = SVC()
svm_model.fit(X, y)

test_data = pd.read_csv(DATA_TEST).dropna(axis = 1)
test_X = test_data.iloc[:,:-1]
test_Y = encoder.transform(test_data.iloc[:,-1])

symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_map = {
    "symptom_index" : symptom_index, "prediction_classes" : encoder.classes_
}

def disease_prediction(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_map["symptom_index"])
    for symptom in symptoms:
        index = data_map["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)
    svm_prediction = data_map["prediction_classes"][svm_model.predict(input_data)[0]]
    return svm_prediction


symptoms = input("Enter comma seperated symptoms = ")
print(disease_prediction(symptoms))