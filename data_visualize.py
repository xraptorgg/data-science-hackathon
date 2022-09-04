import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA_SOURCE = "data\Training.csv"
DATA_TEST = "data\Testing.csv"
data = pd.read_csv(DATA_SOURCE).dropna(axis = 1)

disease_counts = data["prognosis"].value_counts()
df = pd.DataFrame({
    "Disease": disease_counts.index, "Counts": disease_counts.values})


plt.figure(figsize = (12, 2))
sns.barplot(x = "Disease", y = "Counts", data = df)
plt.xticks(fontsize = 4.5, rotation = -60)
plt.show()

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print(f"Train sets : {X_train.shape}, {y_train.shape}")
print(f"Test sets : {X_test.shape}, {y_test.shape}")

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


svm_model = SVC()

score = cross_val_score(svm_model, X, y, cv = 10, n_jobs = -1, scoring = cv_scoring)

print("="*70)
print("Support Vector Classifier")
print(f"Scores = {score}")
print(f"Mean score = {np.mean(score)}")


svm_model.fit(X_train, y_train)
prediction = svm_model.predict(X_test)
print(f"Accuracy on train data by SVM Classifier = {accuracy_score(y_train, svm_model.predict(X_train)) * 100}")
print(f"Accuracy on test data by SVM Classifier = {accuracy_score(y_test, prediction) * 100}")
cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()


final_svm_model = SVC()
final_svm_model.fit(X, y)

test_data = pd.read_csv(DATA_TEST).dropna(axis = 1)

test_X = test_data.iloc[:,:-1]
test_Y = encoder.transform(test_data.iloc[:,-1])

prediction_svm = final_svm_model.predict(test_X)


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
    svm_prediction = data_map["prediction_classes"][final_svm_model.predict(input_data)[0]]
    return svm_prediction


print(disease_prediction("Itching,Skin Rash,Nodal Skin Eruptions"))
