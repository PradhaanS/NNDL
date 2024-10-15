import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

breast_cancer_sklearn = load_breast_cancer()

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

#print(breast_cancer_sklearn)

breast_cancer_df = pd.DataFrame(data=breast_cancer_sklearn.data,
                           columns=breast_cancer_sklearn.feature_names)

breast_cancer_df['target'] = breast_cancer_sklearn.target

#print(breast_cancer_df)

x = breast_cancer_df.iloc[: , :-1]
y = breast_cancer_df.iloc[: , -1]

#print(x)

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.3)

#print(x_train)

mlp = MLPClassifier(hidden_layer_sizes =(5,3),activation='relu',solver='lbfgs' )

mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
