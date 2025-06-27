#Analyze titanic dataset
#1. Linear regression
#2. SVM
#3. naive bayes
#4. Kmeans
#5. Decision Tree
#6. Random forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
df= pd.read_csv('Titanic-Dataset.csv')
X= df[["Pclass","Age","Sex","SibSp","Parch","Fare"]]
y= df[["Survived"]]

X['Age']=X['Age'].fillna(X['Age'].mean())

le=LabelEncoder()
X['Sex']=le.fit_transform(X['Sex'])

joblib.dump(le,"label_encoder.pkl")

#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Dictionary of Model for model setup
model_dic={'LogisticRegression':LogisticRegression(),
            'SVC':SVC(),
            'Naive Bayes':GaussianNB(),
            'K Neighbors':KNeighborsClassifier(),
            'Decision Trees':DecisionTreeClassifier(),
            'Random Forest':RandomForestClassifier()}

results=[]
for name,model in model_dic.items():
    model.fit(X_train,y_train)
    if(name=="LogisticRegression"):
        joblib.dump(model,"Logistic_regression.pkl")
    y_pred=model.predict(X_test)

    #print("Classification report of all the maching learning algorithm")
    cm=confusion_matrix(y_test,y_pred)
    #print(cm)
    #print(classification_report(y_test,y_pred))
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    results.append({"model":name,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"f1_score":f1})
    #Visualize confusion matrix
    # plt.figure(figsize=(5,4))
    # sns.heatmap(cm,annot=True)
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.show()


#Summary of the Model
results_df=pd.DataFrame(results)
print("Summary of all the models")
print(results_df)
#Visualize the result
# plt.figure(figsize=(12,8))
# results_df.set_index("model")[["Accuracy","Precision","Recall","f1_score"]].plot(kind="bar")
# plt.title("Visualization of model performance")
# plt.xlabel("Model")
# plt.ylabel("Score")
# plt.show()