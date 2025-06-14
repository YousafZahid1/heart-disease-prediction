import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
import tkinter as tk


# List of fields
fields = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
          'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']


page = tk.Tk()
page.geometry("500x700")
page.title("Taking Parameters")


tk.Label(page, text="Provide the values", font=("Bold", 19)).pack()


entries = {}


for field in fields:
    label = tk.Label(page, text=field, font=("Arial", 12))
    label.pack()
    entry = tk.Entry(page)
    entry.pack()
    entries[field] = entry


def btn():
    # Get values and convert int fields
    Age = int(entries['Age'].get())
    Sex = entries['Sex'].get()
    ChestPainType = entries['ChestPainType'].get()
    RestingBP = int(entries['RestingBP'].get())
    Cholesterol = int(entries['Cholesterol'].get())
    FastingBS = int(entries['FastingBS'].get())
    RestingECG = entries['RestingECG'].get()
    MaxHR = int(entries['MaxHR'].get())
    ExerciseAngina = entries['ExerciseAngina'].get()
    Oldpeak = int(entries['Oldpeak'].get())
    ST_Slope = entries['ST_Slope'].get()
    read_file = pd.read_csv("heart.csv.xls")
    print(read_file.head())


    model = LogisticRegression(max_iter=1000)
    X=read_file[['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]
    X=pd.get_dummies(X)
    Y=read_file['HeartDisease']
    x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.4,random_state = 42)
    model.fit(x_train,y_train)

    #Data for predicting
    columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
            'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    pred_data = pd.DataFrame([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]], columns=columns)
    sample_encode = pd.get_dummies(pred_data)
    sample_encode = sample_encode.reindex(columns=X.columns, fill_value=0)
    print(model.predict(sample_encode))

    data1 = pd.get_dummies(read_file)
    if("0" in model.predict(sample_encode) or 0 in model.predict(sample_encode)):

        tk.Label(page,text="No Heart Disease :)").pack()
    else:
        tk.Label(page,text="Sorry you have a Heart Disease").pack()

tk.Button(page, text="Submit", command=btn).pack()

print(entries)



'''
'Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
           'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
'''



page.mainloop()