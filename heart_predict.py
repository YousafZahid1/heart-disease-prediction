import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt


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
pred_data = pd.DataFrame([[49,'M','ASY',340,234,0,'Normal',140,'Y',1,'Flat']], columns=columns)
sample_encode = pd.get_dummies(pred_data)
sample_encode = sample_encode.reindex(columns=X.columns, fill_value=0)
print(model.predict(sample_encode))

data1 = pd.get_dummies(read_file)



#accuracy rate

print(accuracy_score(y_test,model.predict(sample_encode)))
# sns.pairplot(data1,hue='HeartDisease')
# plt.show()
