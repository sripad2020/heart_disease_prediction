import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('heart_disease_data.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
col=data.columns.values
for i in col:
    sn.boxplot(data[i])
    plt.show()
for i in col:
    for j in col:
        plt.plot(data[i],marker='o',color='yellow',label=f'{i}')
        plt.plot(data[j],marker='o',color='red',label=f'{j}')
        plt.title(f'{i} vs {j}')
        plt.legend()
        plt.show()
data['z-score']=(data.trestbps-data.trestbps.mean())/(data.trestbps.std())
df=data[(data['z-score'] > -3 ) & (data['z-score'] <3)]
q1=df.trestbps.quantile(0.25)
q3=df.trestbps.quantile(0.75)
iqr=q3-q1
upper=q3+1.5*iqr
lower=q1-1.5*iqr
df=df[(df.trestbps < upper)& (df.trestbps > lower)]
colm=df.columns.values
sn.pairplot(df)
plt.show()
x=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=5)
tree.fit(x_train,y_train)
print(tree.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))
from keras.models import  Sequential
from keras.layers import Dense
import keras.activations,keras.losses
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=1,activation=keras.activations.sigmoid))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_train,y_train,epochs=450,batch_size=20,validation_split=0.35,verbose=True)
print(hist.history)
pred=models.predict(x_test)

plt.scatter(y_test,pred,color='green',edgecolors='red')
plt.xlabel('y_test')
plt.ylabel('predicted')
plt.legend()
plt.show()

plt.plot(hist.history['accuracy'],marker='o',label='accuracy',color='red')
plt.plot(hist.history['val_accuracy'],maker='o',label='val_accuracy',color='blue')
plt.title('Training and Validation accuracy ')
plt.legend()
plt.show()