import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#Load Dataset
df = pd.read_csv("C:\\Users\\PRIYANKA\\Downloads\\archive (2)\\CardioGoodFitness.csv")

#convert columns into numbers
df = pd.get_dummies(df, drop_first=True)

#Select Features
x = df[["Education","Age","Usage", "Fitness", "Miles"]]
y = df["Income"]

#Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train Data
model = LinearRegression()
model.fit(x_train, y_train)

#Save Model 
with open ("model.pkl", "wb") as f:
    pickle.dump(model, f)
    
print ("Model train and saved!")

