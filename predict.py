import pickle 
import numpy as np 
import pandas as pd

#Load model 
with open ("model.pkl", "rb") as f:
    model = pickle.load(f)
    
#Example input 
data = np.array([[25, 16, 3, 4, 100]])

prediction = model.predict(data)
print("Predicted Income:", prediction[0])