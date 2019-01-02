import numpy as np
from keras.models import load_model

#loads the model
model=load_model("model.h5")

#gets the data we will be basing our prediction on
sep_length=float(input("Enter the speal length in cm    "))
sep_width=float(input("Enter the speal width in cm    "))
pet_length=float(input("Enter the petal length in cm    "))
pet_width=float(input("Enter the petal width in cm    "))

#puts the data into the required format
data=np.array([[sep_length, sep_width, pet_length, pet_width]])
#makes a prediction
#also turns it into a list
prediction=list(model.predict(data)[0])
#mimics argmax
index=prediction.index(sorted(prediction)[-1])

#tells the user which plant they have
if index==1:
    print("Iris Setosa")
elif index==2:
    print("Iris Versicolour")
elif index==3:
    print("Iris Virginica")
#in the event of a index 0 (no class) or anything else
else:
    #gives me debug info
    print("Error! Please raise this on GitHub with the following information")
    print(data)
    print(prediction)
    print(index)
    #uses the next most confident prediction
    print("Searching for alternative prediction")
    index=sorted(prediction)[1]
    if index==1:
        print("Iris Setosa")
    elif index==2:
        print("Iris Versicolour")
    elif index==3:
        print("Iris Virginica")
