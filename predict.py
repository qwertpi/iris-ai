import numpy as np
from keras.models import load_model

model=load_model("model.h5")

sep_length=float(input("Enter the speal length in cm    "))
sep_width=float(input("Enter the speal width in cm    "))
pet_length=float(input("Enter the petal length in cm    "))
pet_width=float(input("Enter the petal width in cm    "))

data=np.array([[sep_length, sep_width, pet_length, pet_width]])
prediction=list(model.predict(data)[0])
index=prediction.index(sorted(prediction)[-1])

if index==1:
    print("Iris Setosa")
elif index==2:
    print("Iris Versicolour")
elif index==3:
    print("Iris Virginica")
else:
    print("Error! Please raise this on GitHub with the following information")
    print(data)
    print(prediction)
    print(model.predict(data))
    print("Searching for alternative prediction")
    index=sorted(prediction)[1]
    if index==1:
        print("Iris Setosa")
    elif index==2:
        print("Iris Versicolour")
    elif index==3:
        print("Iris Virginica")
