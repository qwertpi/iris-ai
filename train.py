import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.utils import to_categorical

#loads and formats data
data=np.loadtxt("iris.csv", delimiter=",")
X = data[:,0:4]
Y = data[:,4]
Y=to_categorical(Y)

#train test splits
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)

#model architecture
model = Sequential()
model.add(Dense(32,input_dim=4,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(4,activation='softmax'))

#custom callback to print learning rate
class MyCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))

#reduces learning rate when val_loss stagnates         
ReduceLR=callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.9, verbose=0, cooldown=100, min_lr=0.0001)
#saves the best model based on val_loss with the name of the val_acc rounded to 2 decimal places
savebest=callbacks.ModelCheckpoint(filepath="checkpoint-{val_acc:.2f}.h5", monitor="val_loss", save_best_only=True)
PrintLR=MyCallback()
callbacks_list=[ReduceLR,PrintLR,savebest]

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#trains the model for 10,000 epochs but it should probably be stopped earlier by a keyboard interrupt
model.fit(x_train, y_train, epochs=10000, batch_size=64, validation_data=(x_test, y_test),callbacks=callbacks_list)

from keras import backend as K
K.clear_session()
