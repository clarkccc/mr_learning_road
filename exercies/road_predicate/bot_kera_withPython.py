import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
datax_path = 'X_train.csv'
datay_path = 'Y_train.csv'
x_train = pd.read_csv(datax_path)
y_train = pd.read_csv(datay_path)


n_num_layer1 = 256
n_num_layer2 = 256
n_num_layer3 = 128
n_num_layer4 = 64
n_num_layer5 = 64
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(n_num_layer1, activation=tf.nn.relu, kernel_initializer=keras.initializers.he_normal() ,input_shape=(x_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_num_layer2, activation=tf.nn.relu, kernel_initializer=keras.initializers.he_normal() ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_num_layer3, activation=tf.nn.relu, kernel_initializer=keras.initializers.he_normal() ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_num_layer4, activation=tf.nn.relu, kernel_initializer=keras.initializers.he_normal() ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_num_layer5, activation=tf.nn.relu, kernel_initializer=keras.initializers.he_normal() ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(y_train.shape[1], activation=tf.nn.relu)
    ])
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mse'])
    return model
model = create_model()
model.summary()

checkpoint_path = "my_model.h5"
bot_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weights_only=True, #只保存权重
                                                  verbose=1)
import concurrent.futures
with concurrent.futures.ProcessPoolExecutor() as executor:
    model.fit(x_train, y_train,  epochs = 2000, callbacks = [bot_callback])  #保存模型
    model.save(checkpoint_path);
# model.load_weights(checkpoint_path)
mix_path = 'test2.csv'
x_mix = pd.read_csv(mix_path)
def predict():
    for i in range(len(x_mix.index)-1):
        x_mix.iloc[i+1,100:150]=model.predict(np.array(x_mix.iloc[i]).reshape(-1,158))[0]
predict()
x1_50 = model.predict(x_mix)
x1_50 = pd.DataFrame(x1_50)
y_test_path = 'label_l2w.csv'
y_test = pd.read_csv(y_test_path)
loss,acc = model.evaluate(x_mix, y_test)
print("Restored model, accuracy: {:5.2f}%".format(acc)) #86.2
print(loss,acc)
