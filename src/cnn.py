import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

def get_input(train_pkl, test_pkl, hold_out_pkl):
    '''
    inputs:
        Pickle files created using the preprocess script (train_pkl, test_pkl, hold_out_pkl)
    outputs:
        Train, test, hold_out sample sets in the required shaped for the tensorflow model (128 x 256)
        (X train, y train, X test, y test, X hold-out, y hold-out)
    '''
    # unpickle the pickles from the preprocessed script
    df_tr = pd.read_pickle(train_pkl)
    df_ts = pd.read_pickle(test_pkl)
    df_ho = pd.read_pickle(hold_out_pkl)
    
    # encode the target column
    encoder = LabelEncoder()
    encoder.fit(df_tr['lang'])
    labels = encoder.classes_
    y_tr = encoder.transform(df_tr['lang'])
    y_ts = encoder.transform(df_ts['lang'])
    y_ho = encoder.transform(df_ho['lang'])
    
    # turn the feature matrix into tensorflow-friendly format
    X_tr = np.array(df_tr['features'].tolist())
    X_tr = X_tr.reshape(X_tr.shape[0],128,256,1)
    X_ts = np.array(df_ts['features'].tolist())
    X_ts = X_ts.reshape(X_ts.shape[0],128,256,1)
    X_ho = np.array(df_ho['features'].tolist())
    X_ho = X_ho.reshape(X_ho.shape[0],128,256,1)
    
    return X_tr, y_tr, X_ts, y_ts, X_ho, y_ho


def construct_model():
    '''
    Create the tensoflow CNN model
    Inputs:
        None
    Outputs:
        A compiled tensorflow CNN model
    '''
    
    # Construct the layers
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128, 256, 1), padding = 'same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), activation='relu', padding = 'same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (5, 5), activation='relu', padding = 'same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    tf.keras.losses.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy')

    # compile it
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def train_model(model, X_tr, y_tr, X_ts, y_ts):
    '''
    Fit the tensorflow CNN model
    
    Inputs:
        Compiled tensorflow model, feature matrix, training label, and the coresponding validation set (model, X_tr, y_tr, X_ts, y_ts)
    Output:
        Fitted model
    '''
    
    EPOCHS = 100
    checkpoint_filepath = '../temp_checkpoint/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.fit(X_tr, y_tr, epochs=EPOCHS, validation_data=(X_ts, y_ts), callbacks=[model_checkpoint_callback])
    return model

def validate_model(model, X_ho, y_ho):
    '''
    Print the validation metrics using the hold_out sets
    
    Input:
        Fitted model, hold_out feature matrix, hold_out labels
    Output:
        validation loss, validation accuracy
    '''
    
    checkpoint_filepath = '../temp_checkpoint/'
    model.load_weights(checkpoint_filepath)
    val_loss, val_acc = model.evaluate(X_ho, y_ho, verbose=1)
    return val_loss, val_acc

if __name__ == "__main__":
    train_pkl = '../pickles/train.pkl'
    test_pkl = '../pickles/test.pkl'
    hold_out_pkl = '../pickles/hold_out.pkl'
    X_tr, y_tr, X_ts, y_ts, X_ho, y_ho = get_input(train_pkl, test_pkl, hold_out_pkl)
    
    model = construct_model()
    model = train_model(model, X_tr, y_tr, X_ts, y_ts)
    
    val_loss, val_acc = validate_model(model, X_ho, y_ho)
    
    print(f'Validation loss of the model against the hold-out set:{val_loss}')
    print(f'Validation accuracy of the model against the hold-out set:{val_acc}')