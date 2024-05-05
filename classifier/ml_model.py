import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def load_data(file_path):
    data = np.load(file_path)
    X_train = data['train_images']
    y_train = data['train_labels']
    X_test = data['test_images']
    y_test = data['test_labels']
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    return X_train, X_test

def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, X_train, y_train, datagen, X_val, y_val, batch_size=32, epochs=10):
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                        validation_data=(X_val, y_val))
    return history

def evaluate_model(model, X_test, y_test):
    probabilities = model.predict(X_test)
    predictions = (probabilities > 0.5).astype('int32')
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return report, matrix

if __name__ == "__main__":
    file_path = 'C:/Users/perer/.medmnist/pneumoniamnist_128.npz'
    X_train, y_train, X_test, y_test = load_data(file_path)
    X_train, X_test = preprocess_data(X_train, X_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    model = create_model(X_train.shape[1:])
    history = train_model(model, X_train, y_train, datagen, X_val, y_val)
    report, matrix = evaluate_model(model, X_test, y_test)
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(matrix)
