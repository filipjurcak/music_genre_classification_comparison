import numpy as np
import keras
import pandas as pd
from sklearn import neighbors
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def fit_and_print_results(model, x_train, y_train, x_test, y_test, y_test_label, enc, print_str):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_label = list(enc.inverse_transform(y_pred))

    print(confusion_matrix(y_test_label, y_pred_label))
    print("\n")
    print(classification_report(y_test_label, y_pred_label))

    print("Training set score for {} was: {}".format(print_str, model.score(x_train, y_train)))
    print("Testing  set score for {} was: {}\n".format(print_str, model.score(x_test, y_test)))


if __name__ == "__main__":
    test_size = 0.2
    k = 7  # for k-NN

    # for gtzan dataset
    data = np.array(shuffle(pd.read_csv('gtzan/data.csv')))
    # data = np.array(shuffle(pd.read_csv('gtzan/normalized_data.csv')))

    # melspectograms for gtzan dataset
    X_cnn = np.load('x_gtzan_npy.npy')
    y_cnn = np.load('y_gtzan_npy.npy')

    # for fma small dataset
    # data = np.array(shuffle(pd.read_csv('fma_small/data.csv')))
    # data = np.array(shuffle(pd.read_csv('fma_small/normalized_data.csv')))

    # melspectograms for fma_small dataset
    # X_cnn = np.load('x_fma_small.npy')
    # y_cnn = np.load('y_fma_small.npy')

    split = int(data.shape[0] * test_size)
    test = data[:split]
    train = data[split:]

    X_train = np.array(train[:, 1:-1], dtype=float)
    Y_train_label = train[:, -1]
    X_test = np.array(test[:, 1:-1], dtype=float)
    Y_test_label = test[:, -1]

    encoder = LabelEncoder()
    # encoding train labels
    encoder.fit(Y_train_label)
    Y_train = encoder.transform(Y_train_label)

    # encoding test labels
    encoder.fit(Y_test_label)
    Y_test = encoder.transform(Y_test_label)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # k-nn
    for weights in ['uniform', 'distance']:
        clf = neighbors.KNeighborsClassifier(k, weights=weights)
        fit_and_print_results(clf, X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, encoder,
                              "k-NN with {} wights".format(weights))

    # svm
    svm_model = SVC(C=1000, kernel='rbf', gamma=0.001)
    fit_and_print_results(svm_model, X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, encoder, "SVM")

    # random forrest
    rf = RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=15)
    fit_and_print_results(rf, X_train_scaled, Y_train, X_test_scaled, Y_test, Y_test_label, encoder, "Random Forrest")

    # nn
    genre_list = data[:, -1]
    data_nn = np.array(data[:, 1:-1], dtype=float)
    encoder = LabelEncoder()
    y_nn = encoder.fit_transform(genre_list)
    scaler = StandardScaler()
    X_nn = scaler.fit_transform(data_nn)
    X_train, X_test, y_train, y_test = train_test_split(X_nn, y_nn, test_size=0.2)

    nn = models.Sequential()
    nn.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    nn.add(layers.Dense(128, activation='relu'))
    nn.add(layers.Dense(64, activation='relu'))
    nn.add(layers.Dense(10, activation='softmax'))

    nn.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

    x_val = X_train[:200]
    partial_x_train = X_train[200:]

    y_val = y_train[:200]
    partial_y_train = y_train[200:]

    nn.fit(partial_x_train,
              partial_y_train,
              epochs=200,
              batch_size=32,
              validation_data=(x_val, y_val))
    results = nn.evaluate(X_test, y_test)
    print(results)
    print("Total accuracy for nn on test set was: {}".format(results[1]))
    Y_pred = nn.predict(X_test)
    Y_pred_to_class = np.array(list(map(np.argmax, Y_pred)))
    y_pred_label = list(encoder.inverse_transform(Y_pred_to_class))

    y_test_label = encoder.inverse_transform(y_test)
    print(confusion_matrix(y_test_label, y_pred_label))
    print("\n")
    print(classification_report(y_test_label, y_pred_label))

    # cnn
    y_cnn = to_categorical(y_cnn)

    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn)

    input_shape = X_train[0].shape
    num_genres = 10

    cnn = Sequential()
    # Conv Block 1
    cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu', input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Dropout(0.25))

    # Conv Block 2
    cnn.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Dropout(0.25))

    # Conv Block 3
    cnn.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Dropout(0.25))

    # Conv Block 4
    cnn.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Dropout(0.25))

    # Conv Block 5
    cnn.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    cnn.add(Dropout(0.25))

    # MLP
    cnn.add(Flatten())
    cnn.add(Dense(num_genres, activation='softmax'))

    cnn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    cnn.fit(X_train, y_train,
                     batch_size=32,
                     epochs=20,
                     verbose=1,
                     validation_data=(X_test, y_test))

    score = cnn.evaluate(X_test, y_test, verbose=0)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))
