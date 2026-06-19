import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score
)



df = pd.read_csv("Churn_Modelling.csv")

print("Columns in dataset:")
print(df.columns.tolist())



cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']

for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

print("\nColumns after dropping:")
print(df.columns.tolist())


le_geo = LabelEncoder()
le_gender = LabelEncoder()

if 'Geography' in df.columns:
    df['Geography'] = le_geo.fit_transform(df['Geography'])

if 'Gender' in df.columns:
    df['Gender'] = le_gender.fit_transform(df['Gender'])


X = df.drop('Exited', axis=1)
y = df['Exited']

t

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.0001
)



history = ann.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)


sample = pd.DataFrame(
    [[600, 0, 1, 40, 3, 60000, 2, 1, 1, 50000]],
    columns=X.columns
)

sample_scaled = sc.transform(sample)

prediction = ann.predict(sample_scaled)

print("\nPrediction Probability:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Customer will leave the bank")
else:
    print("Customer will stay in the bank")


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)


print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))



train_loss, train_acc = ann.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = ann.evaluate(X_test, y_test, verbose=0)

y_train_pred = ann.predict(X_train)
train_auc = roc_auc_score(y_train, y_train_pred)

y_test_pred = ann.predict(X_test)
test_auc = roc_auc_score(y_test, y_test_pred)

print("\nTraining Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Test AUC:", test_auc)




plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()
import joblib
ann.save("churn_model.keras")
joblib.dump(sc, "scaler.pkl")
