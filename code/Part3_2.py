# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Data Processing
file_path = 'train.csv'
data = pd.read_csv(file_path)

columns_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN']
data.drop(columns=columns_to_drop, inplace=True)

imputer = SimpleImputer(strategy='mean')
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Age'] = data['Age'].apply(lambda x: np.nan if x < 0 else x)
data['Age'] = imputer.fit_transform(data[['Age']])

def convert_credit_history_age(age_str):
    if pd.isna(age_str):
        return np.nan
    years, months = 0, 0
    if 'Years' in age_str:
        years = int(age_str.split('Years')[0].strip())
    if 'Months' in age_str and age_str.split('Months')[0].strip() != '':
        months = int(age_str.split('Months')[0].split()[-1].strip())
    return years * 12 + months

data['Credit_History_Age'] = data['Credit_History_Age'].apply(convert_credit_history_age)
data['Credit_History_Age'] = imputer.fit_transform(data[['Credit_History_Age']])

categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))

# Multi-layer Feedforward Neural Network
X = data.drop('Credit_Score', axis=1)
y = data['Credit_Score']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model and save history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping], class_weight=class_weights)

# Model Verification
# Loss Convergence
plt.figure(figsize=(8,6))

min_loss_epoch = np.argmin(history.history['loss'])
min_loss_value = history.history['loss'][min_loss_epoch]

min_val_loss_epoch = np.argmin(history.history['val_loss'])
min_val_loss_value = history.history['val_loss'][min_val_loss_epoch]

plt.plot(history.history['loss'], label='Training Loss', linestyle='--', color=(231/255, 188/255, 188/255))
plt.plot(history.history['val_loss'], label='Validation Loss', color=(193/255, 39/255, 45/255))

plt.scatter(min_loss_epoch, min_loss_value, color=(141/255, 31/255, 23/255), s=20, label="Min Training Loss")
plt.text(min_loss_epoch - 1, min_loss_value + 0.005, f'{min_loss_value:.2f}')

plt.scatter(min_val_loss_epoch, min_val_loss_value, color=(141/255, 31/255, 23/255), s=20, label="Min Validation Loss")
plt.text(min_val_loss_epoch - 1, min_val_loss_value + 0.005, f'{min_val_loss_value:.2f}')

plt.title('Loss Convergence', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()

plt.savefig('loss_convergence.png', dpi=300)
plt.show()

# Accuracy
plt.figure(figsize=(8,6))

max_acc_epoch = np.argmax(history.history['accuracy'])
max_acc_value = history.history['accuracy'][max_acc_epoch]

max_val_acc_epoch = np.argmax(history.history['val_accuracy'])
max_val_acc_value = history.history['val_accuracy'][max_val_acc_epoch]

plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='--', color=(231/255, 188/255, 188/255))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color=(193/255, 39/255, 45/255))

plt.scatter(max_acc_epoch, max_acc_value, color=(141/255, 31/255, 23/255), s=20, label="Max Training Accuracy")
plt.text(max_acc_epoch - 1, max_acc_value - 0.006, f'{max_acc_value:.2f}')

plt.scatter(max_val_acc_epoch, max_val_acc_value, color=(141/255, 31/255, 23/255), s=20, label="Max Validation Accuracy")
plt.text(max_val_acc_epoch - 1, max_val_acc_value - 0.006, f'{max_val_acc_value:.2f}')

plt.title('Accuracy Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()

plt.savefig('Accuracy_Over_Epochs.png', dpi=300)
plt.show()

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:\n", classification_report(y_test, y_pred_classes))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_classes) * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.savefig('Confusion Matrix.png', dpi=300)
plt.show()