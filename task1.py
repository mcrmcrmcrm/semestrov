import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Загрузка данных
data = pd.read_csv('health_dataset.csv')

# Разделение на признаки (X) и целевую переменную (y)
X = data.drop('health_status', axis=1)  # Все колонки, кроме health_status
y = data['health_status']  # Только health_status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Трансформируем тестовые данные тем же scaler'ом

# Создаём модель
model = MLPClassifier(
    hidden_layer_sizes=(10,),  # Один скрытый слой с 10 нейронами
    activation='relu',         # Функция активации ReLU
    solver='adam',             # Алгоритм оптимизации
    max_iter=1000,             # Максимальное число итераций
    early_stopping=True,  # Добавляем раннюю остановку
    validation_fraction=0.1,  # Доля данных для валидации
    n_iter_no_change=10,  # Остановка если loss не улучшается 10 эпох
    verbose=True  # Вывод лога обучения
)

# Обучаем модель
model.fit(X_train_scaled, y_train)

train_accuracy = model.score(X_train_scaled, y_train)
print(f"Точность на тренировочных данных: {train_accuracy:.2f}")

test_accuracy = model.score(X_test_scaled, y_test)
print(f"Точность на тестовых данных: {test_accuracy:.2f}")

y_pred = model.predict(X_test_scaled)

# Пример нового пациента
new_patient = [[
    40,    # age
    130,   # blood_pressure
    200,   # cholesterol
    80,    # heart_rate
    25.5,  # bmi
    110    # glucose
]]

# Масштабируем данные
new_patient_scaled = scaler.transform(new_patient)

# Предсказываем
prediction = model.predict(new_patient_scaled)
print("Предсказание:", "Болен" if prediction[0] == 1 else "Здоров")

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(model.loss_curve_, label='Train Loss')
plt.plot(model.validation_scores_, label='Validation Accuracy')  # Для accuracy
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid()
plt.show()
