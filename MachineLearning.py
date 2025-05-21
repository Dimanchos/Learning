import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv("kc_house_data.csv")

# Выбор нужных признаков
features = ["sqft_lot", "floors", "view"]
target = "price"

# Нормализация
scaler = StandardScaler()
X_raw = df[features].values
y_raw = df[[target]].values

X = scaler.fit_transform(X_raw)
y = scaler.fit_transform(y_raw).flatten()  # приведение к вектору

# Добавляем столбец единиц для свободного члена (bias)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # X.shape -> (n_samples, n_features + 1)


# --- Линейная модель ---
def linear_model(X, weights):
    return X @ weights


# --- Функция потерь MSE ---
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# --- Градиент функции потерь ---
def mse_gradient(X, y, weights):
    n = len(y)
    return (-2 / n) * X.T @ (y - linear_model(X, weights))


# --- Инициализация весов ---
def initialize_weights(n_features):
    return np.random.uniform(-1, 1, size=n_features)


# --- Градиентный спуск ---
def gradient_descent(X, y, initial_weights, learning_rate=0.01, iterations=1000):
    weights = initial_weights.copy()
    loss_history = []

    for i in range(iterations):
        y_pred = linear_model(X, weights)
        loss = mse_loss(y, y_pred)
        grad = mse_gradient(X, y, weights)
        weights -= learning_rate * grad
        loss_history.append(loss)

        if i % 100 == 0:
            print(f"Итерация {i}: MSE = {loss:.4f}")

    return weights, loss_history


# --- Запуск обучения ---
initial_weights = initialize_weights(X.shape[1])
trained_weights, loss_history = gradient_descent(
    X, y, initial_weights, learning_rate=0.01, iterations=1000
)

# --- Вывод результатов ---
print("\nОбученные коэффициенты:")
for i, w in enumerate(trained_weights):
    print(f"w{i} = {w:.4f}")
print(f"\nФинальный MSE: {loss_history[-1]:.4f}")

# --- Визуализация ошибки ---
plt.plot(loss_history)
plt.title("Снижение ошибки (MSE) по итерациям")
plt.xlabel("Итерации")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# --- Оценка обученной модели на всей выборке ---
y_pred_final = linear_model(X, trained_weights)
final_mse = mse_loss(y, y_pred_final)

print(f"\nИтоговая среднеквадратичная ошибка (MSE) обученной модели: {final_mse:.4f}")

# --- Вывод ---
if final_mse < 0.1:
    print(
        "Модель обучена хорошо: ошибка мала, предсказания близки к истинным значениям."
    )
elif final_mse < 0.3:
    print("Модель обучена удовлетворительно: возможны некоторые отклонения.")
else:
    print("Модель обучена слабо: ошибка велика, требуется дополнительная настройка.")

# --- Графики зависимости y от каждого признака ---
feature_names = ["sqft_lot", "floors", "view"]
X_means = np.mean(X[:, 1:], axis=0)  # средние значения признаков (без bias)
n_points = 100

for i in range(3):
    x_range = np.linspace(np.min(X[:, i + 1]), np.max(X[:, i + 1]), n_points)
    X_temp = np.tile(X_means, (n_points, 1))  # фиксируем другие признаки
    X_temp[:, i] = x_range  # варьируем только текущий признак
    X_plot = np.hstack([np.ones((n_points, 1)), X_temp])
    y_plot = linear_model(X_plot, trained_weights)

    plt.figure()
    plt.scatter(X[:, i + 1], y, color="red", label="Нормализованные данные")
    plt.plot(x_range, y_plot, color="blue", label="Линия регрессии")
    plt.xlabel(f"{feature_names[i]} (норм.)")
    plt.ylabel("price (норм.)")
    plt.title(f"Зависимость цены от {feature_names[i]}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Вывод уравнения модели
print("Функциональный вид обученной модели (в нормализованных данных):")
print(
    f"price_norm = {trained_weights[0]:.3f} + {trained_weights[1]:.3f} * sqft_lot_norm + {trained_weights[2]:.3f} * floors_norm + {trained_weights[3]:.3f} * view_norm"
)

# Анализ влияния коэффициентов
coeff_names = ["sqft_lot", "floors", "view"]
coeff_values = trained_weights[1:]  # без смещения

max_idx = np.argmax(np.abs(coeff_values))
print("\nАнализ влияния признаков на цену:")
for name, val in zip(coeff_names, coeff_values):
    print(f"{name}: коэффициент = {val:.3f}")

print(f"\nНаибольшее влияние оказывает признак: {coeff_names[max_idx]}")
