import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# ----------------- 1. Загрузка и объединение данных -----------------
try:
    train_data = pd.read_csv('heart_adapt_train.csv')
    test_data = pd.read_csv('heart_adapt_test.csv')
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    print("Данные успешно загружены. Общее количество записей:", len(combined_data))
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()

# ----------------- 2. Предобработка данных -----------------
# Выбор значимых признаков
significant_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease'
]
processed_data = combined_data[significant_features].copy()

# Обработка пропусков
numerical_cols = processed_data.select_dtypes(include=np.number).columns
categorical_cols = processed_data.select_dtypes(exclude=np.number).columns

# Заполнение числовых пропусков медианой
for col in numerical_cols:
    processed_data[col].fillna(processed_data[col].median(), inplace=True)

# Удаление строк с пропусками в категориальных данных
processed_data.dropna(subset=categorical_cols, inplace=True)

# ----------------- 3. Анализ распределений -----------------
# Гистограммы с тестом Шапиро-Уилка
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(processed_data[col], kde=True)
    stat, p = shapiro(processed_data[col])
    plt.title(f"{col}\nНормальное распред: {'Да' if p > 0.05 else 'Нет'}", fontsize=10)
plt.tight_layout()
plt.show()

# ----------------- 4. Корреляционный анализ -----------------
corr_matrix = processed_data[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=np.triu(corr_matrix), cbar_kws={'label': 'Коэффициент корреляции'})
plt.title("Корреляция между числовыми признаками", pad=20)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

#новый признак
temp_data['CardioStress'] = processed_data['Oldpeak'] * processed_data['MaxHR']

# ----------------- 6. Кодирование категориальных переменных -----------------
data_encoded = pd.get_dummies(processed_data, columns=categorical_cols.drop('HeartDisease'))

# ----------------- 7. Анализ важности признаков -----------------
X = data_encoded.drop('HeartDisease', axis=1)
# 'HeartDisease':
#   'Расшифровка': 'Наличие сердечного заболевания',
#   'Варианты': '1: Да, 0: Нет',
#   'Назначение': 'Целевая переменная для классификации'
y = data_encoded['HeartDisease']

# Обучение модели для оценки важности признаков
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Визуализация важности признаков
importance_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': rf_model.feature_importances_
}).sort_values('Важность', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Важность', y='Признак', data=importance_df, palette='viridis')
plt.title('Топ-15 важных признаков для прогнозирования', pad=15)
plt.xlabel('Важность признака', labelpad=10)
plt.ylabel('')
plt.show()

# ----------------- 8. Нормализация данных -----------------
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_encoded)
normalized_df = pd.DataFrame(normalized_data, columns=data_encoded.columns)

# ----------------- 9. Сохранение результатов -----------------
engine = create_engine('sqlite:///heart_data.db')
try:
    processed_data.to_sql('heart_data_processed', engine, index=False, if_exists='replace')
    print("Данные сохранены в SQL базу")
except Exception as e:
    print(f"Ошибка сохранения в базу данных: {e}")

# В CSV
processed_data.to_csv('processed_heart_data.csv', index=False)
normalized_df.to_csv('normalized_heart_data.csv', index=False)



# ----------------- 10. Кластеры -----------------
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Подготовка данных
X = normalized_df.drop('HeartDisease', axis=1).copy()

# 1. Определение оптимального числа кластеров для KMeans
plt.figure(figsize=(12, 5))
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 8), metric='distortion', timings=False)
visualizer.fit(X)
visualizer.show()
plt.title('Метод локтя для определения числа кластеров')
plt.show()

# 2. Тестирование 5 алгоритмов кластеризации
models = {
    'KMeans-3': KMeans(n_clusters=3, random_state=42),
    'KMeans-opt': KMeans(n_clusters=visualizer.elbow_value_, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=10),
    'Agglomerative-3': AgglomerativeClustering(n_clusters=3),
    'BIRCH': Birch(n_clusters=3)
}


results = []
best_score = {'silhouette': -1, 'calinski': -1, 'davies': float('inf')}
best_models = {}

for name, model in models.items():
    clusters = model.fit_predict(X)

    if len(set(clusters)) < 2:
        continue

    metrics = {
        'Метод': name,
        'Кластеры': len(set(clusters)),
        'Силуэт': silhouette_score(X, clusters),
        'Калински-Харабас': calinski_harabasz_score(X, clusters),
        'Дэвис-Болдин': davies_bouldin_score(X, clusters)
    }

    if metrics['Силуэт'] > best_score['silhouette']:
        best_score['silhouette'] = metrics['Силуэт']
        best_models['silhouette'] = (name, clusters)

    if metrics['Калински-Харабас'] > best_score['calinski']:
        best_score['calinski'] = metrics['Калински-Харабас']
        best_models['calinski'] = (name, clusters)

    if metrics['Дэвис-Болдин'] < best_score['davies']:
        best_score['davies'] = metrics['Дэвис-Болдин']
        best_models['davies'] = (name, clusters)

    results.append(metrics)

metrics_df = pd.DataFrame(results).sort_values('Силуэт', ascending=False)
print("\nСравнение алгоритмов кластеризации:")
print(metrics_df.to_markdown(index=False))

#пространства признаков
plt.figure(figsize=(18, 12))

# t-SNE проекция
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X)

for i, (name, model) in enumerate(models.items(), 1):
    clusters = model.fit_predict(X)

    plt.subplot(2, 3, i)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='Spectral', alpha=0.6)
    plt.title(f'{name}\nКластеров: {len(set(clusters))}', fontsize=10)
    plt.xticks([])
    plt.yticks([])

plt.suptitle('t-SNE визуализация кластеров', y=0.95)
plt.show()

# 5. Анализ лучшей модели (по силуэту)
best_name, best_clusters = best_models['silhouette']
processed_data['Cluster'] = best_clusters

# Распределение характеристик в кластерах
cluster_stats = processed_data.groupby('Cluster').agg({
    'Age': 'mean',
    'RestingBP': 'mean',
    'Cholesterol': 'mean',
    'MaxHR': 'mean',
    'HeartDisease': 'mean',
    'Sex': lambda x: x.value_counts().index[0]
}).reset_index()

# Интерпретация кластеров
cluster_stats['Риск'] = pd.qcut(cluster_stats['HeartDisease'], 3,
                                labels=['Низкий', 'Средний', 'Высокий'])
risk_mapping = cluster_stats.set_index('Cluster')['Риск'].to_dict()
processed_data['RiskGroup'] = processed_data['Cluster'].map(risk_mapping)

# 6. Параллельные координаты для анализа кластеров
plt.figure(figsize=(14, 8))
axes = pd.plotting.parallel_coordinates(
    cluster_stats.drop('Sex', axis=1),
    'Риск',
    color=('#FF6B6B', '#4ECDC4', '#45B7D1'),
    axvlines=False
)
plt.title('Параллельные координаты характеристик кластеров')
plt.grid(alpha=0.3)
plt.show()

# 7. Тепловая карта различий кластеров
cluster_means = processed_data.groupby('RiskGroup').mean().T
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_means, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)
plt.title('Средние значения признаков по кластерам риска')
plt.ylabel('Признаки')
plt.xlabel('Группа риска')
plt.show()

# 8. Сохранение расширенных результатов
processed_data.to_csv('advanced_clustered_data.csv', index=False)
print("Расширенная кластеризация завершена. Результаты сохранены в advanced_clustered_data.csv")
























