# Vehicle Breakdown Prediction

## Описание проекта

Этот репозиторий содержит все необходимые данные и ноутбуки для проекта по предсказанию поломок автомобилей. Задача решается с использованием различных моделей машинного обучения, включая ансамбли и оптимизацию гиперпараметров.

## 🚗 Описание задачи

Мы решаем задачу для каршеринговой компании, которая хочет сократить простой машин из-за поломок. Для этого нужно построить модель, которая будет предсказывать:
- **время до поломки** (регрессия);
- **тип поломки** (классификация).

Цель: **создание приоритетного списка машин для осмотра технической бригадой**.


## 📂 Данные

Используются 4 источника данных:

1. `car_train.csv` — основная таблица с машинами и таргетами
2. `rides_info.csv` — информация о поездках
3. `driver_info.csv` — данные о водителях
4. `fix_info.csv` — история ремонтов

## Структура репозитория
.
├── data
│   ├── car_ids_test.csv
│   ├── car_test_final.csv
│   ├── car_train_enriched.csv
│   ├── car_train_filtered.csv
│   ├── car_train_full.csv
│   ├── car_train_merged.csv
│   ├── links_with_models.csv
│   ├── mean_salary_by_city_sorted.csv
├── models
│   └── weighted_soft_top3.pkl
├── Competitive_DS_Zaslavskaia_V_All_Tasks_v2.ipynb
└── README.md
├── submissions
│   └── submission_weighted_soft_311.csv 
│ └── xgb_tuned.pkl 
├── submissions 
│ ├── submission.csv
│ ├── submission_hard_voting.csv 
│ ├── submission_soft_optuna.csv │
 ├── submission_xgb_tuned.csv 
 │ ├── submission (7).csv │
  └── submission (8).csv 
├── models 
├── best_weights.json
├── catboost_tuned.pkl 
├── label_encoder_blend.pkl 

├── lgb_model_tuned.pkl
├── rf_tuned.pkl ├── Competitive_DS_Zaslavskaia_V_All_Tasks_v2.ipynb ├── README.md

## 🔢 Этапы проекта

### ✅ ДЗ 1: Генерация и фильтрация признаков
- Созданы новые признаки из `rides_info`, `driver_info`, `fix_info`.
- Отбор признаков с помощью:
  - константности,
  - корреляции Phik,
  - Permutation Importance (Random Forest),
  - SHAP.
- Выбрано 12 финальных признаков.

### ✅ ДЗ 2: Обучение базовой модели
- Обучение `CatBoostClassifier` на отобранных признаках.
- Accuracy ≈ 0.974 на валидационной выборке.

### ✅ ДЗ 3: Тюнинг гиперпараметров
- Подбор через `Optuna`.
- Улучшение accuracy до ≈ 0.9765.
- Использование `early_stopping_rounds` для контроля переобучения.

### ✅ ДЗ 4: Блендинг моделей
- Обучение и тюнинг моделей:
  - `CatBoostClassifier`
  - `LGBMClassifier (goss)`
  - `XGBClassifier (dart)`
  - `RandomForestClassifier`
- Реализованы:
  - Hard Voting (реализация вручную без VotingClassifier)
  - Soft Voting (с оптимизацией весов через Optuna)
- Лучшее качество после оптимизированного Soft Voting: **accuracy = 0.9808**.

### ✅ ДЗ 5: Сабмит на Kaggle
- Предобработка тестовой выборки.
- Применение обученных моделей для предсказания.
- Сабмит результата на Kaggle.
- Лучшая публичная метрика: **0.96549**.

## 🔺 Финальные выводы

- Наилучший результат показал Soft Voting с оптимизированными весами моделей.
- `LightGBM`, несмотря на простую архитектуру, внёс наибольший вклад в финальную метрику.
- Правильная обработка категориальных признаков и баланс моделей оказались критичны.
- Блендинг моделей значительно повысил устойчивость финального решения.

## 🏋️ Используемые библиотеки

```python
numpy==1.26.4
pandas
matplotlib
seaborn
catboost
shap
optuna
phik
lightgbm
xgboost
scikit-learn
```

## 📁 Описание файлов в `data/`

| Файл | Описание |
|------|----------|
| `car_ids_test.csv` | Список ID машин из тестовой выборки, используется для формирования финального сабмита |
| `car_test_final.csv` | Финальная тестовая выборка с нужными признаками, готовая к инференсу моделей |
| `car_train_enriched.csv` | Обогащённая тренировочная выборка, объединённая с внешними источниками (например, по зарплатам или другим характеристикам) |
| `car_train_filtered.csv` | Отфильтрованная версия тренировочной выборки после отбора признаков (по SHAP, Permutation, Phik) |
| `car_train_full.csv` | Полная версия тренировочного сета после объединения всех источников, до фильтрации признаков |
| `car_train_merged.csv` | Сырые данные после объединения основных таблиц (`car_train`, `rides_info`, `driver_info`, `fix_info`) |
| `links_with_models.csv` | Таблица со ссылками на страницы моделей машин, использовалась для парсинга данных с сайтов |
| `mean_salary_by_city_sorted.csv` | Справочная таблица со средней зарплатой по городам, использовалась при обогащении датасета |
| `submission.csv` | Финальный файл с предсказаниями классов поломки для отправки на Kaggle |


**Автор:** Ника (Data Scientist)

**Формат:** Google Colab / Kaggle Notebook

**Цель:** Использование реальных ML-инструментов в соревновательном pipeline