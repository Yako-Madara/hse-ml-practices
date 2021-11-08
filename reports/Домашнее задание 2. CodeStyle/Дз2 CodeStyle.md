# Отчет по домашнему заданию №2 CodeStyle.
# Инструменты

Для форматирования и анализа кода были выбраны следующие 3 инструмента:

- Форматер **Black**
- Анализатор **radon**
- Линтер **PyFlakes**

Все инструменты были выбраны, исходя из их популярности и простоты в использовании.

## Black

**Black** был запущен для всех скриптов без специальных клучей. В результате получился более читаемый, отформатированный код.

## radon

### Цикломатическая сложность

```
radon cc -a -s path
```

Результат:
```
feature_eng.py
    F 25:0 feature_engineering - D (25)
main.py
    F 55:0 display_missing - A (2)
    F 27:0 concat_df - A (1)
    F 32:0 divide_df - A (1)
plot_importance.py
    F 14:0 plot_importance - A (1)
plot_roc.py
    F 25:0 plot_roc_curve - A (2)
RandomForest.py
    F 26:0 modRandFor - A (5)

7 blocks (classes, functions, methods) analyzed.
Average complexity: B (5.285714285714286)
```
В проекте используются простые функции, поэтому их цикломатическая сложность в среднем ожидаемо низкая.

### Индекс поддерживаемости кода

```
radon mi -s path
```

```
feature_eng.py - A (51.23)
main.py - A (70.37)
plot_importance.py - A (100.00)
plot_roc.py - A (73.87)
RandomForest.py - A (68.53)
```
Анализ показал, что код будет легко поддерживать и редактировать.

### Метрики Холстеда

```
radon hal path
```

Данные метрики описывают сложность программы и количество усилий, предположительно затраченное на написание и понимание кода.

```
feature_eng.py:
    h1: 7
    h2: 35
    N1: 22
    N2: 43
    vocabulary: 42
    length: 65
    calculated_length: 199.17639004747704
    volume: 350.50063248061946
    difficulty: 4.3
    effort: 1507.1527196666636
    time: 83.73070664814799
    bugs: 0.11683354416020648
main.py:
    h1: 3
    h2: 7
    N1: 4
    N2: 8
    vocabulary: 10
    length: 12
    calculated_length: 24.406371956566698
    volume: 39.863137138648355
    difficulty: 1.7142857142857142
    effort: 68.33680652339717
    time: 3.796489251299843
    bugs: 0.013287712379549451
plot_importance.py:
    h1: 0
    h2: 0
    N1: 0
    N2: 0
    vocabulary: 0
    length: 0
    calculated_length: 0
    volume: 0
    difficulty: 0
    effort: 0
    time: 0.0
    bugs: 0.0
plot_roc.py:
    h1: 3
    h2: 4
    N1: 6
    N2: 8
    vocabulary: 7
    length: 14
    calculated_length: 12.75488750216347
    volume: 39.302968908806456
    difficulty: 3.0
    effort: 117.90890672641936
    time: 6.550494818134409
    bugs: 0.01310098963626882
RandomForest.py:
    h1: 5
    h2: 7
    N1: 8
    N2: 14
    vocabulary: 12
    length: 22
    calculated_length: 31.26112492884004
    volume: 78.86917501586544
    difficulty: 5.0
    effort: 394.3458750793272
    time: 21.90810417107373
    bugs: 0.026289725005288478
```
Скрипты RandomForest.py и feature_eng.py оказались самыми сложными в плане написания и понимания, что вполне ожидаемо. Для скрипта plot_importance.py не была посчитана ни одна метрика, с чем это связано сказать сложно.

## PyFlakes

PyFlakes базовый логический flake linter который анализирует программу и ищет потенциальные ошибки. PyFlakes был запущен с настройками по умолчанию

```
flake8 feature_eng.py
feature_eng.py:15:1: E402 module level import not at top of file
feature_eng.py:17:1: E402 module level import not at top of file
feature_eng.py:18:1: E402 module level import not at top of file
feature_eng.py:33:80: E501 line too long (86 > 79 characters)
feature_eng.py:37:80: E501 line too long (85 > 79 characters)
feature_eng.py:69:80: E501 line too long (99 > 79 characters)
feature_eng.py:71:80: E501 line too long (81 > 79 characters)
feature_eng.py:74:80: E501 line too long (81 > 79 characters)
feature_eng.py:88:80: E501 line too long (100 > 79 characters)
feature_eng.py:98:80: E501 line too long (104 > 79 characters)
feature_eng.py:116:80: E501 line too long (82 > 79 characters)
feature_eng.py:144:80: E501 line too long (82 > 79 characters)
feature_eng.py:190:80: E501 line too long (88 > 79 characters)

flake8 main.py
main.py:18:1: E402 module level import not at top of file
main.py:38:80: E501 line too long (84 > 79 characters)
main.py:41:80: E501 line too long (83 > 79 characters)
main.py:57:80: E501 line too long (81 > 79 characters)
main.py:100:80: E501 line too long (85 > 79 characters)
main.py:105:80: E501 line too long (80 > 79 characters)
main.py:169:80: E501 line too long (86 > 79 characters)

flake8 plot_importance.py
plot_importance.py:16:80: E501 line too long (80 > 79 characters)
plot_importance.py:24:80: E501 line too long (88 > 79 characters)

flake8 plot_roc.py
plot_roc.py:15:1: E402 module level import not at top of file
plot_roc.py:18:1: E402 module level import not at top of file
plot_roc.py:67:40: W605 invalid escape sequence '\p'
plot_roc.py:82:17: W605 invalid escape sequence '\p'

flake8 RandomForest.py
RandomForest.py:14:1: E402 module level import not at top of file
RandomForest.py:15:1: E402 module level import not at top of file
RandomForest.py:16:1: E402 module level import not at top of file
RandomForest.py:17:1: E402 module level import not at top of file
RandomForest.py:19:1: E402 module level import not at top of file
RandomForest.py:36:5: F841 local variable 'single_best_model' is assigned to but never used
RandomForest.py:66:80: E501 line too long (85 > 79 characters)
RandomForest.py:86:80: E501 line too long (85 > 79 characters)
RandomForest.py:91:80: E501 line too long (85 > 79 characters)
RandomForest.py:100:80: E501 line too long (86 > 79 characters)
RandomForest.py:103:80: E501 line too long (86 > 79 characters)
RandomForest.py:109:80: E501 line too long (83 > 79 characters)
```

### Статистика по **PyFlakes**

Всего 36 ошибок 2 предупреждения. Самая частая ошибка - E501 line too long. В целом ни одну ошибку, обнаруженную **PyFlakes** нельзя назвать критической.