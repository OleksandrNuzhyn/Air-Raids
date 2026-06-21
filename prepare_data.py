import os
import pandas as pd


def main():
    data_filename = "official_data_uk.csv"
    prepared_silence_data = "prepared_silence_data.csv"
    prepared_duration_data = "prepared_duration_data.csv"

    if not os.path.exists(data_filename):
        print(f"Помилка: Локальний файл '{data_filename}' не знайдено!")
        print("Cпочатку запустіть скрипт 'load_data.py', щоб завантажити дані")
        return

    print(f"1) Зчитування локального датасету '{data_filename}'...")
    df = pd.read_csv(data_filename, encoding="utf-8")

    # За полем level відкидаємо всі рядки, які є не областями, та видаляємо дублікати
    print("2) Фільтрація за рівнем 'oblast' та видалення дублікатів...")
    df = df[df["level"] == "oblast"].copy()
    df = df.drop_duplicates()

    # Видаляємо області з недостатньою кількістю записів (менше 10)
    oblast_counts = df["oblast"].value_counts()
    low_count_oblasts = oblast_counts[oblast_counts < 10].index
    if not low_count_oblasts.empty:
        print("Видалення областей з малою кількістю записів")
        df = df[~df["oblast"].isin(low_count_oblasts)].copy()

    # Перетворюємо дати в об'єкти datetime та переводимо у київський час (Europe/Kyiv)
    df["started_at"] = pd.to_datetime(df["started_at"]).dt.tz_convert("Europe/Kyiv")
    df["finished_at"] = pd.to_datetime(df["finished_at"]).dt.tz_convert("Europe/Kyiv")

    # Сортуємо хронологічно в межах кожної області
    df = df.sort_values(by=["oblast", "started_at"]).reset_index(drop=True)

    print("3) Розрахунок інтервалів тривоги та спокою в годинах...")
    # Розраховуємо тривалість тривоги для кожного екземпляра в годинах
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600.0

    # Розраховуємо час спокою перед цією тривогою в годинах
    df["silence_before_duration"] = (
        df["started_at"] - df.groupby("oblast")["finished_at"].shift(1)
    ).dt.total_seconds() / 3600.0

    print("\nФормування датасету 1 (прогнозування часу спокою після тривоги)")
    df_silence = df.copy()

    # Створення ознак
    print("4) Створення ознак для прогнозування наступного спокою...")

    # Цільова змінна: час спокою ПІСЛЯ поточної тривоги (це silence_before_duration для наступного рядка цієї ж області)
    df_silence["silence_after_duration"] = df_silence.groupby("oblast")[
        "silence_before_duration"
    ].shift(-1)

    # Ознаки:
    # 2 поля тривалості тривог (поточної та передостанньої)
    df_silence["last_alert_duration"] = df_silence["duration"]
    df_silence["prev_alert_duration"] = df_silence.groupby("oblast")["duration"].shift(
        1
    )

    # 2 поля часу спокою (перед поточною та передостанньою тривогами)
    df_silence["last_silence_duration"] = df_silence["silence_before_duration"]
    df_silence["prev_silence_duration"] = df_silence.groupby("oblast")[
        "silence_before_duration"
    ].shift(1)

    # Округляємо всі розраховані значення тривалості до сотих
    duration_columns_silence = [
        "silence_after_duration",
        "last_alert_duration",
        "prev_alert_duration",
        "last_silence_duration",
        "prev_silence_duration",
    ]
    df_silence[duration_columns_silence] = df_silence[duration_columns_silence].round(2)

    # Розбиваємо дату кінця останньої тривоги
    df_silence["last_alert_end_month"] = df_silence["finished_at"].dt.month
    df_silence["last_alert_end_dayofweek"] = df_silence["finished_at"].dt.dayofweek
    df_silence["last_alert_end_hour"] = df_silence["finished_at"].dt.hour

    print("5) Кодування назв областей (One-Hot Encoding)...")
    df_silence_encoded = pd.get_dummies(
        df_silence, columns=["oblast"], prefix="oblast", dtype=int
    )

    print("6) Очистка датасету від рядків та зайвих колонок...")
    # Видаляємо перші рядки, які не мають достатньо історії
    columns_to_check_silence = [
        "silence_after_duration",
        "prev_alert_duration",
        "prev_silence_duration",
    ]
    df_silence_clean = df_silence_encoded.dropna(subset=columns_to_check_silence).copy()

    # Видаляємо проміжні та непотрібні для навчання колонки
    columns_to_drop_silence = [
        "raion",
        "hromada",
        "level",
        "started_at",
        "finished_at",
        "source",
        "duration",
        "silence_before_duration",
    ]
    df_silence_final = df_silence_clean.drop(
        columns=columns_to_drop_silence
    ).reset_index(drop=True)

    print(f"7) Збереження результату у {prepared_silence_data}...")
    df_silence_final.to_csv(prepared_silence_data, index=False, encoding="utf-8")

    print("\nФормування датасету 2 (прогнозування тривалості тривоги, що почалася)")
    df_duration = df.copy()

    # Створення ознак
    print("4) Створення ознак для прогнозування тривалості тривоги...")

    # Цільова змінна: тривалість цієї тривоги (duration)
    # Ознаки:
    # 2 поля часу спокою (перед поточною та передостанньою тривогами)
    df_duration["last_silence_duration"] = df_duration["silence_before_duration"]
    df_duration["prev_silence_duration"] = df_duration.groupby("oblast")[
        "silence_before_duration"
    ].shift(1)

    # 2 поля тривалості тривог (попередньої та передостанньої)
    df_duration["last_alert_duration"] = df_duration.groupby("oblast")[
        "duration"
    ].shift(1)
    df_duration["prev_alert_duration"] = df_duration.groupby("oblast")[
        "duration"
    ].shift(2)

    # Округляємо всі розраховані значення тривалості до сотих
    duration_columns_dur = [
        "duration",
        "last_silence_duration",
        "prev_silence_duration",
        "last_alert_duration",
        "prev_alert_duration",
    ]
    df_duration[duration_columns_dur] = df_duration[duration_columns_dur].round(2)

    # Розбиваємо дату початку поточної тривоги
    df_duration["alert_start_month"] = df_duration["started_at"].dt.month
    df_duration["alert_start_dayofweek"] = df_duration["started_at"].dt.dayofweek
    df_duration["alert_start_hour"] = df_duration["started_at"].dt.hour

    print("5) Кодування назв областей (One-Hot Encoding)...")
    df_duration_encoded = pd.get_dummies(
        df_duration, columns=["oblast"], prefix="oblast", dtype=int
    )

    print("6) Очистка датасету від рядків та зайвих колонок...")
    # Видаляємо перші рядки, які не мають достатньо історії
    columns_to_check_duration = [
        "duration",
        "last_silence_duration",
        "prev_silence_duration",
        "last_alert_duration",
        "prev_alert_duration",
    ]
    df_duration_clean = df_duration_encoded.dropna(
        subset=columns_to_check_duration
    ).copy()

    # Видаляємо проміжні та непотрібні для навчання колонки
    columns_to_drop_duration = [
        "raion",
        "hromada",
        "level",
        "started_at",
        "finished_at",
        "source",
        "silence_before_duration",
    ]
    df_duration_final = df_duration_clean.drop(
        columns=columns_to_drop_duration
    ).reset_index(drop=True)

    print(f"7) Збереження результату у {prepared_duration_data}...")
    df_duration_final.to_csv(prepared_duration_data, index=False, encoding="utf-8")

    pd.set_option("display.max_columns", None)

    print("\nДАТАСЕТ 1: СПОКІЙ (СТРУКТУРА)")
    df_silence_final.info()
    print("\nДАТАСЕТ 1: СПОКІЙ (ПЕРШІ 5 РЯДКІВ)")
    print(df_silence_final.head())

    print("\nДАТАСЕТ 2: ТРИВАЛІСТЬ (СТРУКТУРА)")
    df_duration_final.info()
    print("\nДАТАСЕТ 2: ТРИВАЛІСТЬ (ПЕРШІ 5 РЯДКІВ)")
    print(df_duration_final.head())


if __name__ == "__main__":
    main()
