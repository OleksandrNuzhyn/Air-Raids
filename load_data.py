import os
import pandas as pd


def main():
    url = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
    data_filename = "official_data_uk.csv"

    try:
        if os.path.exists(data_filename):
            print(f"Файл '{data_filename}' знайдено. Зчитуємо локальні дані...")
            df = pd.read_csv(data_filename, encoding="utf-8")
        else:
            print("Локального файлу не знайдено. Завантаження датасету з GitHub...")
            df = pd.read_csv(url, encoding="utf-8")
            print("Датасет успішно завантажено")

            print(f"Збереження сирих даних у локальний файл {data_filename}...")
            df.to_csv(data_filename, index=False, encoding="utf-8")
            print("Локальний файл успішно збережено!\n")

        print("\nСТРУКТУРА ДАТАСЕТУ:")
        df.info()
        print("\nПЕРШІ 5 РЯДКІВ:")
        pd.set_option("display.max_columns", None)
        print(df.head())
    except Exception as e:
        print(f"Помилка під час завантаження даних: {e}")


if __name__ == "__main__":
    main()
