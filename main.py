import os
import pandas as pd
import numpy as np
import joblib


def run_pipeline() -> bool:
    print("\nЗапуск конвеєра обробки даних та навчання моделей")

    try:
        import load_data

        load_data.main()

        import prepare_data

        prepare_data.main()

        import train_model

        train_model.main()

        print("\nНавчання успішно завершено! Моделі готові до використання")
        return True
    except Exception as e:
        print(f"\nПомилка під час запуску конвеєра: {e}")
        return False


def format_hours(hours: float) -> str:
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60

    if h > 0:
        return f"{hours:.2f} год ({h} год {m} хв)"

    return f"{hours:.2f} год ({m} хв)"


def main():
    model_silence_file = "model_silence.joblib"
    model_duration_file = "model_duration.joblib"
    data_silence_file = "prepared_silence_data.csv"
    data_duration_file = "prepared_duration_data.csv"

    files_to_check = [
        model_silence_file,
        model_duration_file,
        data_silence_file,
        data_duration_file,
    ]

    missing_files = [
        filename for filename in files_to_check if not os.path.exists(filename)
    ]

    if missing_files:
        print("Пропущені деякі файли моделей або оброблених даних")

        choice = (
            input("Бажаєте запустити повний конвеєр підготовки та навчання? [Y/n]: ")
            .strip()
            .lower()
        )
        if choice in ("", "y", "yes", "так"):
            success = run_pipeline()

            if not success:
                print("Помилка при ініціалізації середовища. Завершення роботи")
                return
        else:
            print("Робота неможлива без моделей та даних. Завершення роботи")
            return

    model_silence_data = joblib.load(model_silence_file)
    model_silence = model_silence_data["model"]
    features_silence = model_silence_data["features"]
    mae_silence = model_silence_data["mae_by_oblast"]

    model_duration_data = joblib.load(model_duration_file)
    model_duration = model_duration_data["model"]
    features_duration = model_duration_data["features"]
    mae_duration = model_duration_data["mae_by_oblast"]

    df_silence_history = pd.read_csv(data_silence_file)

    oblast_columns = [col for col in features_silence if col.startswith("oblast_")]
    oblasts = sorted([col.replace("oblast_", "") for col in oblast_columns])

    print("\nПрограма прогнозування повітряних тривог готова")

    while True:
        print("\nДоступні регіони:")
        half = (len(oblasts) + 1) // 2

        for i in range(half):
            col1 = f"{i + 1}. {oblasts[i]}"
            if i + half < len(oblasts):
                col2 = f"{i + half + 1}. {oblasts[i + half]}"
                print(f"  {col1:<35} {col2}")
            else:
                print(f"  {col1}")

        user_input = input("\nВведіть номер регіону (або 'q' для виходу): ").strip()

        if user_input.lower() in ("q", "quit", "exit", "вихід"):
            print("Бережіть себе!")
            break

        try:
            num = int(user_input)
            if num < 1 or num > len(oblasts):
                print(f"Введіть число від 1 до {len(oblasts)}!")
                continue
            selected_oblast = oblasts[num - 1]
        except ValueError:
            matches = [o for o in oblasts if user_input.lower() in o.lower()]
            if len(matches) == 1:
                selected_oblast = matches[0]
            else:
                print("Невірний ввід. Введіть правильний номер або назву")
                continue

        print(f"\nОбрано регіон: {selected_oblast}")

        oblast_col = f"oblast_{selected_oblast}"
        hist_silence = df_silence_history[df_silence_history[oblast_col] == 1]
        if hist_silence.empty:
            print("Не знайдено історичних даних для цієї області")
            continue

        latest_silence_row = hist_silence.iloc[-1]

        print("\nОстанній стан тривог у регіоні:")
        print(
            f"  - Остання тривога: {format_hours(latest_silence_row['last_alert_duration'])}"
        )
        print(
            f"  - Передостання тривога: {format_hours(latest_silence_row['prev_alert_duration'])}"
        )
        print(
            f"  - Останній спокій: {format_hours(latest_silence_row['last_silence_duration'])}"
        )
        print(
            f"  - Передостанній спокій: {format_hours(latest_silence_row['prev_silence_duration'])}"
        )

        mae_s = mae_silence.get(selected_oblast, 0.0)
        mae_d = mae_duration.get(selected_oblast, 0.0)

        # 1) Прогноз спокою після останньої відомої тривоги (Model 1)
        X_s = pd.DataFrame([latest_silence_row]).drop(
            columns=["silence_after_duration"]
        )[features_silence]
        pred_s_log = model_silence.predict(X_s)[0]
        pred_s = np.clip(np.expm1(pred_s_log), 0.0, None)

        # 2) Розраховуємо параметри початку наступної тривоги
        end_hour = latest_silence_row["last_alert_end_hour"]
        end_dayofweek = latest_silence_row["last_alert_end_dayofweek"]
        end_month = latest_silence_row["last_alert_end_month"]

        # Додаємо прогнозовані години спокою до часу завершення останньої відомої тривоги
        total_hours = end_hour + pred_s
        predicted_start_hour = int(total_hours % 24)
        days_passed = int(total_hours // 24)
        predicted_start_dayofweek = int((end_dayofweek + days_passed) % 7)
        predicted_start_month = int(end_month)

        # 3) Будуємо рядок для прогнозу тривалості наступної тривоги (Model 2)
        d_row = latest_silence_row.copy()
        d_row["last_silence_duration"] = pred_s
        d_row["prev_silence_duration"] = latest_silence_row["last_silence_duration"]
        d_row["last_alert_duration"] = latest_silence_row["last_alert_duration"]
        d_row["prev_alert_duration"] = latest_silence_row["prev_alert_duration"]
        d_row["alert_start_month"] = predicted_start_month
        d_row["alert_start_dayofweek"] = predicted_start_dayofweek
        d_row["alert_start_hour"] = predicted_start_hour

        X_d = pd.DataFrame([d_row])[features_duration]
        pred_d_log = model_duration.predict(X_d)[0]
        pred_d = np.clip(np.expm1(pred_d_log), 0.0, None)

        print("\nРезультати прогнозування:")
        print("  1) Очікуваний час спокою (до початку наступної тривоги):")
        print(
            f"     -> {format_hours(pred_s)} (похибка області: ± {format_hours(mae_s)})"
        )
        print("  2) Прогнозована тривалість наступної тривоги:")
        print(
            f"     -> {format_hours(pred_d)} (похибка області: ± {format_hours(mae_d)})"
        )

        input("\nНатисніть Enter, щоб продовжити...")


if __name__ == "__main__":
    main()
