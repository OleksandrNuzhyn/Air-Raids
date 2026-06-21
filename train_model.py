import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def train_and_evaluate(
    file_path: str,
    target_column: str,
    task_name: str,
    model_save_name: str,
    plot_name: str,
    plot_title: str,
) -> None:
    if not os.path.exists(file_path):
        print(f"Помилка: Файл '{file_path}' не знайдено!")
        return

    print(f"\nНавчання моделі для завдання: {task_name}...")
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train)

    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train_log)

    predicted_log = model.predict(X_test)
    predicted = np.clip(np.expm1(predicted_log), 0.0, None)

    mae = mean_absolute_error(y_test, predicted)
    print(f"Загальна похибка MAE: {mae:.3f} год ({mae * 60:.1f} хв)")

    onehot_columns = [
        column for column in X_test.columns if column.startswith("oblast_")
    ]
    test_oblasts = X_test[onehot_columns].idxmax(axis=1).str.replace("oblast_", "")

    eval_df = pd.DataFrame(
        {"oblast": test_oblasts, "actual": y_test, "predicted": predicted}
    )

    oblast_metrics = (
        eval_df.groupby("oblast")
        .apply(
            lambda g: pd.Series(
                {
                    "MAE (год)": mean_absolute_error(g["actual"], g["predicted"]),
                    "Кількість тривог": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    model_data = {
        "model": model,
        "features": list(X.columns),
        "mae_by_oblast": oblast_metrics.set_index("oblast")["MAE (год)"].to_dict(),
    }
    joblib.dump(model_data, model_save_name)

    oblast_metrics = oblast_metrics.sort_values(by="MAE (год)").reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(
        oblast_metrics["oblast"],
        oblast_metrics["MAE (год)"],
        color="skyblue",
        edgecolor="gray",
    )
    plt.axvline(
        mae,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Середня MAE: {mae:.2f} год",
    )

    plt.xlabel("Середня помилка MAE (години)")
    plt.ylabel("Область")
    plt.title(plot_title, fontsize=12, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)
    plt.close()


def main():
    train_and_evaluate(
        file_path="prepared_silence_data.csv",
        target_column="silence_after_duration",
        task_name="прогнозування часу спокою після тривоги",
        model_save_name="model_silence.joblib",
        plot_name="mae_silence_by_oblast.png",
        plot_title="Середня помилка (MAE) прогнозування часу спокою після тривоги",
    )

    train_and_evaluate(
        file_path="prepared_duration_data.csv",
        target_column="duration",
        task_name="прогнозування тривалості тривоги",
        model_save_name="model_duration.joblib",
        plot_name="mae_duration_by_oblast.png",
        plot_title="Середня помилка (MAE) прогнозування тривалості тривоги",
    )


if __name__ == "__main__":
    main()
