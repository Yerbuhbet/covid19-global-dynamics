from src.analytics import run_analytics_pipeline
from src.data_loader import run_data_pipeline
from src.database import run_database_pipeline
from src.forecasting import run_forecasting_pipeline


def print_menu() -> None:
    print("\n" + "=" * 54)
    print("  COVID-19 Global Dynamics — Панель управления")
    print("=" * 54)
    print("  1 — Шаг 1: Загрузка CSV чанками, очистка, IQR, Parquet")
    print("  2 — Шаг 2: Создание SQLite (4 таблицы) и 9 SQL-запросов")
    print("  3 — Шаг 3: Аналитика, декомпозиция, heatmap и волны")
    print("  4 — Шаг 4: Прогнозирование и сравнение с Baseline (MAE)")
    print("  5 — Выполнить весь пайплайн целиком (Шаги 1–4)")
    print("  6 — Запустить интерактивный дашборд Dash (http://127.0.0.1:8050)")
    print("  0 — Выход")
    print("=" * 54)


def run_all_steps() -> None:
    print("\n>>> ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА (ШАГИ 1–4) <<<")
    run_data_pipeline()
    run_database_pipeline()
    run_analytics_pipeline()
    run_forecasting_pipeline()
    print("\n>>> ВЕСЬ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН <<<")


def main() -> None:
    while True:
        print_menu()
        try:
            choice = input("Выберите действие (0–6): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход из программы.")
            break

        if choice == "1":
            run_data_pipeline()
        elif choice == "2":
            run_database_pipeline()
        elif choice == "3":
            run_analytics_pipeline()
        elif choice == "4":
            run_forecasting_pipeline()
        elif choice == "5":
            run_all_steps()
        elif choice == "6":
            try:
                from app import launch_dashboard
                launch_dashboard()
            except Exception as exc:
                print(f"[ОШИБКА ДАШБОРДА] {type(exc).__name__}: {exc}")
        elif choice == "0":
            print("Работа завершена.")
            break
        else:
            print("Неверный ввод. Введите цифру от 0 до 6.")


if __name__ == "__main__":
    main()
