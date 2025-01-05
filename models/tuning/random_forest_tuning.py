# models/tuning/random_forest_tuning.py

import csv
import numpy as np
import time
import logging
from models.random_forest import RandomForestModel
import concurrent.futures
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/random_forest_tuning.log", mode="w"),
    ],
)


# Hàm đánh giá hiệu suất mô hình
def evaluate_model(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    X_train,
    y_train,
    X_val,
    y_val,
    seed=42,
    criterion="gini",
):
    # Đảm bảo seed là số nguyên
    seed = int(seed) if isinstance(seed, (str, float)) else seed
    np.random.seed(seed)  # Cố định seed

    # Log thông tin hyperparameters
    logging.info(
        f"Evaluating model with: n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}"
    )

    rf_model = RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        verbose=False,
    )

    # Đo thời gian huấn luyện
    start_time = time.time()
    rf_model.train(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Đánh giá
    train_acc = rf_model.evaluate(X_train, y_train)
    val_acc = rf_model.evaluate(X_val, y_val)

    # Log kết quả
    logging.info(
        f"Completed: n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, training_time={training_time:.2f}s"
    )
    # Convert start_time to string with YYYY-MM-DD HH:MM:SS format
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "training_time": training_time,
    }


# Hàm tối ưu hóa siêu tham số
def random_forest_hyperparameter_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    criterion="gini",
    output_file="random_forest_tuning_results.csv",
    max_workers=None,  # Số tiến trình tối đa
):
    if max_workers is None:
        max_workers = os.cpu_count()  # Mặc định sử dụng tất cả CPU có sẵn
    if max_workers > os.cpu_count():
        max_workers = os.cpu_count()
        logging.warning(
            f"Number of workers exceeds CPU count. Using {max_workers} workers"
        )
    logging.info(f"Using {max_workers} workers for hyperparameter tuning")

    best_params = None
    best_val_acc = 0.0

    # Khởi tạo file CSV
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "train_acc",
                "val_acc",
                "start_time",
                "end_time",
                "training_time",
            ],
        )
        writer.writeheader()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            for n_estimators in param_grid["n_estimators"]:
                for max_depth in param_grid["max_depth"]:
                    for min_samples_split in param_grid["min_samples_split"]:
                        for min_samples_leaf in param_grid["min_samples_leaf"]:
                            futures.append(
                                executor.submit(
                                    evaluate_model,
                                    n_estimators,
                                    max_depth,
                                    min_samples_split,
                                    min_samples_leaf,
                                    X_train,
                                    y_train,
                                    X_val,
                                    y_val,
                                    42,  # Seed cố định
                                    criterion,
                                )
                            )

            total_tasks = len(futures)
            logging.info(f"Total tasks to complete: {total_tasks}")

            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()

                    # Ghi kết quả vào CSV
                    writer.writerow(result)
                    f.flush()  # Ghi dữ liệu ngay lập tức từ buffer vào file
                    # Log chi tiết từng kết quả
                    logging.info(
                        f"Task {i}/{total_tasks} completed with: "
                        f"n_estimators={result['n_estimators']}, max_depth={result['max_depth']}, "
                        f"min_samples_split={result['min_samples_split']}, min_samples_leaf={result['min_samples_leaf']}, "
                        f"train_acc={result['train_acc']:.4f}, val_acc={result['val_acc']:.4f}, "
                        f"start_time={result['start_time']}, end_time={result['end_time']}, "
                        f"training_time={result['training_time']:.2f}s"
                    )

                    # Cập nhật tham số tốt nhất
                    if result["val_acc"] > best_val_acc:
                        best_val_acc = result["val_acc"]
                        best_params = {
                            "n_estimators": result["n_estimators"],
                            "max_depth": result["max_depth"],
                            "min_samples_split": result["min_samples_split"],
                            "min_samples_leaf": result["min_samples_leaf"],
                        }

                except Exception as e:
                    logging.error(f"Error in task {i}/{total_tasks}: {e}")

    # Train lại mô hình tốt nhất
    best_model = RandomForestModel(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=criterion,
        verbose=False,
    )
    best_model.train(X_train, y_train)

    best_model.save_model_weights("weights_random_forest_best.json")

    logging.info("Hyperparameter tuning completed.")
    logging.info(
        f"Best parameters: {best_params}, Best Validation Accuracy: {best_val_acc:.4f}"
    )

    return best_model, best_params, best_val_acc
