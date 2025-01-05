# models/tuning/lightgbm_tuning.py
import csv
import numpy as np
import os
import concurrent.futures
import time
import logging
from models.lightgbm import LightGBMModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/lightgbm_tuning.log", mode="w"),
    ],
)


# Hàm đánh giá hiệu suất mô hình
def evaluate_model(
    n_estimators,
    learning_rate,
    max_depth,
    lambda_l1,
    lambda_l2,
    X_train,
    y_train,
    X_val,
    y_val,
    seed=42,
):
    # Đảm bảo seed là số nguyên
    seed = int(seed) if isinstance(seed, (str, float)) else seed
    np.random.seed(seed)  # Cố định seed

    # Log thông tin về hyperparameters trước khi bắt đầu
    logging.info(
        f"Starting evaluation with: n_estimators={n_estimators}, learning_rate={learning_rate}, "
        f"max_depth={max_depth}, lambda_l1={lambda_l1}, lambda_l2={lambda_l2}"
    )

    lgbm_model = LightGBMModel(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        verbose=False,
    )

    # Đo thời gian huấn luyện
    start_time = time.time()
    lgbm_model.train(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Đánh giá trên tập train và validation
    train_acc = lgbm_model.evaluate(X_train, y_train)
    val_acc = lgbm_model.evaluate(X_val, y_val)

    # Log kết quả chi tiết
    logging.info(
        f"Completed: n_estimators={n_estimators}, learning_rate={learning_rate}, "
        f"max_depth={max_depth}, lambda_l1={lambda_l1}, lambda_l2={lambda_l2}, "
        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, training_time={training_time:.2f}s"
    )

    # Convert start_time to string with YYYY-MM-DD HH:MM:SS format
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "training_time": training_time,
    }


# Hàm tối ưu hóa siêu tham số
def lightgbm_hyperparameter_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    output_file="lgbm_tuning_results.json",
    max_workers=None,  # Số tiến trình tối đa
):
    if max_workers is None:
        max_workers = os.cpu_count()  # Tự động lấy số CPU khả dụng
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
                "learning_rate",
                "max_depth",
                "lambda_l1",
                "lambda_l2",
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
                for learning_rate in param_grid["learning_rate"]:
                    for max_depth in param_grid["max_depth"]:
                        for lambda_l1 in param_grid["lambda_l1"]:
                            for lambda_l2 in param_grid["lambda_l2"]:
                                futures.append(
                                    executor.submit(
                                        evaluate_model,
                                        n_estimators,
                                        learning_rate,
                                        max_depth,
                                        lambda_l1,
                                        lambda_l2,
                                        X_train,
                                        y_train,
                                        X_val,
                                        y_val,
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
                        f"n_estimators={result['n_estimators']}, learning_rate={result['learning_rate']}, "
                        f"max_depth={result['max_depth']}, lambda_l1={result['lambda_l1']}, "
                        f"lambda_l2={result['lambda_l2']}, train_acc={result['train_acc']:.4f}, val_acc={result['val_acc']:.4f}, "
                        f"start_time={result['start_time']}, end_time={result['end_time']}, "
                        f"training_time={result['training_time']:.2f}s"
                    )

                    # Cập nhật tham số tốt nhất
                    if result["val_acc"] > best_val_acc:
                        best_val_acc = result["val_acc"]
                        best_params = {
                            "n_estimators": result["n_estimators"],
                            "learning_rate": result["learning_rate"],
                            "max_depth": result["max_depth"],
                            "lambda_l1": result["lambda_l1"],
                            "lambda_l2": result["lambda_l2"],
                        }

                except Exception as e:
                    logging.error(f"Error in task {i}/{total_tasks}: {e}")

    # Train lại mô hình tốt nhất
    best_model = LightGBMModel(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        lambda_l1=best_params["lambda_l1"],
        lambda_l2=best_params["lambda_l2"],
        verbose=False,
    )
    best_model.train(X_train, y_train)
    best_model.save_model_weights("weights_lightgbm_best.json")

    logging.info("Hyperparameter tuning completed.")
    logging.info(
        f"Best parameters: {best_params}, Best Validation Accuracy: {best_val_acc:.4f}"
    )

    return best_model, best_params, best_val_acc
