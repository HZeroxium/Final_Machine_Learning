# models/tuning/neural_network_tuning.py

import csv
import numpy as np
import time
import logging
from models.neural_network import NeuralNetworkModel
import concurrent.futures
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/neural_network_tuning.log", mode="w"),
    ],
)


# Hàm đánh giá hiệu suất mô hình Neural Network
def evaluate_model(
    hidden_layer_sizes,
    activation,
    learning_rate,
    epochs,
    batch_size,
    dropout_rate,
    l2_lambda,
    X_train,
    y_train,
    X_val,
    y_val,
    seed=42,
):
    # Log thông tin hyperparameters
    logging.info(
        f"Evaluating model with: hidden_layer_sizes={hidden_layer_sizes}, "
        f"activation={activation}, learning_rate={learning_rate}, "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"dropout_rate={dropout_rate}, l2_lambda={l2_lambda}"
    )

    # Khởi tạo mô hình
    nn_model = NeuralNetworkModel(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
        seed=seed,
        verbose=False,
    )

    # Đo thời gian huấn luyện
    start_time = time.time()
    nn_model.train(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    # Đánh giá
    train_acc = nn_model.evaluate(X_train, y_train)
    val_acc = nn_model.evaluate(X_val, y_val)

    # Log kết quả
    logging.info(
        f"Completed: hidden_layer_sizes={hidden_layer_sizes}, "
        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, training_time={training_time:.2f}s"
    )
    # Convert start_time to string with YYYY-MM-DD HH:MM:SS format
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "l2_lambda": l2_lambda,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "training_time": training_time,
    }


# Hàm tối ưu hóa siêu tham số
def neural_network_hyperparameter_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    output_file="neural_network_tuning_results.csv",
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
                "hidden_layer_sizes",
                "activation",
                "learning_rate",
                "epochs",
                "batch_size",
                "dropout_rate",
                "l2_lambda",
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
            for hidden_layer_sizes in param_grid["hidden_layer_sizes"]:
                for activation in param_grid["activation"]:
                    for learning_rate in param_grid["learning_rate"]:
                        for epochs in param_grid["epochs"]:
                            for batch_size in param_grid["batch_size"]:
                                for dropout_rate in param_grid["dropout_rate"]:
                                    for l2_lambda in param_grid["l2_lambda"]:
                                        futures.append(
                                            executor.submit(
                                                evaluate_model,
                                                hidden_layer_sizes,
                                                activation,
                                                learning_rate,
                                                epochs,
                                                batch_size,
                                                dropout_rate,
                                                l2_lambda,
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
                        f"hidden_layer_sizes={result['hidden_layer_sizes']}, activation={result['activation']}, "
                        f"learning_rate={result['learning_rate']}, epochs={result['epochs']}, "
                        f"batch_size={result['batch_size']}, dropout_rate={result['dropout_rate']}, "
                        f"l2_lambda={result['l2_lambda']}, train_acc={result['train_acc']:.4f}, "
                        f"val_acc={result['val_acc']:.4f}, start_time={result['start_time']}, "
                        f"end_time={result['end_time']}, training_time={result['training_time']:.2f}s"
                    )

                    # Cập nhật tham số tốt nhất
                    if result["val_acc"] > best_val_acc:
                        best_val_acc = result["val_acc"]
                        best_params = {
                            "hidden_layer_sizes": result["hidden_layer_sizes"],
                            "activation": result["activation"],
                            "learning_rate": result["learning_rate"],
                            "epochs": result["epochs"],
                            "batch_size": result["batch_size"],
                            "dropout_rate": result["dropout_rate"],
                            "l2_lambda": result["l2_lambda"],
                        }

                except Exception as e:
                    logging.error(f"Error in task {i}/{total_tasks}: {e}")

    # Train lại mô hình tốt nhất
    best_model = NeuralNetworkModel(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        learning_rate=best_params["learning_rate"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        dropout_rate=best_params["dropout_rate"],
        l2_lambda=best_params["l2_lambda"],
        seed=42,
        verbose=False,
    )
    best_model.train(X_train, y_train)

    logging.info("Hyperparameter tuning completed.")
    logging.info(
        f"Best parameters: {best_params}, Best Validation Accuracy: {best_val_acc:.4f}"
    )

    return best_params, best_val_acc, best_model
