# utils/preprocessing.py

import numpy as np
import pandas as pd


def train_test_split(data, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[: -int(test_size * data.shape[0])]
    test_indices = indices[-int(test_size * data.shape[0]) :]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data


def identify_high_correlation(df, threshold=0.9):
    corr_matrix = df.corr()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    print(f"Các đặc trưng có tương quan cao (>{threshold}): {high_corr}")
    return high_corr


def handle_missing_values(df):
    missing_count = df.isnull().sum()
    if missing_count.any():
        print("Giá trị bị thiếu trong dữ liệu:")
        print(missing_count[missing_count > 0])
        # Ví dụ: Thay thế giá trị bị thiếu bằng trung vị
        df.fillna(df.median(), inplace=True)
    else:
        print("Không có giá trị bị thiếu.")


# Hàm Min-Max Scaling thủ công
def min_max_scaling(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


# Chuẩn hóa tập huấn luyện
def fit_transform(data, continuous_features):
    scaling_params = {}

    # Đảm bảo các cột liên tục là float trước khi chuẩn hóa
    data[continuous_features] = data[continuous_features].astype(float)

    for feature in continuous_features:
        min_val = np.min(data[feature])
        max_val = np.max(data[feature])
        scaling_params[feature] = (min_val, max_val)

        # Áp dụng Min-Max Scaling
        data.loc[:, feature] = min_max_scaling(data[feature], min_val, max_val)
    return data, scaling_params


# Chuẩn hóa tập kiểm tra
def transform(data, continuous_features, scaling_params):
    # Đảm bảo các cột liên tục là float trước khi chuẩn hóa
    data[continuous_features] = data[continuous_features].astype(float)

    for feature in continuous_features:
        min_val, max_val = scaling_params[feature]

        # Áp dụng Min-Max Scaling
        data.loc[:, feature] = min_max_scaling(data[feature], min_val, max_val)
    return data


# Feature Engineering
def feature_engineering(train_data, test_data):
    # Tạo đặc trưng mới
    train_data["aspect_ratio"] = train_data["px_height"] / (
        train_data["px_width"] + 1e-9
    )
    train_data["power_to_weight"] = train_data["battery_power"] / (
        train_data["mobile_wt"] + 1e-9
    )

    test_data["aspect_ratio"] = test_data["px_height"] / (test_data["px_width"] + 1e-9)
    test_data["power_to_weight"] = test_data["battery_power"] / (
        test_data["mobile_wt"] + 1e-9
    )

    return train_data, test_data


# Categorical Encoding
def categorical_encoding(train_data, test_data, categorical_features):
    train_data = pd.get_dummies(
        train_data, columns=categorical_features, drop_first=True
    )
    test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)

    # Đảm bảo cả tập huấn luyện và tập kiểm tra có cùng số lượng cột
    missing_cols = set(train_data.columns) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0

    return train_data, test_data
