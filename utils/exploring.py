# utils/exploring.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    """Load dữ liệu từ các file CSV."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data.drop(columns=["id"], inplace=True)
    test_data.drop(columns=["id"], inplace=True)

    return train_data, test_data


def summarize_data(df, name="Dataset"):
    """Tóm tắt thông tin và thống kê cơ bản của dữ liệu."""
    print(f"Thông tin {name}:")
    print(df.info())
    print(f"\nThống kê {name}:")
    print(df.describe())
    print(f"\nKiểm tra giá trị bị thiếu {name}:")
    print(df.isnull().sum())


def visualize_distribution(df, column, title, xlabel, ylabel):
    """Trực quan hóa phân phối của một cột dữ liệu."""
    plt.figure(figsize=(10, 6))
    df[column].value_counts().sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_correlation_matrix(df, title):
    """Trực quan hóa ma trận tương quan."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.title(title)
    plt.show()


def visualize_continuous_features(df, columns, title_prefix):
    for col in columns:
        plt.figure(figsize=(8, 5))
        df[col].hist(bins=30)
        plt.title(f"{title_prefix} - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()




