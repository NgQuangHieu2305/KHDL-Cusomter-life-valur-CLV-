import pandas as pd
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

print("=== PIPELINE MEMBER 3: CLUSTERING + MACHINE LEARNING CLV PREDICTION ===\n")


# =========================================================
# 1. KẾT NỐI SQL SERVER VÀ LẤY DỮ LIỆU
# =========================================================

print("[1/5] Đang lấy dữ liệu từ bảng Customer_RFM_Final...")

conn_str = (
    "Driver={SQL Server};"
    "Server=.\\SQLEXPRESS;"
    "Database=khdl_project;"
    "UID=sa;"
    "PWD=123456;"
)

query = """
SELECT *
FROM Customer_RFM_Final;
"""

try:
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"-> Lấy thành công {df.shape[0]} khách hàng và {df.shape[1]} cột dữ liệu.")
except Exception as e:
    print("Lỗi kết nối SQL:", e)
    exit()


# =========================================================
# 2. MIN-MAX SCALING CHO K-MEANS CLUSTERING
# =========================================================

print("\n[2/5] Đang chuẩn hóa dữ liệu cho K-Means bằng Min-Max Scaling...")

# Các biến dùng để phân cụm khách hàng
# Không đưa Monetary_CLV, M_Score, RFM_Total_Score vào để tránh data leakage
cluster_features = [
    "Recency",
    "Frequency",
    "Avg_Quantity",
    "Customer_Lifespan"
]

if "Discount_Usage_Rate" in df.columns:
    cluster_features.append("Discount_Usage_Rate")

# Kiểm tra cột có tồn tại không
missing_cluster_cols = [col for col in cluster_features if col not in df.columns]

if missing_cluster_cols:
    print("Thiếu các cột dùng cho clustering:", missing_cluster_cols)
    exit()

# Scale dữ liệu clustering về khoảng 0-1
scaler_cluster = MinMaxScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_features])

print("-> Đã scale dữ liệu clustering về khoảng 0-1.")
print("-> Các feature dùng cho clustering:", cluster_features)


# =========================================================
# 3. CLUSTERING BẰNG K-MEANS
# =========================================================

print("\n[3/5] Đang chạy thuật toán K-Means Clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["ML_Cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("-> Đã tạo xong nhãn phân cụm ML_Cluster.")

print("\nPhân bố số khách hàng theo cụm:")
print(df["ML_Cluster"].value_counts().sort_index())


# =========================================================
# 4. CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH MACHINE LEARNING
# =========================================================

print("\n[4/5] Đang chuẩn bị dữ liệu huấn luyện mô hình...")

# Các cột cần loại bỏ để tránh data leakage hoặc không phù hợp làm input
drop_cols = [
    # ID khách hàng
    "CustomerID",
    "Customer_ID",

    # Target cần dự đoán
    "Monetary_CLV",

    # Các biến chứa thông tin từ Monetary_CLV hoặc điểm RFM tổng hợp
    "RFM_Total_Score",
    "RFM_Level",
    "RFM_Code",
    "R_Score",
    "F_Score",
    "M_Score",

    # Các biến liên quan trực tiếp đến Monetary_CLV
    "Avg_Order_Value",
    "Total_Quantity"
]

df_model = df.copy()

# Chuyển ML_Cluster thành biến dummy 0-1
df_model = pd.get_dummies(
    df_model,
    columns=["ML_Cluster"],
    drop_first=True
)

# Tạo X
X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])

# Chỉ giữ các cột số
X = X.select_dtypes(exclude=["object", "string"])

# Biến mục tiêu
if "Monetary_CLV" not in df_model.columns:
    print("Không tìm thấy cột Monetary_CLV trong dữ liệu.")
    exit()

y = df_model["Monetary_CLV"]

# Log transform target để giảm lệch phân phối tiền
y_log = np.log1p(y)

print("-> Các feature được dùng để train model:")
print(list(X.columns))

print("\n-> Số lượng feature đầu vào:", X.shape[1])

# Chia train/test trước
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

# Scale sau khi chia train/test
# Fit scaler trên tập train, transform cả train và test
scaler_final = MinMaxScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

print("-> Đã chia train/test và scale dữ liệu đúng cách.")
print(f"-> Train size: {X_train_scaled.shape[0]} mẫu")
print(f"-> Test size: {X_test_scaled.shape[0]} mẫu")


# =========================================================
# 5. KHỞI TẠO CÁC MÔ HÌNH
# =========================================================

models = {
    "1. Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ),

    "2. Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42
    ),

    "3. Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "4. Linear Regression": LinearRegression()
}

results = []


# =========================================================
# 6. TRAIN MODEL + EVALUATION
# =========================================================

print("\n[5/5] Đang huấn luyện và đánh giá các mô hình...\n")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

fig.suptitle(
    "CLV THỰC TẾ VS. CLV DỰ ĐOÁN",
    fontsize=16,
    fontweight="bold",
    y=1.05
)

for idx, (name, model) in enumerate(models.items()):
    # Huấn luyện mô hình
    model.fit(X_train_scaled, y_train_log)

    # Dự đoán trên tập test
    y_pred_log = model.predict(X_test_scaled)

    # Đưa kết quả về đơn vị tiền thực tế
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)

    # Tính các chỉ số đánh giá
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    results.append({
        "Model": name,
        "R-squared": round(r2, 4),
        "RMSE ($)": round(rmse, 2),
        "MAE ($)": round(mae, 2)
    })

    # Vẽ biểu đồ Actual vs Predicted
    ax = axes[idx]

    ax.scatter(
        y_test_real,
        y_pred_real,
        alpha=0.6,
        color="teal",
        edgecolor="white",
        s=50
    )

    max_val = max(max(y_test_real), max(y_pred_real))

    ax.plot(
        [0, max_val],
        [0, max_val],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Đường lý tưởng"
    )

    ax.set_title(
        f"{name}\nR-squared: {r2:.4f}",
        fontsize=13,
        fontweight="bold",
        color="darkblue"
    )

    ax.set_xlabel("CLV Thực tế ($)")
    ax.set_ylabel("CLV Dự đoán ($)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)


# =========================================================
# 7. IN BẢNG KẾT QUẢ
# =========================================================

results_df = pd.DataFrame(results).sort_values(
    by="R-squared",
    ascending=False
).reset_index(drop=True)

print("=" * 70)
print("BẢNG TỔNG KẾT SAI SỐ CÁC MÔ HÌNH")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

best_model = results_df.iloc[0]

print("\nMô hình tốt nhất:")
print(f"- {best_model['Model']}")
print(f"- R-squared: {best_model['R-squared']}")
print(f"- RMSE ($): {best_model['RMSE ($)']}")
print(f"- MAE ($): {best_model['MAE ($)']}")

plt.tight_layout()
plt.show()

print("\n=== HOÀN TẤT PIPELINE MEMBER 3 ===")


