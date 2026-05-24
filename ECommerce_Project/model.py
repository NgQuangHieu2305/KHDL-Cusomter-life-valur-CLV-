import pandas as pd
import numpy as np
import pyodbc
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =========================================================
# 1. KẾT NỐI SQL SERVER VÀ ĐỌC BẢNG CUSTOMER_RFM_FINAL
# =========================================================

print("=== CLV MODEL PIPELINE: SCALING + CLUSTERING + TUNED MODEL ===\n")

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

    print(f"Đọc dữ liệu thành công: {df.shape[0]} khách hàng, {df.shape[1]} cột.")
except Exception as e:
    print("Lỗi kết nối hoặc đọc dữ liệu SQL:", e)
    exit()


# =========================================================
# 3. SCALE DỮ LIỆU CHO K-MEANS CLUSTERING
# =========================================================

print("\n[1] Đang chuẩn hóa dữ liệu cho K-Means...")

cluster_features = [
    "Recency",
    "Frequency",
    "Avg_Quantity",
    "Customer_Lifespan"
]

if "Discount_Usage_Rate" in df.columns:
    cluster_features.append("Discount_Usage_Rate")

missing_cluster_cols = [col for col in cluster_features if col not in df.columns]

if missing_cluster_cols:
    print("Thiếu các cột dùng cho clustering:", missing_cluster_cols)
    exit()

scaler_cluster = MinMaxScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df[cluster_features])

print("Đã scale dữ liệu clustering về khoảng 0–1.")
print("Feature dùng cho clustering:")
print(cluster_features)


# =========================================================
# 4. K-MEANS CLUSTERING
# =========================================================

print("\n[2] Đang phân cụm khách hàng bằng K-Means...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["ML_Cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("Đã tạo cột ML_Cluster.")
print("\nPhân bố khách hàng theo cụm:")
print(df["ML_Cluster"].value_counts().sort_index())


# =========================================================
# 5. CHUẨN BỊ DỮ LIỆU CHO MODEL
# =========================================================

print("\n[3] Đang chuẩn bị dữ liệu train model...")

drop_cols = [
    # ID khách hàng
    "CustomerID",
    "Customer_ID",
    "customer_id",

    # Target cần dự đoán
    "Monetary_CLV",

    # Biến có nguy cơ data leakage
    "RFM_Total_Score",
    "RFM_Level",
    "RFM_Code",
    "R_Score",
    "F_Score",
    "M_Score",

    # Biến liên quan trực tiếp đến Monetary_CLV
    "Avg_Order_Value",
    "Total_Quantity",

    # 3 feature mới đã thử nhưng không dùng trong bản cuối
    "Return_Rate",
    "Category_Count",
    "Avg_Days_Between_Purchases"
]

df_model = df.copy()

# One-hot encoding cho ML_Cluster
df_model = pd.get_dummies(
    df_model,
    columns=["ML_Cluster"],
    drop_first=True
)

if "Monetary_CLV" not in df_model.columns:
    print("Không tìm thấy cột Monetary_CLV.")
    exit()

# X là feature đầu vào
X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])

# Chỉ giữ cột số
X = X.select_dtypes(include=[np.number])

# y là target
y = df_model["Monetary_CLV"]

# Log transform target để giảm lệch phân phối tiền
y_log = np.log1p(y)

print("\nCác feature được dùng để train model:")
print(list(X.columns))
print("Số lượng feature đầu vào:", X.shape[1])


# =========================================================
# 6. TRAIN / TEST SPLIT
# =========================================================

print("\n[4] Đang chia dữ liệu train/test...")

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

print(f"Train size: {X_train.shape[0]} mẫu")
print(f"Test size: {X_test.shape[0]} mẫu")


# =========================================================
# 7. SCALE SAU KHI CHIA TRAIN/TEST
# =========================================================

print("\n[5] Đang scale dữ liệu train/test...")

scaler_final = MinMaxScaler()

X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

print("Đã scale đúng cách: fit trên train, transform trên test.")


# =========================================================
# 8. TỐI ƯU THAM SỐ MODEL BẰNG GRID SEARCH
# =========================================================

print("\n[6] Đang tối ưu tham số các mô hình bằng Grid Search + Cross Validation...")

# -----------------------------
# Random Forest
# -----------------------------

rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

rf_grid.fit(X_train_scaled, y_train_log)
best_random_forest = rf_grid.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)
print("Best Random Forest CV R²:", round(rf_grid.best_score_, 4))


# -----------------------------
# Gradient Boosting
# -----------------------------

gb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_samples_leaf": [1, 2]
}

gb_grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

gb_grid.fit(X_train_scaled, y_train_log)
best_gradient_boosting = gb_grid.best_estimator_

print("\nBest Gradient Boosting Parameters:")
print(gb_grid.best_params_)
print("Best Gradient Boosting CV R²:", round(gb_grid.best_score_, 4))


# -----------------------------
# Extra Trees
# -----------------------------

et_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
}

et_grid = GridSearchCV(
    estimator=ExtraTreesRegressor(random_state=42),
    param_grid=et_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

et_grid.fit(X_train_scaled, y_train_log)
best_extra_trees = et_grid.best_estimator_

print("\nBest Extra Trees Parameters:")
print(et_grid.best_params_)
print("Best Extra Trees CV R²:", round(et_grid.best_score_, 4))


# Bảng tổng hợp tham số tối ưu
best_params_df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
    "Best Parameters": [
        str(rf_grid.best_params_),
        str(gb_grid.best_params_),
        str(et_grid.best_params_)
    ],
    "Best CV R-squared": [
        round(rf_grid.best_score_, 4),
        round(gb_grid.best_score_, 4),
        round(et_grid.best_score_, 4)
    ]
})

print("\nBẢNG THAM SỐ TỐI ƯU:")
print(best_params_df.to_string(index=False))


# =========================================================
# 9. TRAIN VÀ ĐÁNH GIÁ CÁC MODEL ĐÃ TỐI ƯU
# =========================================================

print("\n[7] Đang train và đánh giá model trên tập test...")

models = {
    "Random Forest - Tuned": best_random_forest,
    "Gradient Boosting - Tuned": best_gradient_boosting,
    "Extra Trees - Tuned": best_extra_trees,
    "Linear Regression": LinearRegression()
}

results = []
predictions_store = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_log)

    y_pred_log = model.predict(X_test_scaled)

    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)

    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    results.append({
        "Model": name,
        "R-squared": round(r2, 4),
        "RMSE ($)": round(rmse, 2),
        "MAE ($)": round(mae, 2)
    })

    predictions_store[name] = pd.DataFrame({
        "Actual_CLV": y_test_real,
        "Predicted_CLV": y_pred_real
    })


results_df = pd.DataFrame(results).sort_values(
    by="R-squared",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
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


# =========================================================
# 10. FEATURE IMPORTANCE CỦA EXTRA TREES
# =========================================================

print("\n[8] Mức độ quan trọng của feature trong Extra Trees Tuned:")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_extra_trees.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.to_string(index=False))


# =========================================================
# LƯU KẾT QUẢ MODEL CHO NGƯỜI 4 ĐÁNH GIÁ
# =========================================================

results_df.to_csv(
    "model_evaluation_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu bảng kết quả model vào: model_evaluation_results.csv")

for model_name, pred_df in predictions_store.items():
    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "")
    )

    file_name = f"predictions_{safe_name}.csv"

    pred_df.to_csv(
        file_name,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Đã lưu dự đoán của {model_name} vào: {file_name}")