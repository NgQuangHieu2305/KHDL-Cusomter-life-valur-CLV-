import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import urllib

from sqlalchemy import create_engine
import pyodbc

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
sns.set_theme(style="whitegrid")


# =========================================================
# 0. KẾT NỐI SQL SERVER
# =========================================================

SERVER_NAME = r".\SQLEXPRESS"
DATABASE_NAME = "khdl_project"
SQL_USER = "sa"
SQL_PASSWORD = "123456"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    f"UID={SQL_USER};"
    f"PWD={SQL_PASSWORD};"
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

conn_str = (
    "Driver={SQL Server};"
    f"Server={SERVER_NAME};"
    f"Database={DATABASE_NAME};"
    f"UID={SQL_USER};"
    f"PWD={SQL_PASSWORD};"
)

print("=== MAIN CLV PIPELINE: EDA + RFM + CLUSTERING + TUNED MACHINE LEARNING ===")


# =========================================================
# PHẦN 1: EDA
# =========================================================

print("\n" + "=" * 70)
print("PHẦN 1: EDA")
print("=" * 70)

query_eda = """
SELECT 
    t.transaction_id, 
    t.date, 
    t.customer_id, 
    c.age, 
    p.category, 
    t.payment_method, 
    t.total_sale_amount, 
    t.return_status 
FROM dbo.Transactions t
JOIN dbo.Customers c ON t.customer_id = c.customer_id
JOIN dbo.Products p ON t.product_id = p.product_id;
"""

try:
    df_eda = pd.read_sql(query_eda, engine)
    df_eda["date"] = pd.to_datetime(df_eda["date"])
    print(f"-> Đã đọc thành công {len(df_eda)} dòng dữ liệu giao dịch.")
except Exception as e:
    print("Lỗi khi đọc dữ liệu EDA từ SQL:", e)
    exit()


# ---------------------------------------------------------
# 1.1. Thống kê tổng quan
# ---------------------------------------------------------

total_transactions = df_eda["transaction_id"].nunique()
total_customers = df_eda["customer_id"].nunique()
total_revenue = df_eda["total_sale_amount"].sum()

print("\n===== THỐNG KÊ TỔNG QUAN =====")
print(f"Tổng số giao dịch: {total_transactions}")
print(f"Tổng số khách hàng: {total_customers}")
print(f"Tổng doanh thu: {total_revenue:,.2f}")

summary_df = pd.DataFrame({
    "Chỉ số": ["Số giao dịch", "Số khách hàng", "Tổng doanh thu"],
    "Giá trị": [total_transactions, total_customers, round(total_revenue, 2)]
})

print("\nBảng thống kê tổng quan:")
print(summary_df)


# ---------------------------------------------------------
# 1.2. Biểu đồ EDA 1: Số lượng giao dịch và khách hàng
# ---------------------------------------------------------

overview_df = pd.DataFrame({
    "Chỉ số": ["Số giao dịch", "Số khách hàng"],
    "Số lượng": [total_transactions, total_customers]
})

plt.figure(figsize=(7, 4))
sns.barplot(
    data=overview_df,
    x="Chỉ số",
    y="Số lượng"
)
plt.title("Số lượng giao dịch và số lượng khách hàng", fontsize=14, fontweight="bold")
plt.xlabel("")
plt.ylabel("Số lượng")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1.3. Biểu đồ EDA 2: Returned / Not Returned
# ---------------------------------------------------------

plt.figure(figsize=(7, 4))
sns.countplot(
    data=df_eda,
    x="return_status"
)
plt.title("Phân bố trạng thái giao dịch", fontsize=14, fontweight="bold")
plt.xlabel("Trạng thái giao dịch")
plt.ylabel("Số lượng giao dịch")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1.4. Biểu đồ EDA 3: Doanh thu theo thời gian
# ---------------------------------------------------------

revenue_by_date = (
    df_eda.groupby(df_eda["date"].dt.to_period("M"))["total_sale_amount"]
    .sum()
    .reset_index()
)

revenue_by_date["date"] = revenue_by_date["date"].astype(str)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=revenue_by_date,
    x="date",
    y="total_sale_amount",
    marker="o"
)
plt.title("Doanh thu theo thời gian", fontsize=14, fontweight="bold")
plt.xlabel("Thời gian")
plt.ylabel("Tổng doanh thu ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1.5. Biểu đồ EDA 4: Doanh thu theo danh mục sản phẩm
# ---------------------------------------------------------

sales_by_category = (
    df_eda.groupby("category")["total_sale_amount"]
    .sum()
    .reset_index()
    .sort_values(by="total_sale_amount", ascending=False)
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=sales_by_category,
    x="category",
    y="total_sale_amount"
)
plt.title("Tổng doanh thu theo danh mục sản phẩm", fontsize=14, fontweight="bold")
plt.xlabel("Danh mục sản phẩm")
plt.ylabel("Tổng doanh thu ($)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1.6. Biểu đồ EDA 5: Hành vi mua hàng trên dữ liệu gốc
# ---------------------------------------------------------

purchase_frequency = (
    df_eda.groupby("customer_id")["transaction_id"]
    .nunique()
    .reset_index()
)

purchase_frequency.columns = ["customer_id", "purchase_count"]

plt.figure(figsize=(8, 5))
sns.histplot(
    purchase_frequency["purchase_count"],
    bins=20,
    kde=False
)
plt.title("Phân bố số lần giao dịch của khách hàng trên dữ liệu gốc", fontsize=14, fontweight="bold")
plt.xlabel("Số lần giao dịch")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 1.7. Biểu đồ bổ sung: Phân phối độ tuổi
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    df_eda["age"],
    bins=20,
    kde=True
)
plt.title("Phân phối độ tuổi khách hàng", fontsize=14, fontweight="bold")
plt.xlabel("Độ tuổi")
plt.ylabel("Số lượng giao dịch")
plt.tight_layout()
plt.show()

print("\n=== HOÀN TẤT PHẦN 1 - EDA ===")


# =========================================================
# PHẦN 2: RFM + FEATURE ENGINEERING
# =========================================================

print("\n" + "=" * 70)
print("PHẦN 2: RFM + FEATURE ENGINEERING")
print("=" * 70)

try:
    conn = pyodbc.connect(conn_str)

    query_rfm = """
    SELECT *
    FROM Customer_RFM_Final;
    """

    df_rfm = pd.read_sql(query_rfm, conn)
    conn.close()

    print("\n[1] Đọc bảng Customer_RFM_Final thành công")
    print("Số dòng:", df_rfm.shape[0])
    print("Số cột:", df_rfm.shape[1])
    print(df_rfm.head())

except Exception as e:
    print("Lỗi khi đọc bảng Customer_RFM_Final:", e)
    print("Kiểm tra lại xem bảng Customer_RFM_Final đã được tạo trong SQL chưa.")
    exit()


# ---------------------------------------------------------
# 2.1. Kiểm tra dữ liệu
# ---------------------------------------------------------

print("\n[2] Kiểm tra missing values:")
print(df_rfm.isnull().sum())

print("\n[3] Thống kê mô tả:")
print(df_rfm.describe())

if "RFM_Level" in df_rfm.columns:
    print("\n[4] Phân bố nhóm RFM:")
    print(df_rfm["RFM_Level"].value_counts())

print("\n[5] Phân bố Frequency:")
print(df_rfm["Frequency"].value_counts().sort_index())


# ---------------------------------------------------------
# 2.2. Min-Max Scaling cho bảng RFM
# ---------------------------------------------------------

features_to_scale = [
    "Recency",
    "Frequency",
    "Monetary_CLV",
    "Avg_Order_Value",
    "Total_Quantity",
    "Avg_Quantity",
    "Customer_Lifespan",
    "RFM_Total_Score"
]

if "Discount_Usage_Rate" in df_rfm.columns:
    features_to_scale.append("Discount_Usage_Rate")

missing_scale_cols = [col for col in features_to_scale if col not in df_rfm.columns]

if missing_scale_cols:
    print("Thiếu các cột cần scale:", missing_scale_cols)
else:
    df_scaled = df_rfm.copy()

    scaler_rfm = MinMaxScaler()
    df_scaled[features_to_scale] = scaler_rfm.fit_transform(df_rfm[features_to_scale])

    # Đảo chiều Recency để giá trị càng cao nghĩa là khách mua càng gần đây
    df_scaled["Recency"] = 1 - df_scaled["Recency"]

    print("\n[6] Đã chuẩn hóa Min-Max Scaling cho RFM")
    print(df_scaled[features_to_scale].head())


# ---------------------------------------------------------
# 2.3. Biểu đồ RFM 1: Frequency
# ---------------------------------------------------------

plt.figure(figsize=(8, 4))
sns.histplot(df_rfm["Frequency"], bins=10)
plt.title("Phân bố Frequency sau khi lọc giao dịch hợp lệ", fontsize=14, fontweight="bold")
plt.xlabel("Frequency - số lần mua hợp lệ")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2.4. Biểu đồ RFM 2: Recency
# ---------------------------------------------------------

plt.figure(figsize=(8, 4))
sns.histplot(df_rfm["Recency"], bins=20)
plt.title("Phân bố Recency của khách hàng", fontsize=14, fontweight="bold")
plt.xlabel("Recency - số ngày từ lần mua gần nhất")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2.5. Biểu đồ RFM 3: Monetary CLV
# ---------------------------------------------------------

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_rfm["Monetary_CLV"])
plt.title("Boxplot của Monetary CLV", fontsize=14, fontweight="bold")
plt.xlabel("Monetary CLV")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2.6. Biểu đồ RFM 4: RFM Level
# ---------------------------------------------------------

if "RFM_Level" in df_rfm.columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(
        x="RFM_Level",
        data=df_rfm,
        order=["High Value", "Medium Value", "Low Value"]
    )
    plt.title("Phân bố khách hàng theo nhóm RFM", fontsize=14, fontweight="bold")
    plt.xlabel("Nhóm RFM")
    plt.ylabel("Số lượng khách hàng")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 2.7. Biểu đồ RFM 5: Heatmap tương quan
# ---------------------------------------------------------

corr_cols = [
    "Recency",
    "Frequency",
    "Monetary_CLV",
    "Avg_Order_Value",
    "Total_Quantity",
    "Avg_Quantity",
    "Customer_Lifespan",
    "RFM_Total_Score"
]

if "Discount_Usage_Rate" in df_rfm.columns:
    corr_cols.append("Discount_Usage_Rate")

existing_corr_cols = [col for col in corr_cols if col in df_rfm.columns]

plt.figure(figsize=(10, 7))
sns.heatmap(
    df_rfm[existing_corr_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Ma trận tương quan giữa RFM và các đặc trưng mở rộng", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n=== HOÀN TẤT PHẦN 2 - RFM + FEATURE ENGINEERING ===")


# =========================================================
# PHẦN 3: CLUSTERING + MACHINE LEARNING
# =========================================================

print("\n" + "=" * 70)
print("PHẦN 3: CLUSTERING + MACHINE LEARNING")
print("=" * 70)

df_ml = df_rfm.copy()


# ---------------------------------------------------------
# 3.1. Scaling cho K-Means
# ---------------------------------------------------------

cluster_features = [
    "Recency",
    "Frequency",
    "Avg_Quantity",
    "Customer_Lifespan"
]

if "Discount_Usage_Rate" in df_ml.columns:
    cluster_features.append("Discount_Usage_Rate")

missing_cluster_cols = [col for col in cluster_features if col not in df_ml.columns]

if missing_cluster_cols:
    print("Thiếu các cột dùng cho clustering:", missing_cluster_cols)
    exit()

scaler_cluster = MinMaxScaler()
X_cluster_scaled = scaler_cluster.fit_transform(df_ml[cluster_features])

print("\n[1] Đã scale dữ liệu cho K-Means.")
print("Các feature dùng cho clustering:", cluster_features)


# ---------------------------------------------------------
# 3.2. K-Means Clustering
# ---------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df_ml["ML_Cluster"] = kmeans.fit_predict(X_cluster_scaled)

print("\n[2] Đã tạo nhãn ML_Cluster.")
print("Phân bố số khách hàng theo cụm:")
print(df_ml["ML_Cluster"].value_counts().sort_index())


# ---------------------------------------------------------
# 3.3. Chuẩn bị dữ liệu cho mô hình ML
# ---------------------------------------------------------

drop_cols = [
    "CustomerID",
    "Customer_ID",
    "Monetary_CLV",
    "RFM_Total_Score",
    "RFM_Level",
    "RFM_Code",
    "R_Score",
    "F_Score",
    "M_Score",
    "Avg_Order_Value",
    "Total_Quantity"
]

df_model = df_ml.copy()

df_model = pd.get_dummies(
    df_model,
    columns=["ML_Cluster"],
    drop_first=True
)

X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])
X = X.select_dtypes(exclude=["object", "string"])

if "Monetary_CLV" not in df_model.columns:
    print("Không tìm thấy cột Monetary_CLV trong dữ liệu.")
    exit()

y = df_model["Monetary_CLV"]
y_log = np.log1p(y)

print("\n[3] Các feature được dùng để train model:")
print(list(X.columns))
print("Số lượng feature đầu vào:", X.shape[1])


# ---------------------------------------------------------
# 3.4. Train/Test Split và Scaling
# ---------------------------------------------------------

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

scaler_final = MinMaxScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

print("\n[4] Đã chia train/test và scale dữ liệu đúng cách.")
print(f"Train size: {X_train_scaled.shape[0]} mẫu")
print(f"Test size: {X_test_scaled.shape[0]} mẫu")


# ---------------------------------------------------------
# 3.5. Tối ưu tham số cho các mô hình bằng Grid Search
# ---------------------------------------------------------

print("\n[5] Đang tối ưu tham số cho các mô hình bằng Grid Search + Cross Validation...")

# Random Forest
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


# Gradient Boosting
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


# Extra Trees
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

print("\n[5] Hoàn tất tối ưu tham số.")


# ---------------------------------------------------------
# 3.6. Khởi tạo các mô hình đã tối ưu
# ---------------------------------------------------------

models = {
    "1. Random Forest - Tuned": best_random_forest,
    "2. Gradient Boosting - Tuned": best_gradient_boosting,
    "3. Extra Trees - Tuned": best_extra_trees,
    "4. Linear Regression": LinearRegression()
}

results = []
predictions_store = {}


# ---------------------------------------------------------
# 3.7. Train model + Evaluation
# ---------------------------------------------------------

print("\n[6] Đang huấn luyện và đánh giá các mô hình đã tối ưu...")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

fig.suptitle(
    "CLV thực tế và CLV dự đoán",
    fontsize=16,
    fontweight="bold",
    y=1.05
)

for idx, (name, model) in enumerate(models.items()):
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

    ax.set_xlabel("CLV thực tế ($)")
    ax.set_ylabel("CLV dự đoán ($)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)


results_df = pd.DataFrame(results).sort_values(
    by="R-squared",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("BẢNG TỔNG KẾT SAI SỐ CÁC MÔ HÌNH SAU TỐI ƯU")
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


# ---------------------------------------------------------
# 3.8. Feature Importance cho Extra Trees đã tối ưu
# ---------------------------------------------------------

print("\n[7] Mức độ quan trọng của các đặc trưng - Extra Trees Tuned")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_extra_trees.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.to_string(index=False))

plt.figure(figsize=(8, 5))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Mức độ quan trọng của các đặc trưng trong Extra Trees", fontsize=14, fontweight="bold")
plt.xlabel("Mức độ quan trọng")
plt.ylabel("Đặc trưng")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3.9. Biểu đồ so sánh R-squared giữa các mô hình
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R-squared"
)

plt.title("So sánh R-squared giữa các mô hình", fontsize=14, fontweight="bold")
plt.xlabel("Mô hình")
plt.ylabel("R-squared")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3.10. Actual vs Predicted cho mô hình tốt nhất
# ---------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_pred_df = predictions_store[best_model_name]

plt.figure(figsize=(7, 6))

plt.scatter(
    best_pred_df["Actual_CLV"],
    best_pred_df["Predicted_CLV"],
    alpha=0.6,
    edgecolor="white",
    s=50
)

max_val = max(
    best_pred_df["Actual_CLV"].max(),
    best_pred_df["Predicted_CLV"].max()
)

plt.plot(
    [0, max_val],
    [0, max_val],
    linestyle="--",
    linewidth=2,
    label="Đường lý tưởng"
)

plt.title(
    f"CLV thực tế và CLV dự đoán - {best_model_name}",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("CLV thực tế ($)")
plt.ylabel("CLV dự đoán ($)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.show()


print("\n=== HOÀN TẤT TOÀN BỘ PIPELINE CLV ĐÃ TỐI ƯU MODEL ===")