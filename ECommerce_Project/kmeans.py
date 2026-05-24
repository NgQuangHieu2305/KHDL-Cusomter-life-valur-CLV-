import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# =========================================================
# 1. KẾT NỐI SQL SERVER VÀ ĐỌC DỮ LIỆU
# =========================================================

print("=== K-MEANS CLUSTERING VISUALIZATION 3D ===\n")

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
    print("Lỗi kết nối SQL:", e)
    exit()


# =========================================================
# 2. CHỌN FEATURE CHO K-MEANS
# =========================================================

cluster_features = [
    "Recency",
    "Frequency",
    "Avg_Quantity",
    "Customer_Lifespan"
]

# Nếu có Discount_Usage_Rate thì thêm vào
if "Discount_Usage_Rate" in df.columns:
    cluster_features.append("Discount_Usage_Rate")

missing_cols = [col for col in cluster_features if col not in df.columns]

if missing_cols:
    print("Thiếu cột dùng cho K-Means:", missing_cols)
    exit()

print("\nCác feature dùng cho K-Means:")
print(cluster_features)


# =========================================================
# 3. SCALE DỮ LIỆU
# =========================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[cluster_features])

df_scaled = pd.DataFrame(
    X_scaled,
    columns=cluster_features
)

print("\nĐã chuẩn hóa dữ liệu về khoảng 0–1.")


# =========================================================
# 4. ELBOW METHOD - GIẢI THÍCH CHỌN K
# =========================================================

print("\nĐang vẽ Elbow Method...")

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_range),
    inertias,
    marker="o"
)

plt.title("Elbow Method để lựa chọn số cụm K", fontsize=14, fontweight="bold")
plt.xlabel("Số cụm K")
plt.ylabel("Inertia / SSE")
plt.xticks(list(K_range))
plt.tight_layout()
plt.show()


# =========================================================
# 5. CHẠY K-MEANS VỚI K = 3
# =========================================================

print("\nĐang chạy K-Means với K = 3...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["ML_Cluster"] = kmeans.fit_predict(X_scaled)
df_scaled["ML_Cluster"] = df["ML_Cluster"]

print("\nPhân bố khách hàng theo cụm:")
print(df["ML_Cluster"].value_counts().sort_index())


# =========================================================
# 6. BIỂU ĐỒ SỐ LƯỢNG KHÁCH HÀNG THEO CỤM
# =========================================================

plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="ML_Cluster",
    order=sorted(df["ML_Cluster"].unique())
)

plt.title("Số lượng khách hàng theo từng cụm K-Means", fontsize=14, fontweight="bold")
plt.xlabel("Cụm khách hàng")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()


# =========================================================
# 7. PCA 3D - TRỰC QUAN HÓA CỤM KHÁCH HÀNG
# =========================================================

print("\nĐang giảm chiều dữ liệu bằng PCA 3D...")

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

df_pca_3d = pd.DataFrame({
    "PCA_1": pca_result[:, 0],
    "PCA_2": pca_result[:, 1],
    "PCA_3": pca_result[:, 2],
    "ML_Cluster": df["ML_Cluster"]
})

print("\nTỷ lệ phương sai được giải thích bởi từng PCA component:")
print(pca.explained_variance_ratio_)

print("Tổng tỷ lệ phương sai được giải thích bởi 3 component:")
print(round(pca.explained_variance_ratio_.sum(), 4))


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    df_pca_3d["PCA_1"],
    df_pca_3d["PCA_2"],
    df_pca_3d["PCA_3"],
    c=df_pca_3d["ML_Cluster"],
    s=50,
    alpha=0.75
)

ax.set_title(
    "Trực quan hóa phân cụm khách hàng bằng PCA 3D",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")

legend = ax.legend(
    *scatter.legend_elements(),
    title="Cụm"
)

ax.add_artist(legend)

plt.tight_layout()
plt.show()


# =========================================================
# 8. CLUSTER PROFILE - ĐẶC ĐIỂM TRUNG BÌNH TỪNG CỤM
# =========================================================

cluster_profile = (
    df.groupby("ML_Cluster")[cluster_features]
    .mean()
    .reset_index()
)

print("\nĐặc điểm trung bình của từng cụm trên dữ liệu gốc:")
print(cluster_profile.to_string(index=False))

df["ML_Cluster"] = kmeans.fit_predict(X_scaled)
df_scaled["ML_Cluster"] = df["ML_Cluster"]

print("\nPhân bố khách hàng theo cụm:")
print(df["ML_Cluster"].value_counts().sort_index())

# Dùng dữ liệu đã scale để heatmap dễ so sánh hơn
cluster_profile_scaled = (
    df_scaled.groupby("ML_Cluster")[cluster_features]
    .mean()
)

plt.figure(figsize=(10, 5))

sns.heatmap(
    cluster_profile_scaled,
    annot=True,
    cmap="YlGnBu",
    fmt=".2f"
)

plt.title(
    "Đặc điểm trung bình của từng cụm sau khi chuẩn hóa",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Feature")
plt.ylabel("Cụm khách hàng")
plt.tight_layout()
plt.show()


# =========================================================
# 9. LƯU KẾT QUẢ PHÂN CỤM
# =========================================================

output_file = "customer_clusters_result.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu kết quả phân cụm vào file: {output_file}")
print("\n=== HOÀN TẤT K-MEANS VISUALIZATION 3D ===")