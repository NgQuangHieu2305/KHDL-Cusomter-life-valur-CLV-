import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

# Tắt các cảnh báo vặt của Python 
warnings.filterwarnings('ignore')

print("=== RFM + FEATURE ENGINEERING - MEMBER 2 ===")

# --- BƯỚC 1: KẾT NỐI SQL VÀ TẠO 6 BIẾN TRỰC TIẾP TỪ DATABASE ---
print("1. Đang kết nối SQL Server và trích xuất dữ liệu...")
conn_str = (
    "Driver={SQL Server};"
    "Server=.\\SQLEXPRESS;"
    "Database=khdl_project;"
    "UID=sa;"
    "PWD=123456;"
)
conn = pyodbc.connect(conn_str)

# ==============================
# 2. ĐỌC BẢNG RFM FINAL TỪ SQL
# ==============================

query = """
SELECT *
FROM Customer_RFM_Final;
"""

df = pd.read_sql(query, conn)

print("\n[1] Đọc dữ liệu từ SQL thành công")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])
print(df.head())

# ==============================
# 3. KIỂM TRA DỮ LIỆU
# ==============================

print("\n[2] Kiểm tra missing values:")
print(df.isnull().sum())

print("\n[3] Thống kê mô tả:")
print(df.describe())

print("\n[4] Phân bố nhóm RFM:")
print(df["RFM_Level"].value_counts())

print("\n[5] Phân bố Frequency:")
print(df["Frequency"].value_counts().sort_index())

# ==============================
# 4. CHỌN FEATURE ĐỂ SCALE
# ==============================

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

# Nếu có Discount_Usage_Rate thì thêm vào
if "Discount_Usage_Rate" in df.columns:
    features_to_scale.append("Discount_Usage_Rate")

# ==============================
# 5. MIN-MAX SCALING
# ==============================

df_scaled = df.copy()

scaler = MinMaxScaler()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Đảo chiều Recency để giá trị càng cao nghĩa là khách mua càng gần đây
df_scaled["Recency"] = 1 - df_scaled["Recency"]

print("\n[6] Đã chuẩn hóa Min-Max Scaling")
print(df_scaled[features_to_scale].head())

# ==============================
# 6. VẼ BIỂU ĐỒ
# ==============================

# 6.1 Histogram Frequency
plt.figure(figsize=(8, 4))
sns.histplot(df["Frequency"], bins=10)
plt.title("Distribution of Purchase Frequency")
plt.xlabel("Frequency")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 6.2 Histogram Recency
plt.figure(figsize=(8, 4))
sns.histplot(df["Recency"], bins=20)
plt.title("Distribution of Recency")
plt.xlabel("Recency Days")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 6.3 Boxplot Monetary CLV
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Monetary_CLV"])
plt.title("Boxplot of Monetary CLV")
plt.xlabel("Monetary CLV")
plt.tight_layout()
plt.show()

# 6.4 Countplot RFM Level
plt.figure(figsize=(7, 4))
sns.countplot(x="RFM_Level", data=df, order=["High Value", "Medium Value", "Low Value"])
plt.title("Customer Distribution by RFM Level")
plt.xlabel("RFM Level")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 6.5 Heatmap tương quan
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

if "Discount_Usage_Rate" in df.columns:
    corr_cols.append("Discount_Usage_Rate")

plt.figure(figsize=(10, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of RFM and Engineered Features")
plt.tight_layout()
plt.show()