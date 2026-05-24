import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import urllib

# ==========================================
# 1. KẾT NỐI PYTHON VỚI SQL SERVER
# ==========================================

server_name = r'.\SQLEXPRESS'
database_name = 'khdl_project'

print(f"Đang kết nối tới SQL Server: {server_name}...")

params = urllib.parse.quote_plus(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server_name};'
    f'DATABASE={database_name};'
    f'UID=sa;'
    f'PWD=123456;'
)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# ==========================================
# 2. TRUY VẤN DỮ LIỆU TỪ DATABASE ĐÃ CHUẨN HÓA
# ==========================================

print("Đang kéo dữ liệu bằng JOIN từ các bảng...")

query = """
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

df = pd.read_sql(query, engine)
df["date"] = pd.to_datetime(df["date"])

print(f"Thành công! Đã kéo về {len(df)} dòng dữ liệu.")

# ==========================================
# 3. THỐNG KÊ TỔNG QUAN
# ==========================================

total_transactions = df["transaction_id"].nunique()
total_customers = df["customer_id"].nunique()
total_revenue = df["total_sale_amount"].sum()

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

sns.set_theme(style="whitegrid")

# ==========================================
# 4. BIỂU ĐỒ 1: SỐ GIAO DỊCH VÀ SỐ KHÁCH HÀNG
# ==========================================

plt.figure(figsize=(7, 4))

overview_df = pd.DataFrame({
    "Chỉ số": ["Số giao dịch", "Số khách hàng"],
    "Số lượng": [total_transactions, total_customers]
})

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

# ==========================================
# 5. BIỂU ĐỒ 2: PHÂN BỐ RETURNED / NOT RETURNED
# ==========================================

plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="return_status"
)

plt.title("Phân bố trạng thái giao dịch", fontsize=14, fontweight="bold")
plt.xlabel("Trạng thái giao dịch")
plt.ylabel("Số lượng giao dịch")
plt.tight_layout()
plt.show()

# ==========================================
# 6. BIỂU ĐỒ 3: DOANH THU THEO THỜI GIAN
# ==========================================

revenue_by_date = (
    df.groupby(df["date"].dt.to_period("M"))["total_sale_amount"]
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

# ==========================================
# 7. BIỂU ĐỒ 4: DOANH THU THEO DANH MỤC SẢN PHẨM
# ==========================================

sales_by_category = (
    df.groupby("category")["total_sale_amount"]
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

# ==========================================
# 8. BIỂU ĐỒ 5: HÀNH VI MUA HÀNG CỦA KHÁCH HÀNG
#    Dựa trên số lần mua của từng khách hàng
# ==========================================

purchase_frequency = (
    df.groupby("customer_id")["transaction_id"]
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

plt.title("Phân bố số lần mua của khách hàng", fontsize=14, fontweight="bold")
plt.xlabel("Số lần mua")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()

# ==========================================
# 9. BIỂU ĐỒ 6: PHÂN PHỐI ĐỘ TUỔI KHÁCH HÀNG
#    Có thể dùng thêm nếu muốn EDA đầy đủ hơn
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    bins=20,
    kde=True
)

plt.title("Phân phối độ tuổi khách hàng", fontsize=14, fontweight="bold")
plt.xlabel("Độ tuổi")
plt.ylabel("Số lượng giao dịch")
plt.tight_layout()
plt.show()

print("\n=== HOÀN TẤT EDA ===")