import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# =========================================================
# 1. ĐỌC BẢNG KẾT QUẢ MODEL TỪ NGƯỜI 3
# =========================================================

results_df = pd.read_csv("model_evaluation_results.csv")

print("BẢNG KẾT QUẢ MODEL:")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]

print("\nMô hình tốt nhất:")
print(f"- {best_model['Model']}")
print(f"- R-squared: {best_model['R-squared']}")
print(f"- RMSE ($): {best_model['RMSE ($)']}")
print(f"- MAE ($): {best_model['MAE ($)']}")


# =========================================================
# 2. VẼ BIỂU ĐỒ SO SÁNH R-SQUARED
# =========================================================

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


# =========================================================
# 3. VẼ ACTUAL VS PREDICTED CHO TỪNG MODEL
# =========================================================

for _, row in results_df.iterrows():
    model_name = row["Model"]
    r2 = row["R-squared"]
    rmse = row["RMSE ($)"]
    mae = row["MAE ($)"]

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "")
    )

    file_name = f"predictions_{safe_name}.csv"

    try:
        pred_df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_name}")
        continue

    plt.figure(figsize=(7, 6))

    plt.scatter(
        pred_df["Actual_CLV"],
        pred_df["Predicted_CLV"],
        alpha=0.6,
        edgecolor="white",
        s=50
    )

    max_val = max(
        pred_df["Actual_CLV"].max(),
        pred_df["Predicted_CLV"].max()
    )

    plt.plot(
        [0, max_val],
        [0, max_val],
        linestyle="--",
        linewidth=2,
        label="Đường lý tưởng"
    )

    plt.title(
        f"{model_name}\nR² = {r2} | RMSE = {rmse} | MAE = {mae}",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel("CLV thực tế ($)")
    plt.ylabel("CLV dự đoán ($)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


print("\n=== HOÀN TẤT PHẦN ĐÁNH GIÁ MODEL CỦA NGƯỜI 4 ===")