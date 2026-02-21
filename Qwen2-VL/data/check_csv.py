import pandas as pd
from pathlib import Path
import os

# 讀取 NegBench 的 CSV
csv_path = './negbench.csv'
df = pd.read_csv(csv_path)

print(f"總共有 {len(df)} 筆資料")
print(f"\n前幾筆的 filepath:")
print(df['filepath'].head())

# 檢查 COCO 圖片是否存在
coco_base = '/home/sc305/VLM/Qwen2-VL/data/val2017'

if os.path.exists(coco_base):
    print(f"\n✅ COCO 目錄存在: {coco_base}")
    num_images = len(list(Path(coco_base).glob('*.jpg')))
    print(f"   找到 {num_images} 張圖片")
else:
    print(f"\n❌ COCO 目錄不存在: {coco_base}")
    print("   請先下載 COCO 2017 validation set")

# 根據 image_id 構建新的路徑
df['new_filepath'] = df['image_id'].apply(
    lambda x: f'/home/sc305/VLM/Qwen2-VL/data/val2017/{x:012d}.jpg'
)

# 驗證前 10 張圖片是否存在
print("\n驗證前 10 張圖片:")
for idx, row in df.head(10).iterrows():
    img_path = row['new_filepath']
    exists = os.path.exists(img_path)
    status = "✅" if exists else "❌"
    print(f"{status} {Path(img_path).name}")

# 保存更新後的 CSV
output_path = './negbench_with_coco.csv'
df.to_csv(output_path, index=False)
print(f"\n💾 已保存更新後的 CSV 到: {output_path}")

# 統計 negative_objects 的分佈
print("\n📊 Negative objects 統計:")
print(df['negative_objects'].value_counts().head(10))