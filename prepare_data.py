import pandas as pd
from datasets import Dataset, DatasetDict
import json

def create_medical_dataset():
    """สร้าง Dataset จำลองสำหรับทดสอบ - ในการใช้งานจริงให้ใช้ข้อมูลของคุณ"""
    
    # ตัวอย่างข้อมูลการแปลทางการแพทย์
    data = {
        'chinese': [
            '患者主诉头痛三天',
            '血压测量结果为120/80毫米汞柱',
            '建议每日服用阿司匹林100毫克',
            '体温37.5摄氏度，属于低烧',
            '心电图显示窦性心律',
            '胸部X光检查未见异常',
            '血糖水平为95毫克/分升',
            '患者有高血压病史',
            '建议进行血常规检查',
            '诊断为急性上呼吸道感染'
        ],
        'thai': [
            'ผู้ป่วยมีอาการปวดหัวมา 3 วัน',
            'ผลการวัดความดันโลหิต 120/80 มิลลิเมตรปรอท',
            'แนะนำให้รับประทานแอสไพริน 100 มิลลิกรัมต่อวัน',
            'อุณหภูมิร่างกาย 37.5 องศาเซลเซียส ถือเป็นไข้ต่ำ',
            'ผลคลื่นไฟฟ้าหัวใจแสดงจังหวะปกติ',
            'การเอกซเรย์ปอดไม่พบความผิดปกติ',
            'ระดับน้ำตาลในเลือด 95 มิลลิกรัมต่อเดซิลิตร',
            'ผู้ป่วยมีประวัติความดันโลหิตสูง',
            'แนะนำให้ตรวจนับเม็ดเลือด',
            'วินิจฉัยเป็นการติดเชื้อทางเดินหายใจส่วนบนเฉียบพลัน'
        ]
    }
    
    # สร้าง CSV file
    df = pd.DataFrame(data)
    
    # เพิ่มข้อมูลให้มากขึ้นโดยการ duplicate และ shuffle
    df_expanded = pd.concat([df] * 200, ignore_index=True)  # ขยายเป็น 2000 rows
    df_expanded = df_expanded.sample(frac=1).reset_index(drop=True)
    
    # แบ่งข้อมูล 80:10:10
    train_size = int(0.8 * len(df_expanded))
    val_size = int(0.1 * len(df_expanded))
    
    train_df = df_expanded[:train_size]
    val_df = df_expanded[train_size:train_size+val_size]
    test_df = df_expanded[train_size+val_size:]
    
    # บันทึกเป็น CSV
    train_df.to_csv('datasets/train.csv', index=False)
    val_df.to_csv('datasets/val.csv', index=False)
    test_df.to_csv('datasets/test.csv', index=False)
    
    print(f"สร้าง Dataset เรียบร้อย:")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # สร้าง HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # บันทึก Dataset
    dataset_dict.save_to_disk('datasets/chinese_thai_medical')
    
    return dataset_dict

def load_custom_dataset(train_path, val_path, test_path):
    """โหลด Dataset จากไฟล์ CSV ของคุณเอง"""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # ตรวจสอบว่ามี columns chinese และ thai
    required_cols = ['chinese', 'thai']
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"{name} dataset ต้องมี columns: {required_cols}")
    
    # สร้าง HuggingFace Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    dataset_dict.save_to_disk('datasets/chinese_thai_medical')
    return dataset_dict

if __name__ == "__main__":
    import os
    os.makedirs('datasets', exist_ok=True)
    
    # ใช้อันนี้สำหรับสร้าง dataset ทดสอบ
    dataset = create_medical_dataset()
    
    # หรือใช้อันนี้สำหรับโหลดจาก CSV ของคุณ
    # dataset = load_custom_dataset(
    #     'path/to/train.csv',
    #     'path/to/val.csv', 
    #     'path/to/test.csv'
    # )
