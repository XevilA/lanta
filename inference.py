import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

class ChineseThaiMedicalTranslator:
    def __init__(self, model_path="./final_model"):
        """โหลดโมเดลและ tokenizer"""
        print(f"กำลังโหลดโมเดลจาก {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"โมเดลพร้อมใช้งานบน {self.device}")
    
    def translate(self, text, max_length=128, num_beams=5, temperature=1.0):
        """แปลข้อความจากภาษาจีนเป็นไทย"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(self, texts, batch_size=8, **kwargs):
        """แปลข้อความหลายประโยคพร้อมกัน"""
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True
                )
            
            # Decode batch
            batch_translations = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            translations.extend(batch_translations)
        
        return translations

def main():
    # สร้าง translator
    translator = ChineseThaiMedicalTranslator()
    
    # ตัวอย่างการแปล
    test_sentences = [
        "患者主诉头痛三天",
        "血压测量结果为120/80毫米汞柱",
        "建议每日服用阿司匹林100毫克",
        "体温37.5摄氏度，属于低烧",
        "心电图显示窦性心律"
    ]
    
    print("\n=== ทดสอบการแปลทีละประโยค ===")
    for sentence in test_sentences:
        start_time = time.time()
        translation = translator.translate(sentence)
        end_time = time.time()
        
        print(f"\nจีน: {sentence}")
        print(f"ไทย: {translation}")
        print(f"เวลา: {end_time - start_time:.2f} วินาที")
    
    print("\n=== ทดสอบการแปล Batch ===")
    start_time = time.time()
    batch_translations = translator.translate_batch(test_sentences)
    end_time = time.time()
    
    print(f"\nแปล {len(test_sentences)} ประโยคใน {end_time - start_time:.2f} วินาที")
    for orig, trans in zip(test_sentences, batch_translations):
        print(f"\nจีน: {orig}")
        print(f"ไทย: {trans}")
    
    # Interactive mode
    print("\n=== โหมดแปลแบบ Interactive ===")
    print("พิมพ์ 'quit' เพื่อออก")
    
    while True:
        chinese_text = input("\nใส่ข้อความภาษาจีน: ")
        if chinese_text.lower() == 'quit':
            break
        
        translation = translator.translate(chinese_text)
        print(f"ผลการแปล: {translation}")

if __name__ == "__main__":
    main()
