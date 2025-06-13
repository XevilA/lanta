import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import onnx
import numpy as np
from pathlib import Path
import shutil
import zipfile
import json

def export_to_onnx(model_path="./final_model", output_path="./exported_models"):
    """Export model to ONNX format"""
    print("กำลัง export โมเดลเป็น ONNX...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    
    # Prepare dummy input
    dummy_text = "患者主诉头痛三天"
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Export to ONNX
    output_dir = Path(output_path) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_dir / "model.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"ONNX model saved to {output_dir}")

def export_to_torchscript(model_path="./final_model", output_path="./exported_models"):
    """Export model to TorchScript"""
    print("กำลัง export โมเดลเป็น TorchScript...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    
    # Prepare dummy input
    dummy_text = "患者主诉头痛三天"
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Trace model
    output_dir = Path(output_path) / "torchscript"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    traced_model = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))
    traced_model.save(str(output_dir / "model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"TorchScript model saved to {output_dir}")

def create_deployment_package(model_path="./final_model", output_path="./deployment"):
    """สร้าง deployment package พร้อมใช้งาน"""
    print("กำลังสร้าง deployment package...")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    model_dir = output_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(model_path, model_dir)
    
    # Create API script
    api_script = '''import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route("/translate", methods=["POST"])
def translate():
    """API endpoint for translation"""
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=5)
        
        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "source": text,
            "translation": translation
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
'''
    
    with open(output_dir / "api.py", "w") as f:
        f.write(api_script)
    
    # Create requirements
    requirements = '''torch==2.0.1
transformers==4.35.2
flask==2.3.3
sentencepiece==0.1.99
'''
    
    with open(output_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    
    # Create Dockerfile
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api.py"]
'''
    
    with open(output_dir / "Dockerfile", "w") as f:
        f.write(dockerfile)
    
    # Create README
    readme = '''# Chinese-Thai Medical Translation API

## การใช้งาน

### 1. รัน API Server
```bash
python api.py
```

### 2. เรียกใช้ API
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d \'{"text": "患者主诉头痛三天"}\'
```

### 3. Docker
```bash
docker build -t chinese-thai-translator .
docker run -p 5000:5000 chinese-thai-translator
```
'''
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    
    # Create zip file
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir.parent)
                zipf.write(file_path, arcname)
    
    print(f"Deployment package created: {zip_path}")
    
    # Create model card
    model_card = {
        "model_name": "Chinese-Thai Medical Translator",
        "task": "translation",
        "source_language": "Chinese",
        "target_language": "Thai",
        "domain": "Medical",
        "base_model": "Helsinki-NLP/opus-mt-zh-en",
        "training_data": "Chinese-Thai medical parallel corpus",
        "metrics": {
            "bleu_score": "Check test_results.json"
        },
        "usage": "Use api.py or load with transformers"
    }
    
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./final_model", help="Path to trained model")
    parser.add_argument("--format", choices=["all", "deployment", "onnx", "torchscript"], 
                       default="deployment", help="Export format")
    args = parser.parse_args()
    
    if args.format in ["all", "deployment"]:
        create_deployment_package(args.model_path)
    
    if args.format in ["all", "onnx"]:
        export_to_onnx(args.model_path)
    
    if args.format in ["all", "torchscript"]:
        export_to_torchscript(args.model_path)
    
    print("\nExport เสร็จสมบูรณ์!")

if __name__ == "__main__":
    main()
