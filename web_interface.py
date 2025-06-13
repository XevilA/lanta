#!/usr/bin/env python3
"""
Web Interface for Chinese-Thai Medical Translation
เว็บอินเตอร์เฟซสำหรับการแปลทางการแพทย์จีน-ไทย

A simple web interface for testing translation models.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import threading
from datetime import datetime

# Web framework
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

# Import project modules
from config import config
from main import ChineseThaiTranslator, MedicalTranslationDataset
from model_comparison import ModelComparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationWebApp:
    """Web application for translation testing"""
    
    def __init__(self):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web interface")
        
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        CORS(self.app)
        
        # Initialize models
        self.models = {}
        self.model_configs = {
            'opus_pivot': {
                'name': 'OPUS Pivot (Chinese→English→Thai)',
                'model_name': 'Helsinki-NLP/opus-mt-zh-en',
                'description': 'Uses pivot translation through English'
            },
            'nllb_direct': {
                'name': 'NLLB Direct (Chinese→Thai)',
                'model_name': 'facebook/nllb-200-distilled-600M',
                'description': 'Direct translation using multilingual model'
            }
        }
        
        # Translation history
        self.translation_history = []
        self.max_history = 100
        
        # Setup routes
        self._setup_routes()
        
        # Load models in background
        self._load_models_background()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return self._render_template('index.html')
        
        @self.app.route('/api/models')
        def get_models():
            """Get available models"""
            model_info = {}
            for model_id, config in self.model_configs.items():
                model_info[model_id] = {
                    'name': config['name'],
                    'description': config['description'],
                    'loaded': model_id in self.models,
                    'loading': hasattr(self, f'_loading_{model_id}')
                }
            return jsonify(model_info)
        
        @self.app.route('/api/translate', methods=['POST'])
        def translate():
            """Translate text"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()
                model_id = data.get('model', 'opus_pivot')
                context = data.get('context', '').strip()
                
                if not text:
                    return jsonify({'error': 'No text provided'}), 400
                
                if model_id not in self.models:
                    return jsonify({'error': f'Model {model_id} not available'}), 400
                
                # Perform translation
                start_time = time.time()
                translator = self.models[model_id]
                
                if context:
                    translations = translator.translate_batch([text], [context])
                else:
                    translations = translator.translate_batch([text])
                
                translation_time = time.time() - start_time
                
                # Store in history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'source': text,
                    'translation': translations[0],
                    'context': context,
                    'model': model_id,
                    'model_name': self.model_configs[model_id]['name'],
                    'translation_time': translation_time
                }
                
                self.translation_history.insert(0, history_entry)
                if len(self.translation_history) > self.max_history:
                    self.translation_history = self.translation_history[:self.max_history]
                
                return jsonify({
                    'translation': translations[0],
                    'model': self.model_configs[model_id]['name'],
                    'translation_time': translation_time,
                    'source_length': len(text),
                    'translation_length': len(translations[0])
                })
                
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history')
        def get_history():
            """Get translation history"""
            return jsonify(self.translation_history)
        
        @self.app.route('/api/examples')
        def get_examples():
            """Get example sentences"""
            examples = [
                {
                    'chinese': '患者感到头痛和发烧。',
                    'context': '医生询问病人的症状',
                    'category': 'symptoms'
                },
                {
                    'chinese': '请问您有什么症状？',
                    'context': '医生问诊',
                    'category': 'inquiry'
                },
                {
                    'chinese': '医生建议进行血液检查。',
                    'context': '医生给出建议',
                    'category': 'recommendation'
                },
                {
                    'chinese': '这个药需要每天服用三次。',
                    'context': '医生解释用药方法',
                    'category': 'medication'
                },
                {
                    'chinese': '手术后需要休息一周。',
                    'context': '医生说明术后护理',
                    'category': 'recovery'
                },
                {
                    'chinese': '血压有点高，需要注意饮食。',
                    'context': '医生分析检查结果',
                    'category': 'diagnosis'
                }
            ]
            return jsonify(examples)
        
        @self.app.route('/api/compare', methods=['POST'])
        def compare_models():
            """Compare models on given text"""
            try:
                data = request.get_json()
                text = data.get('text', '').strip()
                context = data.get('context', '').strip()
                
                if not text:
                    return jsonify({'error': 'No text provided'}), 400
                
                results = {}
                for model_id, translator in self.models.items():
                    try:
                        start_time = time.time()
                        if context:
                            translations = translator.translate_batch([text], [context])
                        else:
                            translations = translator.translate_batch([text])
                        
                        translation_time = time.time() - start_time
                        
                        results[model_id] = {
                            'translation': translations[0],
                            'model_name': self.model_configs[model_id]['name'],
                            'translation_time': translation_time
                        }
                    except Exception as e:
                        results[model_id] = {
                            'error': str(e),
                            'model_name': self.model_configs[model_id]['name']
                        }
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Comparison error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get application statistics"""
            stats = {
                'total_translations': len(self.translation_history),
                'models_loaded': len(self.models),
                'models_available': len(self.model_configs),
                'uptime': time.time() - getattr(self, 'start_time', time.time()),
                'last_translation': self.translation_history[0]['timestamp'] if self.translation_history else None
            }
            return jsonify(stats)
    
    def _load_models_background(self):
        """Load models in background threads"""
        def load_model(model_id, config):
            setattr(self, f'_loading_{model_id}', True)
            try:
                logger.info(f"Loading model: {config['name']}")
                translator = ChineseThaiTranslator(config['model_name'])
                self.models[model_id] = translator
                logger.info(f"Successfully loaded: {config['name']}")
            except Exception as e:
                logger.error(f"Failed to load {config['name']}: {e}")
            finally:
                delattr(self, f'_loading_{model_id}')
        
        # Load models in separate threads
        for model_id, config in self.model_configs.items():
            thread = threading.Thread(target=load_model, args=(model_id, config))
            thread.daemon = True
            thread.start()
    
    def _render_template(self, template_name, **kwargs):
        """Render template with error handling"""
        try:
            return render_template(template_name, **kwargs)
        except:
            # Return simple HTML if template not found
            return self._get_simple_html()
    
    def _get_simple_html(self):
        """Get simple HTML interface"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Chinese-Thai Medical Translation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .input-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        textarea { height: 100px; resize: vertical; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result { background: white; padding: 15px; border-radius: 4px; border-left: 4px solid #007bff; }
        .error { border-left-color: #dc3545; color: #dc3545; }
        .loading { color: #ffc107; }
        .examples { display: flex; flex-wrap: wrap; gap: 10px; }
        .example { background: #e9ecef; padding: 8px 12px; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .example:hover { background: #dee2e6; }
        .stats { font-size: 14px; color: #666; }
        .history { max-height: 300px; overflow-y: auto; }
        .history-item { background: white; padding: 10px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #eee; }
        .tab-buttons { margin-bottom: 20px; }
        .tab-button { background: #e9ecef; border: 1px solid #ccc; padding: 10px 20px; cursor: pointer; }
        .tab-button.active { background: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <h1>🏥 Chinese-Thai Medical Translation</h1>
    <p>การแปลบทสนทนาทางการแพทย์จากภาษาจีนเป็นภาษาไทย</p>
    
    <div class="tab-buttons">
        <button class="tab-button active" onclick="showTab('translate')">Translate</button>
        <button class="tab-button" onclick="showTab('compare')">Compare Models</button>
        <button class="tab-button" onclick="showTab('history')">History</button>
    </div>
    
    <div id="translate-tab" class="tab-content active">
        <div class="container">
            <div class="input-group">
                <label for="model-select">Model:</label>
                <select id="model-select">
                    <option value="">Loading models...</option>
                </select>
            </div>
            
            <div class="input-group">
                <label for="source-text">Chinese Text (中文):</label>
                <textarea id="source-text" placeholder="Enter Chinese medical text here..."></textarea>
            </div>
            
            <div class="input-group">
                <label for="context-text">Context (optional):</label>
                <textarea id="context-text" placeholder="Enter dialogue context..."></textarea>
            </div>
            
            <button onclick="translateText()" id="translate-btn">Translate</button>
            <button onclick="clearText()">Clear</button>
            
            <div id="result-container" style="margin-top: 20px;"></div>
        </div>
        
        <div class="container">
            <h3>Example Sentences:</h3>
            <div id="examples-container" class="examples"></div>
        </div>
    </div>
    
    <div id="compare-tab" class="tab-content">
        <div class="container">
            <div class="input-group">
                <label for="compare-text">Chinese Text (中文):</label>
                <textarea id="compare-text" placeholder="Enter Chinese medical text to compare..."></textarea>
            </div>
            
            <div class="input-group">
                <label for="compare-context">Context (optional):</label>
                <textarea id="compare-context" placeholder="Enter dialogue context..."></textarea>
            </div>
            
            <button onclick="compareModels()" id="compare-btn">Compare All Models</button>
            
            <div id="comparison-results" style="margin-top: 20px;"></div>
        </div>
    </div>
    
    <div id="history-tab" class="tab-content">
        <div class="container">
            <h3>Translation History</h3>
            <div id="history-container" class="history"></div>
        </div>
    </div>
    
    <div class="container">
        <div id="stats-container" class="stats"></div>
    </div>

    <script>
        // Global variables
        let models = {};
        let examples = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadModels();
            loadExamples();
            loadStats();
            loadHistory();
            
            // Auto-refresh every 5 seconds
            setInterval(loadStats, 5000);
        });
        
        // Tab management
        function showTab(tabName) {
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        // Load available models
        function loadModels() {
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    models = data;
                    const select = document.getElementById('model-select');
                    select.innerHTML = '';
                    
                    for (const [id, model] of Object.entries(data)) {
                        const option = document.createElement('option');
                        option.value = id;
                        option.textContent = model.name + (model.loaded ? '' : ' (Loading...)');
                        option.disabled = !model.loaded;
                        select.appendChild(option);
                    }
                    
                    // Select first available model
                    const availableModels = Object.entries(data).filter(([id, model]) => model.loaded);
                    if (availableModels.length > 0) {
                        select.value = availableModels[0][0];
                    }
                })
                .catch(error => console.error('Error loading models:', error));
        }
        
        // Load example sentences
        function loadExamples() {
            fetch('/api/examples')
                .then(response => response.json())
                .then(data => {
                    examples = data;
                    const container = document.getElementById('examples-container');
                    container.innerHTML = '';
                    
                    data.forEach(example => {
                        const div = document.createElement('div');
                        div.className = 'example';
                        div.textContent = example.chinese;
                        div.title = example.context;
                        div.onclick = () => useExample(example);
                        container.appendChild(div);
                    });
                })
                .catch(error => console.error('Error loading examples:', error));
        }
        
        // Use example sentence
        function useExample(example) {
            document.getElementById('source-text').value = example.chinese;
            document.getElementById('context-text').value = example.context;
        }
        
        // Translate text
        function translateText() {
            const text = document.getElementById('source-text').value.trim();
            const context = document.getElementById('context-text').value.trim();
            const model = document.getElementById('model-select').value;
            
            if (!text) {
                showResult('Please enter some Chinese text to translate.', 'error');
                return;
            }
            
            if (!model) {
                showResult('Please select a model.', 'error');
                return;
            }
            
            const btn = document.getElementById('translate-btn');
            btn.disabled = true;
            btn.textContent = 'Translating...';
            
            showResult('Translating...', 'loading');
            
            fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, context, model })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showResult(data.error, 'error');
                } else {
                    const resultHtml = `
                        <div class="result">
                            <h4>Translation Result:</h4>
                            <p><strong>Thai:</strong> ${data.translation}</p>
                            <p><strong>Model:</strong> ${data.model}</p>
                            <p><strong>Time:</strong> ${(data.translation_time * 1000).toFixed(1)}ms</p>
                            <p><strong>Length:</strong> ${data.source_length} → ${data.translation_length} characters</p>
                        </div>
                    `;
                    document.getElementById('result-container').innerHTML = resultHtml;
                }
            })
            .catch(error => {
                showResult('Error: ' + error.message, 'error');
            })
            .finally(() => {
                btn.disabled = false;
                btn.textContent = 'Translate';
                loadHistory(); // Refresh history
            });
        }
        
        // Compare models
        function compareModels() {
            const text = document.getElementById('compare-text').value.trim();
            const context = document.getElementById('compare-context').value.trim();
            
            if (!text) {
                document.getElementById('comparison-results').innerHTML = '<div class="result error">Please enter some Chinese text to compare.</div>';
                return;
            }
            
            const btn = document.getElementById('compare-btn');
            btn.disabled = true;
            btn.textContent = 'Comparing...';
            
            document.getElementById('comparison-results').innerHTML = '<div class="result loading">Comparing models...</div>';
            
            fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, context })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('comparison-results').innerHTML = `<div class="result error">${data.error}</div>`;
                } else {
                    let resultsHtml = '<h4>Model Comparison Results:</h4>';
                    for (const [modelId, result] of Object.entries(data)) {
                        if (result.error) {
                            resultsHtml += `
                                <div class="result error">
                                    <strong>${result.model_name}:</strong> Error - ${result.error}
                                </div>
                            `;
                        } else {
                            resultsHtml += `
                                <div class="result">
                                    <strong>${result.model_name}:</strong><br>
                                    <strong>Translation:</strong> ${result.translation}<br>
                                    <strong>Time:</strong> ${(result.translation_time * 1000).toFixed(1)}ms
                                </div>
                            `;
                        }
                    }
                    document.getElementById('comparison-results').innerHTML = resultsHtml;
                }
            })
            .catch(error => {
                document.getElementById('comparison-results').innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            })
            .finally(() => {
                btn.disabled = false;
                btn.textContent = 'Compare All Models';
            });
        }
        
        // Load translation history
        function loadHistory() {
            fetch('/api/history')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('history-container');
                    container.innerHTML = '';
                    
                    if (data.length === 0) {
                        container.innerHTML = '<p>No translation history yet.</p>';
                        return;
                    }
                    
                    data.slice(0, 10).forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'history-item';
                        div.innerHTML = `
                            <strong>Source:</strong> ${item.source}<br>
                            <strong>Translation:</strong> ${item.translation}<br>
                            <strong>Model:</strong> ${item.model_name}<br>
                            <strong>Time:</strong> ${new Date(item.timestamp).toLocaleString()}<br>
                            <strong>Duration:</strong> ${(item.translation_time * 1000).toFixed(1)}ms
                        `;
                        container.appendChild(div);
                    });
                })
                .catch(error => console.error('Error loading history:', error));
        }
        
        // Load statistics
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const uptimeHours = (data.uptime / 3600).toFixed(1);
                    const statsHtml = `
                        📊 <strong>Statistics:</strong> 
                        ${data.total_translations} translations | 
                        ${data.models_loaded}/${data.models_available} models loaded | 
                        Uptime: ${uptimeHours}h
                        ${data.last_translation ? ` | Last: ${new Date(data.last_translation).toLocaleTimeString()}` : ''}
                    `;
                    document.getElementById('stats-container').innerHTML = statsHtml;
                })
                .catch(error => console.error('Error loading stats:', error));
        }
        
        // Show result
        function showResult(message, type = 'result') {
            const container = document.getElementById('result-container');
            container.innerHTML = `<div class="result ${type}">${message}</div>`;
        }
        
        // Clear text
        function clearText() {
            document.getElementById('source-text').value = '';
            document.getElementById('context-text').value = '';
            document.getElementById('result-container').innerHTML = '';
        }
        
        // Refresh models periodically
        setInterval(loadModels, 10000);
    </script>
</body>
</html>
        '''
    
    def run(self, host='0.0.0.0', port=8000, debug=False):
        """Run the web application"""
        self.start_time = time.time()
        logger.info(f"Starting web interface at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chinese-Thai Medical Translation Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if not FLASK_AVAILABLE:
        print("Error: Flask is not installed. Install with:")
        print("pip install flask flask-cors")
        return
    
    try:
        app = TranslationWebApp()
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down web interface...")
    except Exception as e:
        print(f"Error starting web interface: {e}")

if __name__ == "__main__":
    main()
