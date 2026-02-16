from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# 수정된 main_logic.py에서 단순화된 분석 함수를 불러옵니다.
from main_logic import run_simplified_analysis

app = Flask(__name__)
CORS(app)

# --- 웹페이지 라우트 ---
@app.route('/')
def home():
    return render_template('index.html')

# --- 분석 API 라우트 ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. 파일 수신 및 기본 검사
    if 'file' not in request.files:
        return jsonify({"error": "파일이 전송되지 않았습니다."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    try:
        # 2. 단순화된 분석 파이프라인 함수를 호출합니다.
        analysis_results = run_simplified_analysis(file)
        
        # 3. 분석 결과를 웹사이트(프론트엔드)에 전달합니다.
        return jsonify(analysis_results)

    except Exception as e:
        # 분석 중 발생한 모든 오류를 처리합니다.
        print(f"분석 파이프라인 실행 중 오류 발생: {e}")
        return jsonify({"error": f"서버 분석 중 오류 발생: {str(e)}"}), 500

# ==================================================================
# 서버 실행
# ==================================================================
if __name__ == '__main__':
    # static 폴더는 main_logic.py에서 시각화 파일 저장 시 자동으로 생성됩니다.
    app.run(host='0.0.0.0', port=5000, debug=True)