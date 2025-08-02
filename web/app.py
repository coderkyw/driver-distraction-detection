from flask import Flask, render_template, request, url_for, jsonify
import os
import uuid
from PIL import Image
from inference.predict import predict_image  # 复用单张推理逻辑

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None
    selected_model = None

    if request.method == 'POST':
        if 'image' not in request.files or 'model' not in request.form:
            return "缺少文件或模型选择", 400

        file = request.files['image']
        model_choice = request.form.get('model')
        selected_model = model_choice

        if file.filename == '':
            return "未选择文件", 400

        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        if model_choice not in ['resnet', 'mobilenet']:
            return "无效的模型选择", 400

        result = predict_image(file_path, model_choice)
        image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', result=result, image_url=image_url, selected_model=selected_model)

@app.route('/predict_video_frame', methods=['POST'])
def predict_video_frame():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': '缺少文件或模型选择'}), 400

    file = request.files['image']
    model_choice = request.form.get('model')

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': '无法打开图片'}), 400

    if model_choice not in ['resnet', 'mobilenet']:
        return jsonify({'error': '无效的模型选择'}), 400

    # 复用 predict_image 函数，但需改造支持PIL对象或者单独写个 predict_image_from_pil
    # 这里简化，先保存临时文件再调用
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    img.save(temp_path)

    try:
        result = predict_image(temp_path, model_choice)
    except Exception as e:
        result = None
        print(f"推理异常: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if result:
        return jsonify({'result': result})
    else:
        return jsonify({'error': '推理失败'}), 500

if __name__ == "__main__":
    app.run(debug=True)
