"""
使わない

"""

from flask import Flask, request, jsonify, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../images/test_images/' # 受け取った画像を置いておく場所
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/upload', methods=['POST'])
def upload():
    # リクエストから画像と難易度を受け取る
    image_file = request.files.get('image')
    difficulty = int(request.form.get('difficulty', 1))

    if not image_file:
        return jsonify({'error': '画像がありません'}), 400

    # 受け取った画像を保存
    # ファイル名決定の処理はいい感じにおねがいします
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
    image_file.save(input_path)

    # 画像処理スレッド開始
    thread = threading.Thread(target=process_image, args=(input_path, difficulty))
    thread.start()

    # 加工済み画像のURLを生成して返す
    processed_image_filename = 'いい感じの名前.png' # 加工した画像の名前
    processed_image_url = url_for('../images/output_images/', filename=processed_image_filename, _external=True) # 加工した画像の場所

    result = {
        'processed_image_url': processed_image_url,
        'changes': [] # あとでいいかんじにする
    }
    return jsonify(result)

def process_image(input_path, difficulty):
    # 画像処理をする
    # 後でいい感じにする
    for i in range(0.0, 1.1, 0.01):
        progress = i

        # 進捗をクライアントへ
        socketio.emit('progress', progress)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) # IPアドレス，ポートをあとでいい感じにする