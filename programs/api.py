import eventlet
import eventlet.wsgi
eventlet.monkey_patch()  # monkey patching for socketio
import os
import time
from flask import Flask, request, jsonify, url_for, send_file, Response
from flask_socketio import SocketIO, emit
from spot_the_diff_maker import Spot_the_diff_maker
import threading
import socket

RECEIVED_IMG_DIR = "../images/main_images/received_images/"
SPOT_THE_DIFF_IMG_DIR = "spot_the_diff_images/"


class Api:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config["UPLOAD_FOLDER"] = (
            RECEIVED_IMG_DIR  # 受け取った画像を置いておく場所
        )
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.app.add_url_rule("/upload", "upload", self.upload, methods=["POST"])
        self.app.add_url_rule(
            "/images/main_images/spot_the_diff_images/<path:filename>",
            "serve_processed_image",
            self.serve_processed_image,
        )
        self.app.add_url_rule("/", "accept", self.accept, methods=["POST"])

        # サーバーのIPアドレスを取得
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        print(f"Server IP address: {ip}")

        self.spot_the_diff_maker = Spot_the_diff_maker()

    # @self.app.route('/upload', methods=['POST'])
    def upload(self):
        """
        - POST リクエストを受信すると実行される
        - リクエストから画像と難易度を受信
        - 間違い探し画像を生成
        - 間違い探し画像のURLと間違い箇所を返す
        """
        # リクエストから画像と難易度を受け取る
        received_img_path, difficulty = self.receive_img_difficulty()
        if (
            received_img_path is None
        ):  # エラー時は `jsonify(...)` を返しているので、それを検出
            return jsonify({"error": "画像がありません"}), 400

        print(
            f"Calling make_spot_the_diff with: received_img_path={received_img_path}, difficulty={difficulty}, emit_progress={self.emit_progress}"
        )
        try:
            spot_the_diff_img_path, all_mask_path, obj_pos_list_xywh = (
                self.spot_the_diff_maker.make_spot_the_diff(
                    received_img_path,
                    difficulty=difficulty,
                    emit_progress_func=self.emit_progress,
                )
            )
        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            return jsonify({"error": "Image processing failed"}), 500

        if not os.path.exists(spot_the_diff_img_path):
            print(f"Error: {spot_the_diff_img_path} not found!")
            return jsonify({"error": "Processed image not found"}), 500
        if not os.path.exists(all_mask_path):
            print(f"Error: {all_mask_path} not found!")
            return jsonify({"error": "Mask image not found"}), 500

        # 加工済み画像のURLを生成
        spot_the_diff_img_url = self.create_img_url(spot_the_diff_img_path)
        all_mask_url = self.create_img_url(all_mask_path)
        print(spot_the_diff_img_url)
        print(all_mask_url)
        print(obj_pos_list_xywh)

        result = {
            "processed_image_url": spot_the_diff_img_url,
            "mask_image_url": all_mask_url,
            "changes": obj_pos_list_xywh,
        }

        return jsonify(result)

    def receive_img_difficulty(self):
        """
        - リクエストから画像と難易度を受信

        Return:
            - received_img_path: str
            - difficulty: int
        """
        image_file = request.files.get("image")
        difficulty = int(request.form.get("difficulty", 1)) # 0: 簡単（初期値）, 1: 普通, 2: 難しい

        print("Received files:", request.files.keys())  # 受信したファイルのキーを表示

        if not image_file:
            return None, None

        # 受け取った画像を保存
        now_time = str(time.time())
        received_img_path = os.path.join(
            self.app.config["UPLOAD_FOLDER"], f"received_image_{now_time}.jpg"
        )
        image_file.save(received_img_path)

        return received_img_path, difficulty

    def emit_progress(self, progress):
        """
        - 進捗を送信する

        Args:
            - progress: float (0.0 ~ 1.0)
        """
        eventlet.spawn(lambda: self.socketio.emit("progress", progress, namespace="/"))

    def create_img_url(self, spot_the_diff_img_path):
        """
        画像のURLを作成する

        Args:
            - image_filename: str
        Return:
            - image_url: str
        """
        image_filename = os.path.basename(spot_the_diff_img_path)
        image_url = url_for(
            "serve_processed_image",
            filename=image_filename,
            _external=True,
        )

        return image_url

    def serve_processed_image(self, filename):
        directory = os.path.abspath(
            os.path.join(os.getcwd(), "../images/main_images/spot_the_diff_images/")
        )
        file_path = os.path.join(directory, filename)

        if not os.path.exists(file_path):
            print(f"Error: Requested image not found at {file_path}")
            return jsonify({"error": "Image not found"}), 404

        print(f"Serving image: {file_path}")
        with open(file_path, 'rb') as f:
            data = f.read()
        headers = {
            'Content-Length': str(len(data)),
            'Cache-Control': 'no-cache',
        }
        print(f"Content-Length: {len(data)}")
        print(f"Headers: {headers}")

        return Response(data, mimetype='image/jpeg', headers=headers)

    def accept(self):
        # 準備開始
        print("start preparing")
        eventlet.spawn(self.spot_the_diff_maker.prepare())
        return jsonify({"message": "start preparing"})

if __name__ == "__main__":
    """
    デバッグモード：
        - コード変更時に自動リロードされる
        - 詳細なエラーログが表示される
        - use_reloader=Falseを付けないと2重起動する
    """
    debug_mode = False
    api = Api()

    if debug_mode:
        api.socketio.run(
            api.app, host="0.0.0.0", port=5000, debug=True, use_reloader=False
        )

    else:
        api.socketio.run(api.app, host="0.0.0.0", port=5000)
