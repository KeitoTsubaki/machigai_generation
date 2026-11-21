import random
import os
import time
import pathlib
import gc

import cv2
from PIL import Image
import torch
import numpy as np
import json

from segmentation import Segmentation
from captioning import Captioning_CoCa
from change_object import Changing_object
from object_info import Object_info


SEGMENT_BOX_DIR = "../images/main_images/box_images"
SEGMENT_MASK_DIR = "../images/main_images/mask_images"
SEGMENT_MASKED_IMG_DIR = "../images/main_images/masked_images"
SPOT_THE_DIFF_IMAGES_DIR = "../images/main_images/spot_the_diff_images"


class Spot_the_diff_maker:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        start_loading_time = time.time()
        self.segmentation = Segmentation(device)
        self.captioning = Captioning_CoCa(device)
        self.changing_object = Changing_object(device)
        end_loading_time = time.time()
        print("Total Loading Time: ", end_loading_time - start_loading_time, "[s]")

        self.easy_mode_params = {
            "change_object_num": 2,
            "max_size_ratio": 0.2,
            "min_size_ratio": 0.04,
        }
        self.normal_mode_params = {
            "change_object_num": 4,
            "max_size_ratio": 0.04,
            "min_size_ratio": 0.009,
        }
        self.hard_mode_params = {
            "change_object_num": 5,
            "max_size_ratio": 0.04,
            "min_size_ratio": 0.005,
        }

        # 画像を保存するディレクトリを作成
        if not os.path.exists(SEGMENT_BOX_DIR):
            os.makedirs(SEGMENT_BOX_DIR)
            print(f"box dir created")
        if not os.path.exists(SEGMENT_MASK_DIR):
            os.makedirs(SEGMENT_MASK_DIR)
            print(f"mask dir created")
        if not os.path.exists(SEGMENT_MASKED_IMG_DIR):
            os.makedirs(SEGMENT_MASKED_IMG_DIR)
            print(f"masked image dir created")
        if not os.path.exists(SPOT_THE_DIFF_IMAGES_DIR):
            os.makedirs(SPOT_THE_DIFF_IMAGES_DIR)
            print(f"spot the diff image dir created")

        with open("prepared_prompt.json") as f:
            self.prepared_prompts = json.load(f)
            print("prepared prompts loaded")

        self.img_h = 0
        self.img_w = 0

        self.spot_the_diff_img_path = ""
        self.all_mask_path = ""
        self.obj_pos_list_xywh = []

        self.emit_progress_func = None

    def load_seg_cap(self):
        start_load_seg_cap = time.time()
        self.segmentation = Segmentation()
        self.captioning = Captioning_CoCa()
        end_load_seg_cap = time.time()
        print(
            "Segmentation and Captioning model's Loading Time: ",
            end_load_seg_cap - start_load_seg_cap,
            "[s]",
        )

    def make_spot_the_diff(self, img_path, difficulty=1, emit_progress_func=None):
        """
        - 間違い探し画像を生成する
        - Tasks:
            - segmentation
            - select_change_objects
            - generate caption * change_object_num
            - change objects * change_object_num

        """
        print(
            f"make_spot_the_diff called with: img_path={img_path}, difficulty={difficulty}, emit_progress_func={emit_progress_func}"
        )
        start_create_time = time.time()
        self.emit_progress_func = emit_progress_func

        change_object_num = self.get_change_object_num(difficulty)
        print(f"The number of difference: {change_object_num}")
        self.task_num = 2 + 2 * change_object_num
        self.task_count = 0
        print(f"The number of tasks: {self.task_num}")

        img = cv2.imread(img_path)
        self.img_h, self.img_w = img.shape[0], img.shape[1]
        img_size = img.shape[0] * img.shape[1]

        # セグメンテーション
        object_infos = self.segmentation.segment_image(
            img,
            box_save_dir=SEGMENT_BOX_DIR,
            mask_save_dir=SEGMENT_MASK_DIR,
            masked_img_save_dir=SEGMENT_MASKED_IMG_DIR,
        )
        print(f"object infos len: {len(object_infos)}")
        # print(f"object infos: {object_infos}")
        self.done_task()

        # キャッシュを削除
        del self.segmentation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 変更する領域を決定
        change_object_infos = self.select_change_objects(
            img_size, object_infos, difficulty
        )
        self.done_task()

        for object_info in change_object_infos.values():
            object_info.show_object_info()

        # キャプション生成
        self.generate_caption(change_object_infos)

        for object_info in change_object_infos.values():
            object_info.show_object_info()

        # キャッシュを削除
        del self.captioning
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # 画像生成
        self.change_objects(img, change_object_infos, difficulty)

        end_create_time = time.time()

        del img, object_infos, change_object_infos
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("Total Creation Time: ", end_create_time - start_create_time, "[s]")

        json_data = {
            "obj_pos_list_xywh": self.obj_pos_list_xywh,
            "spot_the_diff_img_path": self.spot_the_diff_img_path,
            "all_mask_path": self.all_mask_path,
        }
        with open(
            f"../output_jsons/json_data_{str(time.time())}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            print("json file created")

        return self.spot_the_diff_img_path, self.all_mask_path, self.obj_pos_list_xywh

    def done_task(self):
        if self.emit_progress_func is not None:
            if self.task_count < self.task_num:
                self.task_count += 1
                progress = self.task_count / self.task_num
                # 小数点以下を3桁にする
                progress = round(progress, 3)
                print(f"progress: {(progress * 100):.1f} %")
                self.emit_progress_func(progress)

    def get_change_object_num(self, difficulty):
        if difficulty == 0:
            return self.easy_mode_params["change_object_num"]
        elif difficulty == 1:
            return self.normal_mode_params["change_object_num"]
        elif difficulty == 2:
            return self.hard_mode_params["change_object_num"]
        else:
            print("ERROR: invalid difficulty")
            return 0

    def prepare(self):
        self.obj_pos_list_xywh = []
        self.load_seg_cap()

    def select_change_objects(self, img_size, object_infos, difficulty):
        """
        - box_imagesディレクトリの中から変更する物体をランダムに選択する
        - 元画像サイズの 0.2% 以上 40% 以下のオブジェクトから選択
        """
        object_num = len(object_infos)
        print(f"total object num: {object_num}")
        print(f"image size: {img_size}")

        max_object_size, min_object_size, change_object_num = (
            self.set_parameters_by_difficulty(difficulty, img_size)
        )

        # 大きすぎるものと小さすぎるものを除外する
        cropped_object_infos = self.remove_too_big_small(
            object_infos, max_object_size, min_object_size, img_size, change_object_num
        )

        # ランダムに選択する
        change_object_infos = self.random_select_objects(
            cropped_object_infos, change_object_num
        )
        del cropped_object_infos
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return change_object_infos

    def set_parameters_by_difficulty(self, difficulty, img_size):
        """
        - difficultyを元にパラメータを設定する

        Args:
            - difficulty (0,1,2)
            - img_size (int)
        Returns:
            - max_object_size (int)
            - min_object_size (int)
            - change_object_num (int)
        """
        if difficulty == 0:
            print("easy mode")
            max_object_size = self.easy_mode_params["max_size_ratio"] * img_size
            min_object_size = self.easy_mode_params["min_size_ratio"] * img_size
            change_object_num = self.easy_mode_params["change_object_num"]

        elif difficulty == 1:
            print("normal mode")
            max_object_size = self.normal_mode_params["max_size_ratio"] * img_size
            min_object_size = self.normal_mode_params["min_size_ratio"] * img_size
            change_object_num = self.normal_mode_params["change_object_num"]

        elif difficulty == 2:
            print("hard mode")
            max_object_size = self.hard_mode_params["max_size_ratio"] * img_size
            min_object_size = self.hard_mode_params["min_size_ratio"] * img_size
            change_object_num = self.hard_mode_params["change_object_num"]
        else:
            print("ERROR: invalid difficulty")
            return 0, 0, 0

        return max_object_size, min_object_size, change_object_num

    def remove_too_big_small(
        self,
        object_infos,
        max_object_size,
        min_object_size,
        img_size,
        change_object_num,
    ):
        """
        - 大きすぎる物体、小さすぎる物体を除外

        Args:
            - object_infos (dict)
            - max_object_size (int)
            - min_object_size (int)
            - img_size (int)
        Returns:
            - cropped_object_infos (dict)
        """
        print(
            f"max object size: {max_object_size} ({max_object_size / img_size * 100}%)"
        )
        print(
            f"min object size: {min_object_size} ({min_object_size / img_size * 100}%)"
        )
        cropped_object_infos = {}
        # 一定間隔の領域が100個ある
        # 検出数が100+change object num 以下ならmin object size を下げる
        if len(object_infos) < 100 + change_object_num:
            min_object_size = img_size * 0.009
            print(
                f"min object size: {min_object_size} ({min_object_size / img_size * 100}%)"
            )

        for key, object_info in object_infos.items():
            if (
                object_info.box_size < max_object_size
                and object_info.box_size > min_object_size
            ) or key.split("_")[0] == "fix":
                cropped_object_infos[key] = object_info

        print(f"cropped object num: {len(cropped_object_infos)}")
        # print(f"cropped object info: {cropped_object_infos}")

        return cropped_object_infos

    def random_select_objects(self, cropped_object_infos, change_object_num):
        """
        - ランダムに選択する
        - 難易度によって選択する個数が変動
        - 領域が被らないように選択する

        Args:
            - cropped_object_infos (dict)
            - change_object_num (int)
        Returns:
            - change_box_name_list (list)
        """
        change_object_infos = {}
        box_xyxy_list = []
        random_key_list = []
        max_loop_count = 1

        object_key_list = []
        fixed_object_key_list = []
        for object_key in list(cropped_object_infos.keys()):
            if object_key.split("_")[0] == "fix":
                fixed_object_key_list.append(object_key)
            else:
                object_key_list.append(object_key)
        print(f"object_key_list length: {len(object_key_list)}")
        print(f"fixed_object_key_list length: {len(fixed_object_key_list)}")

        for i in range(change_object_num):
            if len(object_key_list) > 0:
                random_key = random.choice(object_key_list)
            else:
                random_key = random.choice(fixed_object_key_list)

            if len(box_xyxy_list) > 0:
                loop_count = 0
                while True:
                    print(f"random_key: {random_key}")
                    ious = self.calc_iou(
                        cropped_object_infos[random_key].box_xyxy, box_xyxy_list
                    )
                    max_iou = np.max(ious)
                    print(f"max IoU: {max_iou}")
                    if max_iou == 0.0 and random_key not in random_key_list:
                        break

                    print("selecting again because max IoU > 0")

                    loop_count += 1
                    if loop_count > max_loop_count or len(object_key_list) == 0:
                        print("pick up from fixed object list")
                        random_key = random.choice(fixed_object_key_list)
                    else:
                        print("pick up from object list")
                        random_key = random.choice(object_key_list)

            box_xyxy_list.append(cropped_object_infos[random_key].box_xyxy)
            random_key_list.append(random_key)
            change_object_infos[random_key] = cropped_object_infos[random_key]
            box_left_upper_x = cropped_object_infos[random_key].box_xyxy[0]
            box_left_upper_y = cropped_object_infos[random_key].box_xyxy[1]
            box_width = cropped_object_infos[random_key].box_xyxy[2] - box_left_upper_x
            box_height = cropped_object_infos[random_key].box_xyxy[3] - box_left_upper_y
            self.obj_pos_list_xywh.append(
                [box_left_upper_x, box_left_upper_y, box_width, box_height]
            )

        print(f"randomly selected object infos: {change_object_infos}")

        del box_xyxy_list, random_key_list
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return change_object_infos

    def calc_iou(self, box1_xyxy, box2_xyxy):
        """
        IoUを計算する

        Args:
            - box1_xyxy (list)
            - box2_xyxy (list(list))
        Returns:
            - ious (list(float))
        """
        box2_xyxy = np.asarray(box2_xyxy)
        # print(box1_xyxy)
        # print(box2_xyxy)
        box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        box2_area = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (
            box2_xyxy[:, 3] - box2_xyxy[:, 1]
        )

        intersection_xmin = np.maximum(box1_xyxy[0], box2_xyxy[:, 0])
        intersection_ymin = np.maximum(box1_xyxy[1], box2_xyxy[:, 1])
        intersection_xmax = np.minimum(box1_xyxy[2], box2_xyxy[:, 2])
        intersection_ymax = np.minimum(box1_xyxy[3], box2_xyxy[:, 3])

        intersectioin_area = np.maximum(
            0, intersection_xmax - intersection_xmin
        ) * np.maximum(0, intersection_ymax - intersection_ymin)

        union_area = box1_area + box2_area - intersectioin_area

        ious = intersectioin_area / union_area

        del (
            box1_area,
            box2_area,
            intersection_xmin,
            intersection_ymin,
            intersection_xmax,
            intersection_ymax,
            intersectioin_area,
            union_area,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return ious

    def generate_caption(self, change_object_infos):
        # captions = {}
        for object_info in change_object_infos.values():
            box_name = object_info.box_name
            box_image_path = os.path.join(SEGMENT_BOX_DIR, box_name)
            # image = Image.open(box_image_path).convert("RGB")
            image = cv2.imread(box_image_path)
            caption = self.captioning.caption_image(image)
            object_info.set_caption(caption)
            # captions[box_name] = caption
            self.done_task()

        # print(f"captions: {captions}")
        # return captions
        del image, caption
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def change_objects(self, image, change_object_infos, difficulty):
        """
        - box sizeが大きいものに対しては画像全体を使用して生成
        - box sizeが小さいものに対しては物体付近の領域を切り取って、生成後張り付ける

        """

        # small_size = 0.0625 * image.shape[0] * image.shape[1]
        # 画像の1/4
        small_size = 0.2 * image.shape[0] * image.shape[1]
        print(f"small size: {small_size}")

        spot_the_diff_image = image.copy()
        spot_the_diff_image_RGB = cv2.cvtColor(spot_the_diff_image, cv2.COLOR_BGR2RGB)

        # 全てのマスクを含む画像？配列？の初期化
        all_mask = np.zeros((image.shape[0], image.shape[1]))
        all_mask_visible = np.zeros((image.shape[0], image.shape[1]))

        for i, object_info in enumerate(change_object_infos.values()):
            mask_cv2 = cv2.imread(
                os.path.join(SEGMENT_MASK_DIR, object_info.mask_name),
                cv2.IMREAD_GRAYSCALE,
            )
            box_xmin, box_ymin, box_xmax, box_ymax = object_info.box_xyxy

            all_mask += mask_cv2 // 255 * (i + 1)
            all_mask_visible += mask_cv2

            box_width = box_xmax - box_xmin
            box_height = box_ymax - box_ymin
            if box_width > box_height:
                aspect_ratio = box_width / box_height
            else:
                aspect_ratio = box_height / box_width

            # プロンプト生成
            prompt = self.generate_prompt(aspect_ratio, object_info, difficulty)

            # 画像生成
            now_time_str = str(time.time())
            prompt_name = prompt.replace(" ", "_")
            save_path = pathlib.Path(SPOT_THE_DIFF_IMAGES_DIR)
            box_out_name = (
                object_info.box_name.replace(".jpg", "")
                + "_"
                + prompt_name
                + "_"
                + now_time_str
                + ".jpg"
            )
            mask_out_name = (
                object_info.mask_name.replace(".jpg", "")
                + "_"
                + prompt_name
                + "_"
                + now_time_str
                + ".jpg"
            )
            # difficulty == 2のときはプロンプトの反映を弱くする
            guidance_scale = 10.0
            # if difficulty == 2:
            #     guidance_scale = 3.0

            if object_info.box_size < small_size:
                spot_the_diff_image_RGB = cv2.cvtColor(
                    spot_the_diff_image, cv2.COLOR_BGR2RGB
                )

                # 切り取り領域の計算。少し大きめに切り取る。
                crop_xmin, crop_ymin, crop_xmax, crop_ymax = (
                    box_xmin - 50,
                    box_ymin - 50,
                    box_xmax + 50,
                    box_ymax + 50,
                )
                offset_xmin = 50
                offset_ymin = 50
                offset_xmax = 50
                offset_ymax = 50

                if crop_xmin < 0:
                    crop_xmin = 0
                    offset_xmin = box_xmin - crop_xmin
                if crop_ymin < 0:
                    crop_ymin = 0
                    offset_ymin = box_ymin - crop_ymin
                if crop_xmax > image.shape[1]:
                    crop_xmax = image.shape[1]
                    offset_xmax = crop_xmax - box_xmax
                if crop_ymax > image.shape[0]:
                    crop_ymax = image.shape[0]
                    offset_ymax = crop_ymax - box_ymax

                # 画像を切り取り、PIL Imageに変換
                around_box_img = spot_the_diff_image_RGB[
                    crop_ymin:crop_ymax, crop_xmin:crop_xmax
                ]
                around_box_img_pil = Image.fromarray(around_box_img)

                # マスクは少し小さく切り取ることで、サイズを合わせたときにマスクが少し大きくなる
                cropped_mask_cv2 = mask_cv2[
                    crop_ymin + 20 : crop_ymax - 20, crop_xmin + 20 : crop_xmax - 20
                ]
                cropped_mask_pil = Image.fromarray(cropped_mask_cv2)
                cropped_mask_pil = cropped_mask_pil.convert("L")

                self.changing_object.change_object(
                    around_box_img_pil,
                    cropped_mask_pil,
                    prompt,
                    save_dir_path=save_path,
                    out_name=box_out_name,
                    mask_name=mask_out_name,
                    guidance_scale=guidance_scale,
                )

                # 生成画像を元画像に貼り付ける
                changed_box_img = cv2.imread(
                    os.path.join(SPOT_THE_DIFF_IMAGES_DIR, box_out_name)
                )
                cropped_box_width = crop_xmax - crop_xmin
                cropped_box_height = crop_ymax - crop_ymin
                changed_box_img = cv2.resize(
                    changed_box_img,
                    (int(cropped_box_width), int(cropped_box_height)),
                    interpolation=cv2.INTER_AREA,
                )
                changed_box_img = cv2.resize(
                    changed_box_img,
                    (int(cropped_box_width), int(cropped_box_height)),
                    interpolation=cv2.INTER_AREA,
                )
                spot_the_diff_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = (
                    changed_box_img
                )

            else:
                # マスクは少し小さく切り取ることで、サイズを合わせたときにマスクが少し大きくなる
                cropped_mask_cv2 = mask_cv2[
                    0 + 40 : image.shape[0] - 40, 0 + 40 : image.shape[1] - 40
                ]
                cropped_mask_pil = Image.fromarray(cropped_mask_cv2)
                cropped_mask_pil = cropped_mask_pil.convert("L")

                spot_the_diff_image_RGB = cv2.cvtColor(
                    spot_the_diff_image, cv2.COLOR_BGR2RGB
                )
                spot_the_diff_image_pil = Image.fromarray(spot_the_diff_image_RGB)

                self.changing_object.change_object(
                    spot_the_diff_image_pil,
                    cropped_mask_pil,
                    prompt,
                    save_dir_path=save_path,
                    out_name=box_out_name,
                    mask_name=mask_out_name,
                    guidance_scale=guidance_scale,
                )
                changed_box_img = cv2.imread(
                    os.path.join(SPOT_THE_DIFF_IMAGES_DIR, box_out_name)
                )
                changed_box_img = cv2.resize(
                    changed_box_img,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                spot_the_diff_image = changed_box_img

                del spot_the_diff_image_pil
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            self.done_task()

        self.spot_the_diff_img_path = os.path.join(
            SPOT_THE_DIFF_IMAGES_DIR, "spot_the_diff_image_" + now_time_str + ".jpg"
        )
        self.all_mask_path = os.path.join(
            SPOT_THE_DIFF_IMAGES_DIR, "all_mask_" + now_time_str + ".png"
        )
        self.all_mask_visible_path = os.path.join(
            SPOT_THE_DIFF_IMAGES_DIR, "all_mask_visible_" + now_time_str + ".png"
        )
        cv2.imwrite(
            self.spot_the_diff_img_path,
            spot_the_diff_image,
        )
        print("final spot the diff image saved")
        cv2.imwrite(
            self.all_mask_path,
            all_mask,
        )
        print("all mask (0,1,2,...) saved")
        cv2.imwrite(
            self.all_mask_visible_path,
            all_mask_visible,
        )
        print(f"all mask visible (0 or 255) saved")

        del (
            spot_the_diff_image,
            spot_the_diff_image_RGB,
            all_mask,
            all_mask_visible,
            change_object_infos,
            mask_cv2,
            around_box_img,
            around_box_img_pil,
            cropped_mask_cv2,
            cropped_mask_pil,
            changed_box_img,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def generate_prompt(self, aspect_ratio, object_info, difficulty):
        prompt = ""
        if aspect_ratio > 3:
            prompt = random.choice(self.prepared_prompts["long_object"])
        else:
            if difficulty == 0:
                prompt = random.choice(self.prepared_prompts["adjective"])
                prompt += ", " + random.choice(self.prepared_prompts["noun"])
            elif difficulty == 1:
                if object_info.is_fix:
                    prompt = random.choice(self.prepared_prompts["adjective"])
                    prompt += ", " + random.choice(self.prepared_prompts["noun"])
                    prompt += (
                        ", naturally placed, blend naturally into the surroundings"
                    )
                else:
                    if random.randint(0, 1) == 0:
                        prompt = object_info.caption
                        prompt = (
                            prompt.replace("/", " ")
                            .replace(":", " ")
                            .replace("・", " ")
                            .replace("\\", " ")
                            .replace("png", " ")
                        )
                    else:
                        prompt = random.choice(self.prepared_prompts["adjective"])
                        prompt += ", " + random.choice(self.prepared_prompts["noun"])
            elif difficulty == 2:
                if object_info.is_fix:
                    prompt = random.choice(self.prepared_prompts["adjective"])
                    prompt += ", " + random.choice(self.prepared_prompts["noun"])
                    prompt += (
                        ", naturally placed, blend naturally into the surroundings"
                    )
                else:
                    prompt = object_info.caption
                    prompt = (
                        prompt.replace("/", " ")
                        .replace(":", " ")
                        .replace("・", " ")
                        .replace("\\", " ")
                        .replace("png", " ")
                    )

        prompt = prompt + ", blended in naturally, highly detailed, 4k"

        return prompt


if __name__ == "__main__":
    spot_the_diff_maker = Spot_the_diff_maker()

    img_path = "../images/test_images/park_square.jpg"
    # img_path = "../images/test_images/zoo_square.jpg"
    # img_path = "../images/test_images/amusement_park_square.jpg"
    # img_path = "../images/test_images/room_square.jpg"
    # spot_the_diff_maker.make_spot_the_diff(
    #     img_path,
    # )
    while True:
        print("image path ( programs/ ):")
        img_path = input()
        print("difficulty: (0, 1, 2)")
        difficulty = int(input())
        if img_path == "exit":
            break
        elif os.path.isfile(img_path):
            print("generate")
            spot_the_diff_maker.make_spot_the_diff(img_path, difficulty)
            spot_the_diff_maker.prepare()
        else:
            print("file not found")
            continue
