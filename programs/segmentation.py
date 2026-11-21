import os
import gc

import torch
from ultralytics import SAM, FastSAM
import cv2
import numpy as np

from object_info import Object_info


"""
[Models]

SAM2:
    - Large:    sam2.1_l.pt
        - room.jpg: 215190.6ms
    - Base:     sam2.1_b.pt
        - room.jpg: 16285.7ms
    - Small:    sam2.1_s.pt
        - room: 12671.2ms
    - Tiny:     sam2.1_t.pt
        - room.jpg: 12581.8ms

MobileSAM:
    - mobile_sam.pt
        - room.jpg: 28008.2ms
        - zoo.jpg: 18225.8ms

FastSAM:
    - Small:    FastSAM-s.pt
        - room.jpg: 103.1ms
    - Large:    FastSAM-x.pt
        - room.jpg: 161.9ms
"""


class Segmentation:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # print(f"Using device: {self.device}")
        print("Loading Segmentation model...")
        # self.model = SAM("../models/sam2.1_s.pt")
        self.model = FastSAM("../models/FastSAM-x.pt")
        self.model = self.model.to(self.device)
        
        print("The segmentation model has been loaded.")

        self.save_dir = "room"

        self.object_infos = {}

    def segment_image(self, image, box_save_dir, mask_save_dir, masked_img_save_dir):
        """
        セグメンテーションを行い出力画像を保存

        Args:
            - image
            - box_save_dir: save directory for boxes
            - mask_save_dir: save directory for masks
        Returns:
            - boxes_xyxy: int, xyxy format
            - masks_list: list of masks
        """
        with torch.no_grad():
            print("Segmenting image...")
            out = self.model(image)
            print("The image has been segmented.")

        boxes = out[0].boxes
        masks = out[0].masks
        shape = out[0].orig_shape
        speed = out[0].speed

        output_image = out[0].plot()
        # TODO: 検出数0の時の処理

        boxes_xyxy = self.format_boxes(boxes)
        masks_list = self.format_masks(masks, shape)
        # if boxes_xyxy is not None:
        #     print("there is no object")

        self.save_boxes(image, boxes_xyxy, box_save_dir)
        self.save_masks(image, masks_list, mask_save_dir)
        self.save_masked_images(image, masks_list, masked_img_save_dir)
        
        print("The boxex, masks, and masked images have been saved.")
        # self.save_image(output_image, "room_fastsam_x")

        # self.object_num = len(boxes_xyxy)

        # キャッシュを削除
        del out, boxes, masks, shape, speed, output_image, boxes_xyxy, masks_list
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return self.object_infos

    def format_boxes(self, boxes):
        """
        - xyxyのデータを抽出
        - int型に変換

        Args:
            - boxes: ultralytics.engine.results.Boxes object
        Return:
            - boxes_xyxy: int, xyxy format
        """
        boxes_xyxy = boxes.xyxy
        boxes_xyxy = boxes_xyxy.int()

        return boxes_xyxy

    def format_masks(self, masks, image_shape):
        """
        Args:
            - masks: ultralytics.engine.results.Masks object
            - image_shape: (height, width)
        Return:
            - None
        """
        img_height, img_width = image_shape[0], image_shape[1]

        mask_list = []
        if masks is not None:
            for _mask in masks:
                mask = np.array(_mask.data.cpu() * 255, dtype=np.uint8)[0]

                # mask を画像サイズに合わせる
                mask = cv2.resize(mask, (img_width, img_height))
                mask_list.append(mask)

        return mask_list

    def save_masks(self, image, mask_list, mask_save_dir):
        """
        画像に対してマスクをかける

        Args:
            - original image
            - mask_list: list of masks
            - mask_save_dir: save directory
        """
        if mask_list is not None:
            for i, _mask in enumerate(mask_list):
                mask_path = os.path.join(mask_save_dir, f"mask_{i}.jpg")
                cv2.imwrite(mask_path, _mask)

        # 一定間隔でマスクをかける
        box_width = int(image.shape[1] * 0.1)
        box_height = int(image.shape[0] * 0.1)
        for box_ymin in range(0, image.shape[0] - box_height, box_height):
            for box_xmin in range(0, image.shape[1] - box_width, box_width):
                # box_xmax = box_xmin + box_width
                # box_ymax = box_ymin + box_height

                mask_fix = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.circle(
                    mask_fix,
                    (box_xmin + box_width // 2, box_ymin + box_height // 2),
                    box_width // 2,
                    255,
                    -1,
                )

                mask_path = os.path.join(
                    mask_save_dir, f"mask_fix{box_ymin}_{box_xmin}.jpg"
                )
                cv2.imwrite(mask_path, mask_fix)
        print("The masks have been saved.")

    def save_masked_images(self, image, mask_list, masked_img_save_dir):
        """
        画像に対してマスクをかけて保存

        Args:
            - original image
            - mask_list: list of masks
            - masked_img_save_dir: save directory
        """
        if mask_list is not None:
            for i, _mask in enumerate(mask_list):
                # 画像に対してマスクをかける
                masked_image = cv2.bitwise_and(image, image, mask=_mask)
                masked_img_path = os.path.join(
                    masked_img_save_dir, f"masked_image_{i}.jpg"
                )
                cv2.imwrite(masked_img_path, masked_image)
        print(f"The masked images have been saved.")

    def save_boxes(self, image, boxes_xyxy, box_save_dir):
        """
        検出したボックスを切り取り保存

        Args:
            - original image
            - boxes_xyxy: int, xyxy format
            - box_save_dir: save directory
        """
        if boxes_xyxy is not None:
            for i, _box_xyxy in enumerate(boxes_xyxy):
                xmin, ymin, xmax, ymax = _box_xyxy
                box_size = (xmax - xmin) * (ymax - ymin)
                object_info = Object_info(
                    box_size=box_size.item(),
                    box_name=f"box_{i}.jpg",
                    box_xyxy=_box_xyxy.tolist(),
                    mask_name=f"mask_{i}.jpg",
                )
                self.object_infos[f"{i}"] = object_info

                box = image[ymin:ymax, xmin:xmax]
                box_path = os.path.join(box_save_dir, f"box_{i}.jpg")
                cv2.imwrite(box_path, box)

        # 一定間隔でボックスを切り取る
        box_width = int(image.shape[1] * 0.1)
        box_height = int(image.shape[0] * 0.1)
        for box_ymin in range(0, image.shape[0] - box_height, box_height):
            for box_xmin in range(0, image.shape[1] - box_width, box_width):
                box_xmax = box_xmin + box_width
                box_ymax = box_ymin + box_height
                box_xyxy = [box_xmin, box_ymin, box_xmax, box_ymax]
                box_size = box_width * box_height

                object_info = Object_info(
                    box_size=box_size,
                    box_name=f"box_fix{box_ymin}_{box_xmin}.jpg",
                    box_xyxy=box_xyxy,
                    mask_name=f"mask_fix{box_ymin}_{box_xmin}.jpg",
                    is_fix=True,
                )
                self.object_infos[f"fix_{box_ymin}_{box_xmin}"] = object_info
                box = image[box_ymin:box_ymax, box_xmin:box_xmax]
                box_path = os.path.join(
                    box_save_dir, f"box_fix{box_ymin}_{box_xmin}.jpg"
                )
                cv2.imwrite(box_path, box)
        print("The boxes have been saved.")

    def save_image(self, image, filename):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"../images/output_images/{filename}.jpg", image)


if __name__ == "__main__":
    seg = Segmentation()

    img_path = "../images/test_images/room_square.jpg"
    img = cv2.imread(img_path)
    box_save_dir = "../images/main_images/box_images"
    mask_save_dir = "../images/main_images/mask_images"
    masked_img_save_dir = "../images/main_images/masked_images"
    seg.segment_image(img, box_save_dir, mask_save_dir, masked_img_save_dir)
