import gc
import sys
import time
import pathlib

import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
import torch
from PIL import Image


class Changing_object:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inpainting parameters:
            - guidance_scale:
                - プロンプトをどれくらい強く反映させるかを決める
                - 4~12がいい
                - default = 7.5
                - 大きくするとノイズが増える場合がある
            - num_inference_steps:
                - 推論回数
                - 30~100がいい
                - default = 50
                - 大きくするとより詳細な出力が得られる可能性がある
            - negative_prompt:
                - 望ましくない要素を排除
                - examples:
                    - "blurry, low quality" → ぼやけた画像を防ぐ
                    - "distorted, unrealistic" → 変な形のオブジェクトを防ぐ
            - strength:
                - 元の画像をどれくらい維持するかを決める (0~1)
                - 0.5～0.8 にすると、元画像の情報を活かしつつ高品質な inpainting が可能
                - default = 1.0 (大きく変更)
        """
        self.device = device
        self.set_up_model()

        self.guidance_scale = 12.0
        self.num_inference_steps = 40
        self.negative_prompt = "bad anatomy, deformed, ugly, disfigured, distorted, unrealistic, trademark, logo, copyrighted, famous character, Disney, Pixar, Marvel, anime, manga, brand, signature, watermark, artist name, copyrighted design, celebrity, recognizable character"
        self.strength = 1.0
        
        print("[Inpainting parameters]==========")
        print("guidance_scale: ", self.guidance_scale)
        print("num_inference_steps: ", self.num_inference_steps)
        print("negative_prompt: ", self.negative_prompt)
        print("strength: ", self.strength)
        print("================================")

        self.image_save_path = pathlib.Path("../images/tmp_images")

    def set_up_model(self):
        """
        - モデルのロード
        - モデルの設定
        """
        start_time = time.perf_counter()
        print("loading Inpainting model...")
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            "../models/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",  # 軽量版
        )

        #self.model = StableDiffusionInpaintPipeline.from_pretrained(
        #    "stabilityai/stable-diffusion-2-inpainting"
        #)

        self.model.to(self.device)
        self.model.scheduler = EulerDiscreteScheduler.from_config(
            self.model.scheduler.config
        )
        self.model.enable_attention_slicing()
        self.model.enable_model_cpu_offload()
        print("The Inpainting model has been loaded.")
        end_time = time.perf_counter()
        print("loading time: ", end_time - start_time, "[s]")

    def change_object(
        self,
        image,
        mask,
        prompt,
        save_dir_path,
        out_name="out.jpg",
        mask_name="mask.jpg",
        guidance_scale=12.0,
    ):
        """
        Args:
            - image: PIL Image (RGB)
            - mask: PIL Image (L)
            - prompt: str
            - save_dir_path: pathlib.Path
        """
        image = self.ensure_img_format(image)
        mask = self.ensure_mask_format(mask, image.size)
        prompt_name = prompt.replace(" ", "_")

        mask_save_path = save_dir_path / mask_name

        # マスクにぼかしを入れる
        mask = self.model.mask_processor.blur(mask, blur_factor=3)

        mask.save(mask_save_path)

        with torch.inference_mode():
            print("Inpainting...")
            out = self.model(
                prompt=prompt,
                image=image,
                mask_image=mask,
                strength=self.strength,
                guidance_scale=guidance_scale,
                num_inference_steps=self.num_inference_steps,
                negative_prompt=self.negative_prompt,
            )
            print("The image has been inpainted.")

        out_image = out.images[0]

        out_image_save_path = save_dir_path / out_name
        out_image.save(out_image_save_path)
        print("The spot-the-diff image has been saved.")

        # キャッシュを削除
        del out, out_image, image, mask
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def ensure_img_format(self, image):
        """
        imageをRGBに変換

        Args:
            - image: PIL Image
        Return:
            - image: PIL Image (RGB)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def ensure_mask_format(self, mask, img_size):
        """
        - mask をグレースケールに変換
        - imageのサイズと合わせる

        Args:
            - mask: PIL Image
            - img_size: (width, height)
        Return:
            - mask: PIL Image (L)
        """
        if mask.mode != "L":
            mask = mask.convert("L")

        image_width, image_height = img_size
        mask = mask.resize((image_width, image_height))

        return mask


if __name__ == "__main__":
    changing_object = Changing_object()

    image = cv2.imread("../images/test_images/room.jpg")
    mask = cv2.imread(
        "../images/output_images/room/masks/mask_5.jpg", cv2.IMREAD_GRAYSCALE
    )

    crop_xmin, crop_ymin, crop_xmax, crop_ymax = 0, 0, image.shape[1], image.shape[0]

    # マスクが小さいと質が低くなる

    crop_xyxy = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
    mask_crop_xyxy = [crop_xmin + 20, crop_ymin + 20, crop_xmax - 20, crop_ymax - 20]

    image_croped = image[crop_xyxy[1] : crop_xyxy[3], crop_xyxy[0] : crop_xyxy[2]]
    mask_croped = mask[
        mask_crop_xyxy[1] : mask_crop_xyxy[3], mask_crop_xyxy[0] : mask_crop_xyxy[2]
    ]

    # cv2.imwrite("croppeded_image.jpg", image_croped)
    # cv2.imwrite("croppeded_mask.jpg", mask_croped)

    image_croped_pil = Image.fromarray(cv2.cvtColor(image_croped, cv2.COLOR_BGR2RGB))
    mask_croped_pil = Image.fromarray(mask_croped)

    # image = (
    #     Image.open("../images/test_images/room.jpg").convert("RGB").resize((512, 512))
    # )
    # mask = Image.open("../images/output_images/room/masks/mask_5.jpg").convert("L")

    # print("image size, mode: ", image.size, image.mode)
    # print("mask size, mode: ", mask.size, mask.mode)
    print("image_croppeded size, mode: ", image_croped_pil.size, image_croped_pil.mode)
    print("mask_croppeded size, mode: ", mask_croped_pil.size, mask_croped_pil.mode)

    # sys.exit()

    while True:
        print("Input your prompt >")
        prompt = input()

        if prompt == "exit":
            break

        print("your prompt is: ", prompt)

        # changing_object.change_object(image, mask, prompt)
        changing_object.change_object(
            image_croped_pil,
            mask_croped_pil,
            prompt,
            save_dir_path=changing_object.image_save_path,
        )
