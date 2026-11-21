import time
import gc

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import open_clip
import cv2
import torch
from PIL import Image


# CoCa
class Captioning_CoCa:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        model_path = "../models/open_clip_pytorch_model.bin"
        print("Loading Captioning model...")
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            "coca_ViT-L-14",
            pretrained=model_path,
        )

        #self.model, _, self.transform = open_clip.create_model_and_transforms(
        #    'coca_ViT-L-14',
        #    pretrained='mscoco_finetuned_laion2b_s13b_b90k',  # 公式タグを使う
        #    device=device
        #)

        self.model.to(self.device)
        print("The Captioning model has been loaded.")

    def caption_image(self, image):
        """
        Args:
            - image: cv2 image
        Return:
            - caption: str
        """
        start_time = time.perf_counter()
        if not isinstance(image, Image.Image):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = image
        im = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            print("Captioning...")
            out = self.model.generate(im, seq_len=20)
            print("The image has been captioned.")

        caption = (
            open_clip.decode(out[0].detach())
            .split("<end_of_text>")[0]
            .replace("<start_of_text>", "")
        )
        end_time = time.perf_counter()
        print(caption)
        print("CoCa inference time: ", end_time - start_time, "[s]")

        # キャッシュを削除
        del out, im, pil_image, img_rgb
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return caption


if __name__ == "__main__":

    captioning_coca = Captioning_CoCa()

    image = cv2.imread("../images/output_images/room_box_8.jpg")
    captioning_coca.caption_image(image)

    # captioning_blip2 = Captioning_BLIP2()
    # print("start captioning")
    # captioning_blip2.caption_image(image)
