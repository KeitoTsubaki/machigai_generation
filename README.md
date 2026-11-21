# フロー
セグメンテーション -> 画像説明生成 -> 画像変更


# セグメンテーション
- SAM2
- Mobile SAM
- Fast SAM

複数のモデルで検証した結果、質は低くなるがとても高速な FastSAM_Large を採用した。
| model         | inference time (room.jpg) [s] | quality | the number of objects | model size |
| :------------ | ----------------------------: | :-----: | :-------------------: | ---------: |
| SAM2_Large    |                      215.1906 |  high   |         many          |      428MB |
| SAM2_Base     |                       16.2857 |  high   |         many          |      154MB |
| SAM2_Small    |                       12.6712 |  high   |        normal         |       88MB |
| SAM2_Tiny     |                       12.5818 |  high   |          few          |     74.4MB |
| MobileSAM     |                       28.0082 | medium  |         many          |     38.8MB |
| FastSAM_Small |                        0.1031 |   low   |         many          |     22.7MB |
| FastSAM_Large |                        0.1619 |   low   |         many          |      138MB |


### sample (Fast_SAM_Large)
<img src="./images/output_images/room_fastsam_x.jpg" width="300">
<img src="./images/output_images/room_mask_8.jpg" width="300">
<img src="./images/output_images/room_object_8.jpg" width="300">
<img src="./images/output_images/room_box_8.jpg" width="300">

<details><summary>Other samples</summary>

### sample (SAM2_Large)
<img src="./images/output_images/room_sam2_l.jpg" width="300">

### sample (SAM2_Base)
<img src="./images/output_images/room_sam2_b.jpg" width="300">

### sample (SAM2_Small)
<img src="./images/output_images/room_sam2_s.jpg" width="300">

### sample (SAM2_Tiny)
<img src="./images/output_images/room_sam2_t.jpg" width="300">

### sample (MobileSAM)
<img src="./images/output_images/room_mobilesam.jpg" width="300">

### sample (Fast_SAM_Small)
<img src="./images/output_images/room_fastsam_s.jpg" width="300">
</details>

<br>

# 画像説明生成
- BLIP2
- CoCa
- それか普通に画像分類

BLIP2はモデルサイズが大きく推論時間が長いので、CoCaを採用した。

| model | inference time (room_box_8.jpg) [s] | model size |
| :---: | ----------------------------------: | ---------: |
| BLIP2 |                      1890.167497847 |       15GB |
| CoCa  |                   2.197664805999920 |      2.5GB |

### sample(CoCa)
<img src="./images/output_images/room_box_8.jpg" width="150">

caption: **"sofa covers for living room modern elastic slipcovers stretch couch cover sofa covers for 1 2 3 4"**

<details><summary>Other samples</summary>

### BLIP2 (prompt: "What is it?")
<img src="./images/output_images/room_box_8.jpg" width="150">

caption: **"It is a sofa"**

</details>

# 画像変更
- Stable Diffusion 2 Inpainting

|             model             | inference time (croppeded_image_square.jpg) [s] |
| :---------------------------: | ----------------------------------: |
| Stable Diffusion 2 Inpainting |                                  11 |

<img src="./images/output_images/croppeded_image_square.jpg" width="150">
<img src="./images/output_images/blurred_mask.jpg" width="150">
<img src="./images/output_images/out_a_real_dog,_sitting,_highly_detailed,_8k.jpg" width="150">

prompt: **"a real dog, sitting, highly detailed, 8k"**
