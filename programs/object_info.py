class Object_info:
    def __init__(
        self, box_size, box_name, box_xyxy, mask_name, caption="", is_fix=False
    ):
        self.box_size = box_size
        self.box_name = box_name
        self.box_xyxy = box_xyxy
        self.mask_name = mask_name
        self.caption = caption
        self.is_fix = is_fix

    def set_box_size(self, box_size):
        self.box_size = box_size

    def set_box_name(self, box_name):
        self.box_name = box_name

    def set_box_xyxy(self, box_xyxy):
        self.box_xyxy = box_xyxy

    def set_mask_name(self, mask_name):
        self.mask_name = mask_name

    def set_caption(self, caption):
        self.caption = caption

    def set_is_fix(self, is_fix):
        self.is_fix = is_fix

    def show_object_info(self):
        print("<object info>===================")
        print(f"box size: {self.box_size}")
        print(f"box name: {self.box_name}")
        print(f"box xyxy: {self.box_xyxy}")
        print(f"caption: {self.caption}")
        print("================================")
