import tensorflow as tf

from six.moves.urllib.request import urlopen
from six import BytesIO

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

import tempfile

import numpy as np


class ObjectDetection:
    def __init__(self, detector):
        self.detector = detector
        self.path = None

    def downloadResize(self, url, width=256, height=256):
        """

        :param url:
        :param width:
        :param height:
        :return:
        """
        _, filename = tempfile.mkstemp(suffix=".jpg")
        response = urlopen(url)
        image_data = response.read()
        image_data = BytesIO(image_data)
        pil_image = Image.open(image_data)
        pil_image = ImageOps.fit(pil_image, (width, height), Image.ANTIALIAS)
        pil_image_rgb = pil_image.convert("RGB")
        pil_image_rgb.save(filename, format="JPEG", quality=90)
        print("[od]\tImage temporarily downloaded to {}".format(filename))
        return filename

    def drawBoundingBox(self, image, ymin, xmin, ymax, xmax,
                        color, font, thickness=4, display_str_list=()):
        """

        :param image:
        :param ymin:
        :param xmin:
        :param ymax:
        :param xmax:
        :param color:
        :param font:
        :param thickness:
        :param display_str_list:
        :return:
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness, fill=color)

        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height

        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str, fill="black", font=font)

            text_bottom -= text_height - 2 * margin

    def drawBoxes(self, image, boxes, class_names, scores, max_boxes=10,
                  min_score=0.1):
        """

        :param image:
        :param boxes:
        :param class_names:
        :param scores:
        :param max_boxes:
        :param min_score:
        :return:
        """
        colors = list(ImageColor.colormap.values())

        font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                               int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                self.drawBoundingBox(image_pil, ymin, xmin, ymax, xmax,
                                  color, font, display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
        return image

    def loadImg(self, path):
        """Load Image from path

        :param path: Imagepath
        :return: img
        """
        self.path = path
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def runDetector(self, path):
        """Runs detector on Image

        :param path: Imagepath
        :return:
        """
        img = self.loadImg(path)

        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        converted_img = converted_img[:, :, :, :3]
        result = self.detector(converted_img)

        result = {key: value.numpy() for key, value in result.items()}

        image_with_boxes = self.drawBoxes(img.numpy(), result["detection_boxes"],
                                          result["detection_class_entities"],
                                          result["detection_scores"])

        return image_with_boxes
    
    def saveImage(self, image):
        image = Image.fromarray(image_with_boxes, "RGB")
        image.save(self.path[:-4] + "_od.jpg", format="JPEG", quality=90)
    
