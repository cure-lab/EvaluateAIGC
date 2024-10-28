from utils.detection import Detection
from utils.clip import get_clip_feature
from attribute.vqa import get_attributes
from PIL import Image
from aesthetic.scores import get_aesthetic_scores
from defect.face_defect import get_defect_rate

class GenerativeDebug():
    def __init__(self):
        self.body_detection = Detection("body")

    def encode(self, image):
        return get_clip_feature(image)

    def get_image_attribute(self, path, attributes):
        det = Detection("body")
        human_imgs = det.crop_image(path)
        results = []
        for bbox, human_img in human_imgs:
            results.append([bbox, get_attributes(human_img, attributes)])
        return results
    
    def get_aesthetic_score(self, path):
        image = Image.open(path)
        feature = self.encode(image)
        return get_aesthetic_scores(image, feature)
    
    def get_image_face_defect(self, path):
        det = Detection("face")
        face_imgs = det.crop_image(path)
        results = []
        for bbox, face_img in face_imgs:
            result = get_defect_rate(face_img)
            results.append([bbox, result])
        return results
