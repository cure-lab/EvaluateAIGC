import argparse
from utils.logic import GenerativeDebug
import os

# Global Variables
ATTRIBUTES = ["gender", "age", "race"]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, help="input image dir", default="./example.jpg")
    parser.add_argument("--out-dir", type=str, help="output dir", default=".")
    return parser.parse_args()


if __name__ == '__main__':
    # parse args
    args = get_parser()
    debug = GenerativeDebug()
    image = args.image_dir
    result_path = os.join(args.output_dir, "results.json")
    aesthetic_scores = debug.get_aesthetic_score(image)
    defect_scores = debug.get_image_face_defect(image)
    image_attributes = debug.get_image_attribute(image, ATTRIBUTES)
    print("aesthetic score:", aesthetic_scores)
    print("face defect rate: ", defect_scores)
    print("fairness attributes:", image_attributes)