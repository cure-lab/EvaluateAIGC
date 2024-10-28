import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

def get_attributes(attribute_names, raw_image):
    # questions
    questions = {'person': "Is there a real human in the picture?",
                 'age': "How old is this person?",
                 'gender': "What is the gender of this person?",
                 'race': "What is the race of this person?"}
    for key, value in questions.items():
        questions[key] = txt_processors["eval"](value)
    raw_image = raw_image.convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = {}
    for key, value in questions.items():
        if key in attribute_names:
            answer = model.predict_answers(samples={"image": image, "text_input": value}, inference_method="generate")
            result[key] = answer
    return result