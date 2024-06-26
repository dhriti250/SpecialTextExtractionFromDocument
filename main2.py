import re
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import random
import numpy as np
from flask import Flask, request, jsonify
from datasets import load_dataset
import json

# Load our model from Hugging Face
processor = DonutProcessor.from_pretrained("philschmid/donut-base-sroie")
model = VisionEncoderDecoderModel.from_pretrained("philschmid/donut-base-sroie")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize Flask app
app = Flask(__name__)

# Load dataset
base_path = "Test"
dataset_test = load_dataset("imagefolder", data_dir=base_path, split="train")

new_special_tokens = [] # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>" # eos token of tokenizer

def json2token_test(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token_test(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token_test(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut_test(sample):
    # create Donut-style input
    # text = json.loads(sample["text"])
    # d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image}

proc_dataset_test = dataset_test.map(preprocess_documents_for_donut_test)


def transform_and_tokenize_test(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}


    return {"pixel_values": pixel_values}

# need at least 32-64GB of RAM to run this
processed_dataset_test = proc_dataset_test.map(transform_and_tokenize_test,remove_columns=["image"])

test_sample_test = processed_dataset_test

def run_prediction(sample, model=model, processor=processor):
    # prepare inputs
    pixel_values = torch.tensor(test_sample_test["pixel_values"]).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=False,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    # load reference target
    #target = processor.token2json(test_sample["target_sequence"])
    return prediction

# Define endpoint for generating predictions
@app.route('/generate_prediction', methods=['POST'])
def generate_prediction():
    # Load the image
    # image_file = request.files['image']
    # image = Image.open(image_file)

    # # Preprocess image for the model
    # sample = {"image": image}
    # pixel_values = processor(sample, return_tensors="pt").pixel_values.squeeze()

    # # Run prediction
    for i in range(0,9):
        with torch.no_grad():
            # outputs = model(pixel_values.to(device))
            prediction = run_prediction(test_sample_test[i])
            # prediction = processor.batch_decode(outputs.sequences)[0]
            prediction_json = json.dumps(prediction)
            print(prediction)
            print("JSON FOrmat")
            print(prediction_json)

    #return jsonify({"prediction": prediction_json})

if __name__ == '__main__':
    #  app.run(debug=True)
    generate_prediction()
    
    

