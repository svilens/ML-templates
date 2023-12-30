from datasets import load_dataset

data = load_dataset("sbu_captions", split="train").shuffle(seed=42)


##########
# Data processor
##########

import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO


class ProcessDataset(Dataset):
    def __init__(self, df, tokenizer,feature_extractor, decoder_max_length=20):
        self.df = df
        self.tokenizer = tokenizer # this is for language model 
        self.feature_extractor = feature_extractor # this is for vision model 
        self.decoder_max_length = decoder_max_length # this is for caption output

    def __len__(self):
        # this is necessary so that HuggingFace won't complain that the dataset doesn't have __len__ method 
        # when it starts training
        return len(self.df)

    def __getitem__(self, idx):
        # this is another method name that HuggingFace expects 
        # get file name + text 
        img_path = self.df["image_url"][idx]
        caption = self.df["caption"][idx]
        
        # process image 
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content))
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        # labels here refer to each token in the caption
        labels = self.tokenizer(caption, 
                                truncation=True,
                                padding="max_length", 
                                max_length=self.decoder_max_length).input_ids

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


##########
# Init tokenizer and feature extractor
##########

from transformers import GPT2TokenizerFast, ViTFeatureExtractor

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# GPT2 doesn't have a pad token 
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
train_dataset = ProcessDataset(df=data[:2000],
                               tokenizer=tokenizer,
                               feature_extractor=feature_extractor)

from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained\
    (encoder_pretrained_model_name_or_path="google/vit-base-patch16-224-in21k", 
     decoder_pretrained_model_name_or_path="gpt2", 
     tie_encoder_decoder=True)

# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# We will adjust several more model configuration settings here 
model.config.vocab_size = model.config.decoder.vocab_size
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3 # this determines a sequence of N words that cannot be repeated 
model.config.length_penalty = 2.0

# For decoder only 
model.decoder.num_beams = 4
model.decoder.max_length = 20 


##########
# Train image captioning
##########

from transformers import Trainer, TrainingArguments
from transformers import default_data_collator
import os

BATCH_SIZE = 16
TRAIN_EPOCHS = 20

output_directory = os.path.join('.', "captioning_outputs")

training_args = TrainingArguments(
    output_dir=output_directory,
    per_device_train_batch_size=BATCH_SIZE,
    do_train=True,
    num_train_epochs=TRAIN_EPOCHS,
    overwrite_output_dir=True,
    no_cuda=True,
    dataloader_pin_memory=False
)

trainer = Trainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)

##########
# Generate captions
##########

test_img = data[2021]

test_img_path = test_img["image_url"]
test_img_response = requests.get(test_img_path)
test_image = Image.open(BytesIO(test_img_response.content))
caption = tokenizer.decode(trainer.model.to("cpu").generate(feature_extractor(test_image, return_tensors="pt").pixel_values)[0])
print("--"*20)
print(caption)

# if the output isn't good enough, we might need to:
# - train the decoder
# - increase the number of epochs


##########
# A zero-shot image captioning model
##########
from transformers import BlipProcessor, BlipForConditionalGeneration

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# conditional - adding a prefix text
text = "a photo of"
inputs = blip_processor(test_image, text, return_tensors="pt")
conditional_output = blip_model.generate(**inputs)
print("Conditional output: ", blip_processor.decode(conditional_output[0], skip_special_tokens=True))

# unconditional - no prefix text
inputs = blip_processor(test_image, return_tensors="pt")
unconditional_output = blip_model.generate(**inputs)
print("Unconditional output: ", blip_processor.decode(unconditional_output[0], skip_special_tokens=True))
