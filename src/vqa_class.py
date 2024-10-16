from transformers import AutoProcessor, AutoModelForCausalLM, BlipForQuestionAnswering, ViltForQuestionAnswering
import torch

import numpy as np

class VQA():
  def __init__(self, checkpoint):
    self.checkpoint = checkpoint
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.validate_checkpoint()
    self.processor = None
    self.model = None
    self.create_processor()
    self.create_model()
    # self.model.to(self.device)

  def validate_checkpoint(self):
    if self.model_type not in self.checkpoint:
      raise Exception("Checkpoint is either not available or it does not match with the model type!")
      return False
    return True

  def create_processor(self):
    self.processor = AutoProcessor.from_pretrained(self.checkpoint)

  def create_model(self):
    pass

  def generate_output(self, image, question):
    pass

  def generate_outputs(self, images=list(), questions=list()):
    if not self.is_len_match(len(images), len(questions)): return None
    outputs = list()
    for i, q in zip(images, questions):
      outputs.append(self.generate_output(i, q))
    return outputs

  def is_len_match(self, len_images, len_questions):
    if len_images != len_questions:
      raise Exception("Number of images and questions do not match!")
      return False
    return True

  def decode_output(self, output, skip_special_tokens=True):
    return self.processor.batch_decode(output.sequences, skip_special_tokens=skip_special_tokens)

  def get_logits(self, output):
    scores = output.scores #[:-1]
    logits = np.array([s[0].numpy() for s in scores])
    return logits #output.scores[-1][0].numpy()

  # def logits_prob(self, logits):
  #   return self.softmax_stable(logits)

  # def softmax_stable(self, x):
  #   return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


class GiT(VQA):
  def __init__(self, checkpoint):
    '''
    allowed checkpoints are: "microsoft/git-base-vqav2", "microsoft/git-large-vqav2"
    '''
    self.model_type = "git"
    super().__init__(checkpoint)

  def create_model(self):
    self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint)

  def generate_output(self, image, question):
    # prepare image
    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
    # prepare question
    input_ids = self.processor(text=question, add_special_tokens=False).input_ids
    input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # generate answer
    # if self.device == "cuda":
    #   pixel_values = pixel_values.to(self.device)
    #   input_ids = input_ids.to(self.device)
    output = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50, return_dict_in_generate=True, output_scores=True)
    # answer = self.processor.batch_decode(generated_output.sequences, skip_special_tokens=True)
    return output

  # def decode_output(self, output):
  #   defined in the parent class

  # def get_logits(self, output):
  #   defined in the parent class


class Blip(VQA):
  def __init__(self, checkpoint):
    '''
    allowed checkpoints are: "Salesforce/blip-vqa-base", "Salesforce/blip-vqa-capfilt-large"
    '''
    self.model_type = "blip"
    super().__init__(checkpoint)

  def create_model(self):
    self.model = BlipForQuestionAnswering.from_pretrained(self.checkpoint)

  def generate_output(self, image, question): # TRY output_attentions
    inputs = self.processor(images=image, text=question, return_tensors="pt")
    output = self.model.generate(**inputs, max_length=50, return_dict_in_generate=True, output_scores=True, output_attentions=True)
    # answer = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
    return output

  # def decode_output(self, output):
  #   defined in the parent class

  # def get_logits(self, output):
  #   defined in the parent class

class Vilt(VQA):
  def __init__(self, checkpoint):
    '''
    allowed checkpoint is: "dandelin/vilt-b32-finetuned-vqa"
    '''
    self.model_type = "vilt"
    super().__init__(checkpoint)

  def create_model(self):
    self.model = ViltForQuestionAnswering.from_pretrained(self.checkpoint)

  def generate_output(self, image, question):
    # prepare image + question
    encoding = self.processor(images=image, text=question, return_tensors="pt")
    with torch.no_grad():
        output = self.model(**encoding)
    # predicted_class_idx = outputs.logits.argmax(-1).item()
    # answer = self.model.config.id2label[predicted_class_idx] # check the outputs!!!
    return output

  def decode_output(self, output):
    # predicted_class_idx = output.logits.argmax(-1).item()
    predicted_class_idx = output.logits.argmax().item()
    return self.model.config.id2label[predicted_class_idx]

  def get_logits(self, output):
    scores = output.logits #[:-1]
    logits = np.array([s.numpy() for s in scores])
    return logits # np.array(output.logits[-1])