import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url = "http://images.cocodataset.org/val2017/000000039785.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# img_path = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/beach_on_a_clock.png"
img_path = "/home/czr/contrast_decoding_LVLMs/hallucinatory_image/test.png"
image = Image.open(img_path)
# texts = [["a photo of a cat"]]#, "a photo of a dog"]]
# texts = [["surfboard", "clock"]]
texts = [["cat"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
# results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
results = processor.post_process_object_detection(outputs=outputs, threshold=0.05)

print("results: ", results)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for bbox, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in bbox.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")



    # Calculate the absolute coordinates of the bounding box
    im_width, im_height = image.size
    left = int(bbox[0] * im_width)
    top = int(bbox[1] * im_height)
    right = int(bbox[2] * im_width)
    bottom = int(bbox[3] * im_height)

    print("left: ", left)

    # Crop the image to the bounding box
    cropped_image = image.crop((left, top, right, bottom))

    save_path = f"/home/czr/HaLC/decoder_zoo/HaLC/cache_dir/cropped_level_{i}.png"
    cropped_image.save(save_path)
    input()
    # saved_paths.append(save_path)
