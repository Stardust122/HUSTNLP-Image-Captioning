import pathlib
import json
import time
import textwrap
import google.generativeai as genai
from IPython.display import display
import PIL.Image

# API
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# list models
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel('gemini-1.5-flash')

# prompt
prompt = """
Please provide a detailed description of the given image. The description should include notable objects, actions, background, and important details. Here is an example:

Example:
Input image: A young boy playing in the park.
Output description: A young boy wearing a red T-shirt and blue jeans is playing happily in a park surrounded by trees. Beside him is a yellow soccer ball and a blue water bottle. In the background, some people are walking.
"""

data_dir = pathlib.Path('nlp_train')

max_images_per_day = 1000

results = []

json_file_path = 'train.json'
if pathlib.Path(json_file_path).exists():
    with open(json_file_path, 'r') as f:
        results = json.load(f)

processed_images = len(results)

image_files = list(data_dir.glob('*.jpg'))
images_to_process = image_files[processed_images:processed_images + max_images_per_day]

max_requests_per_minute = 15
request_count = 0
start_time = time.time()

for img_path in images_to_process:

    img = PIL.Image.open(img_path)

    try:
        response = model.generate_content([prompt, img], stream=True)

        response_data = list(response)
        description = response_data[0].content['parts'][0]['text'].strip()

        results.append({
            'image': img_path.name,
            'description': description
        })

        request_count += 1

        elapsed_time = time.time() - start_time
        if request_count % max_requests_per_minute == 0:
            sleep_time = 60 - elapsed_time % 60
            print(f"Sleeping for {sleep_time:.2f} seconds to meet the rate limit.")
            time.sleep(sleep_time)

        print(f"Processed image {img_path.name}. Total processed: {len(results)}/{max_images_per_day}")

        if len(results) >= max_images_per_day:
            break
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

# save
with open(json_file_path, 'w') as f:
    json.dump(results, f, indent=4)

print("Descriptions generated and saved to 'train.json'")