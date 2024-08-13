import base64
import requests
import api_key

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "output/warped_124.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key.api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "These images are cells from a table of nutritional data for children under 5 years old. All responses should be reasonable measurements for a child of that age. Each image will be labeled with the column it comes from. The date column contains the date of data collection in d/m/y format, and should be after 2015. The age column contains the child’s age in years and months with “a” meaning “año” or year, and “m” meaning “mes” or month. The weight column contains the child’s weight in kg. The height column contains the child’s height in cm. For any image that does not contain a number, return “empty”. Any number may contain a decimal point or comma and may be followed by its units. A comma means the same as a decimal point."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "weight"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

response.json()["choices"][0]["message"]["content"]