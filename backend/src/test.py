import requests

url = "http://127.0.0.1:8000/api/generate-caption/"
image_path = "C:\\Users\\grado\\Desktop\\dsl-research-assistant\\backend\\image_captions\\Abdal_Gaussian_Shell_Maps_for_Efficient_3D_Human_Generation_CVPR_2024_paper\\Abdal_Gaussian_Shell_Maps_for_Efficient_3D_Human_Generation_CVPR_2024_paper-picture-8.png"

with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    data = {
        "prompt": "Describe this image.",
        "context": "This is for a CVPR 2024 paper."
    }
    response = requests.post(url, files=files, data=data)

print("Response:", response.json())


url2 = "http://localhost:8000/api/generate-citations/"
data2 = {
    "text": "Explain the basics of machine learning and its real-world applications."
}

response2 = requests.post(url2, json=data2)

print("Response:", response2.json())

url3 = "http://localhost:8000/api/generate-continuation/"
data3 = {
    "text": "In the field of artificial intelligence, machine learning has emerged as a powerful tool. "
            "It allows computers to learn from data and make predictions or decisions without being explicitly programmed."
}

response3 = requests.post(url3, json=data3)
print("Response:", response3.json())