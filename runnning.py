import os
import subprocess
import requests

model = "RAM"
images_dir = "images/demo"
image_files = [f"{images_dir}/{file}" for file in sorted(os.listdir(
    images_dir)) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
task = "one image"
print('You selected', model)
print('You selected', task)


def download_checkpoint(url, save_path):
    print("downloading......")
    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        print("Downloaded successfully!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")


model = "RAM"  # Specify the model type
if not os.path.exists('pretrained'):
    os.makedirs('pretrained')

if model == "RAM":
    ram_weights_path = 'pretrained/ram_swin_large_14m.pth'
    if not os.path.exists(ram_weights_path):
        url = "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth"
        download_checkpoint(url, ram_weights_path)


def run_inference_once(image_path):
    pretrained_path = "pretrained/ram_swin_large_14m.pth"

    command = [
        "python",
        "inference_ram.py",
        "--image",
        image_path,
        "--pretrained",
        pretrained_path
    ]

    # Run the command and capture the output
    completed_process = subprocess.run(command, capture_output=True, text=True)

    # Get the standard output and standard error
    stdout = completed_process.stdout
    stderr = completed_process.stderr

    # Print the output (you can also store it in a file or process it as needed)
    image_tags_line = [line for line in stdout.split(
        '\n') if line.startswith("Image Tags:")][0]
    image_tags = image_tags_line.replace("Image Tags:", "").strip()

    # Print the extracted image tags
    print("Image Tags:", image_tags)


def batch_inference():

    pretrained_path = "pretrained/ram_swin_large_14m.pth"
    model_type = "ram"

    command = [
        "python",
        "batch_inference.py",
        "--image-dir",
        images_dir,
        "--pretrained",
        pretrained_path,
        "--model-type",
        model_type
    ]

    # Run the command and capture the output
    completed_process = subprocess.run(command, capture_output=True, text=True)

    # Get the standard output and standard error
    stdout = completed_process.stdout
    stderr = completed_process.stderr


file_len = len(image_files)
for i in range(0, file_len):
    image_path = image_files[i]
    run_inference_once(image_path)
