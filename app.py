import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
MAX_IMAGES = 150

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update for the captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))

    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # if it's a caption text file skip the next bit
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # if caption_path exists, do not write
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. use the existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, *captions):
    print(f"run_captioning")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        print(f"inputs {inputs}")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        print(f"generated_ids {generated_ids}")

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"generated_text: {generated_text}")
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(f"parsed_answer = {parsed_answer}")
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
        print(f"download {base_model}")
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    # download vae
    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        gr.Info(f"Downloading vae")
        print(f"downloading ae.sft...")
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    # download clip
    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        gr.Info(f"Downloading clip...")
        print(f"download clip_l.safetensors")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    # download t5xxl
    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        print(f"download t5xxl_fp16.safetensors")
        gr.Info(f"Downloading t5xxl...")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")


def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):

    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, sample_every_n_steps={sample_every_n_steps}")

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
      # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"


    #######################################################
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    if base_model == "flux-dev" or base_model == "flux-schnell":
        model_folder = "models/unet"
    else:
        model_folder = f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)

    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # check dirty
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # dirty
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images, current_project=""):
    """
    Extended version that works with both images and loaded projects
    """
    try:
        num_images = 0
        
        # First try to get count from images widget (normal upload)
        if images and len(images) > 0:
            num_images = len(images)
            print(f"📊 Using uploaded images: {num_images}")
        
        # If no images, but current_project available (loaded project)
        elif current_project and current_project != "Select a project...":
            project_data = load_project_data(current_project)
            if project_data:
                num_images = len(project_data.get('images', []))
                print(f"📊 Using loaded project images: {num_images} from '{current_project}'")
        
        if num_images > 0:
            total_steps = max_train_epochs * num_images * num_repeats
            print(f"📊 Calculated: {max_train_epochs} epochs × {num_images} images × {num_repeats} repeats = {total_steps}")
            return gr.update(value=total_steps)
        else:
            print("📊 No images found for calculation")
            return gr.update(value=0)
            
    except Exception as e:
        print(f"Error in update_total_steps: {e}")
        return gr.update()

def update_total_steps_from_loaded_project(max_train_epochs, num_repeats, project_name):
    """
    Alternative version of update_total_steps for loaded projects
    """
    try:
        if not project_name or project_name == "Select a project...":
            return gr.update()
            
        # Get number of images from loaded project
        project_data = load_project_data(project_name)
        if project_data:
            num_images = len(project_data.get('images', []))
            total_steps = max_train_epochs * num_images * num_repeats
            print(f"📊 Updated total steps from loaded project: {max_train_epochs} × {num_images} × {num_repeats} = {total_steps}")
            return gr.update(value=total_steps)
    except Exception as e:
        print(f"Error updating total steps from loaded project: {e}")
    
    return gr.update()

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)


def update(
    base_model,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolution,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    # IMPORTANT: Only update if sample_prompts is empty (prevents overwriting when loading project)
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced():
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    advanced_component_ids = []
    advanced_components = []
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            print("")
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
    #            elif action_type == "<class 'str'>":
    #                component = gr.Textbox()
    #            elif action_type == "<class 'int'>":
    #                component = gr.Number(precision=0)
    #            elif action_type == "<class 'float'>":
    #                component = gr.Number()
    #            elif "int_or_float" in action_type:
    #                component = gr.Number()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                if action['help'] != None:
                    component.info = action['help']
            advanced_components.append(component)
            advanced_component_ids.append(component.elem_id)
    return advanced_components, advanced_component_ids

# =============================================================================
# LOAD PROJECT FUNCTIONS
# =============================================================================

def scan_existing_projects():
    """
    Scans for complete FluxGym projects
    Returns: List of project names that can be loaded
    """
    projects = []

    try:
        outputs_path = "outputs"
        if not os.path.exists(outputs_path):
            return []

        for item in os.listdir(outputs_path):
            project_path = os.path.join(outputs_path, item)
            dataset_path = os.path.join("datasets", item)
            toml_path = os.path.join(project_path, "dataset.toml")

            # Check if complete project exists
            if (os.path.isdir(project_path) and
                os.path.isdir(dataset_path) and
                os.path.exists(toml_path)):

                # Additionally check if images are present in dataset
                images = get_project_images(item)
                if len(images) > 0:
                    projects.append(item)

    except Exception as e:
        print(f"Error scanning projects: {e}")

    projects.sort(reverse=True)  # Newest first
    return projects

def get_project_images(project_name):
    """
    Gets all images of a project from the datasets/ folder
    Returns: List of (image_path, caption_text) tuples
    """
    images = []
    dataset_path = os.path.join("datasets", project_name)

    if not os.path.exists(dataset_path):
        return images

    try:
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)
            if os.path.isfile(file_path):
                name, ext = os.path.splitext(file)

                if ext.lower() in image_extensions:
                    # Search for caption file
                    caption_path = os.path.join(dataset_path, f"{name}.txt")
                    caption_text = ""

                    if os.path.exists(caption_path):
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption_text = f.read().strip()
                        except Exception as e:
                            print(f"Error reading caption {caption_path}: {e}")

                    images.append((file_path, caption_text))

    except Exception as e:
        print(f"Error getting project images: {e}")

    return images

def load_project_settings(project_name):
    """
    Loads training settings from dataset.toml
    Returns: Dict with settings or None
    """
    toml_path = os.path.join("outputs", project_name, "dataset.toml")

    if not os.path.exists(toml_path):
        return None

    try:
        config = toml.load(toml_path)

        # Extract relevant settings
        settings = {}

        # General settings
        if 'general' in config:
            general = config['general']
            settings['caption_extension'] = general.get('caption_extension', '.txt')
            settings['shuffle_caption'] = general.get('shuffle_caption', False)

        # Dataset settings
        if 'datasets' in config and len(config['datasets']) > 0:
            dataset = config['datasets'][0]
            settings['resolution'] = dataset.get('resolution', 512)
            settings['batch_size'] = dataset.get('batch_size', 1)

            # Subset settings (here is the class_tokens/trigger word)
            if 'subsets' in dataset and len(dataset['subsets']) > 0:
                subset = dataset['subsets'][0]
                settings['class_tokens'] = subset.get('class_tokens', '')
                settings['num_repeats'] = subset.get('num_repeats', 10)

        return settings

    except Exception as e:
        print(f"Error loading project settings: {e}")
        return None

def load_project_data(project_name):
    """
    Loads all data of a project
    Returns: Dict with all project data
    """
    if not project_name:
        return None

    # Load images and captions
    images = get_project_images(project_name)

    # Load settings
    settings = load_project_settings(project_name)

    # Load sample prompts (if available)
    sample_prompts = ""
    prompts_path = os.path.join("outputs", project_name, "sample_prompts.txt")
    
    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                sample_prompts = f.read().strip()
        except Exception as e:
            print(f"Error reading sample prompts: {e}")

    # Load train script (train.bat or train.sh)
    train_script = ""
    script_extensions = ['.bat', '.sh']
    for ext in script_extensions:
        script_path = os.path.join("outputs", project_name, f"train{ext}")
        if os.path.exists(script_path):
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    train_script = f.read()
                break
            except Exception as e:
                print(f"Error reading train script {script_path}: {e}")

    # Load dataset config (dataset.toml)
    train_config = ""
    toml_path = os.path.join("outputs", project_name, "dataset.toml")
    if os.path.exists(toml_path):
        try:
            with open(toml_path, 'r', encoding='utf-8') as f:
                train_config = f.read()
        except Exception as e:
            print(f"Error reading dataset.toml {toml_path}: {e}")

    return {
        'name': project_name,
        'images': images,
        'settings': settings or {},
        'sample_prompts': sample_prompts,
        'train_script': train_script,
        'train_config': train_config
    }

def populate_original_captioning_ui(project_data):
    """
    Populates the original captioning UI (table) with project data
    Returns: Updates for all output_components (like load_captioning)
    """
    if not project_data or not project_data.get('images'):
        # Empty updates if no data
        updates = [gr.update(visible=False)]  # captioning_area
        for i in range(1, MAX_IMAGES + 1):
            updates.append(gr.update(visible=False))  # captioning_row
            updates.append(gr.update(value=None, visible=False))  # image
            updates.append(gr.update(value="", visible=False))  # caption
        updates.append(gr.update(visible=False))  # start button area 1
        updates.append(gr.update(visible=False))  # start button area 2  
        return updates

    images = project_data['images']
    concept_sentence = project_data['settings'].get('class_tokens', '')
    
    updates = []
    
    # Make captioning_area visible
    updates.append(gr.update(visible=True))
    
    # For each row (1 to MAX_IMAGES)
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(images)
        
        if visible:
            image_path, caption_text = images[i - 1]
            
            # Row visible
            updates.append(gr.update(visible=True))
            
            # Set image
            updates.append(gr.update(value=image_path, visible=True))
            
            # Set caption (or concept_sentence as fallback)
            final_caption = caption_text if caption_text else concept_sentence
            updates.append(gr.update(value=final_caption, visible=True))
        else:
            # Hide row
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None, visible=False))
            updates.append(gr.update(value="", visible=False))
    
    # Make sample caption area and start button visible
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))
    
    return updates

def on_load_project(project_name):
    """
    Event handler when a project is loaded - settings only
    Returns: Updates for the main UI components
    """
    if not project_name or project_name == "Select a project...":
        # Empty update if nothing selected
        return (
            gr.update(),  # lora_name
            gr.update(),  # concept_sentence
            gr.update(),  # resolution
            gr.update(),  # num_repeats
            gr.update(),  # sample_prompts
            gr.update(),  # vram
            gr.update(),  # max_train_epochs
            gr.update(),  # base_model
            gr.update(),  # train_script
            gr.update(),  # train_config
            gr.update(),  # total_steps (leave empty for now)
            gr.update(value="ℹ️ Select a project from the dropdown to load it.")  # load_status
        )

    try:
        # Load project data
        project_data = load_project_data(project_name)

        if not project_data:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(value="❌ Failed to load project data.")
            )

        settings = project_data.get('settings', {})
        sample_prompts_content = project_data.get('sample_prompts', '')

        # Generate UI updates
        lora_name_update = gr.update(value=project_data['name'])
        trigger_word_update = gr.update(value=settings.get('class_tokens', ''))
        resolution_update = gr.update(value=settings.get('resolution', 512))
        num_repeats_update = gr.update(value=settings.get('num_repeats', 10))
        sample_prompts_update = gr.update(value=sample_prompts_content)
        
        vram_update = gr.update(value="20G")
        max_epochs_update = gr.update(value=16)
        base_model_update = gr.update()

        # Train script and config updates
        train_script_update = gr.update(value=project_data.get('train_script', ''))
        train_config_update = gr.update(value=project_data.get('train_config', ''))

        # TOTAL STEPS: Leave empty for now - will be calculated in next step
        total_steps_update = gr.update()

        # Status message
        num_images = len(project_data.get('images', []))
        status_msg = f"✅ Loaded project '{project_data['name']}' with {num_images} images"
        if sample_prompts_content:
            status_msg += f" and {len(sample_prompts_content.splitlines())} sample prompts"
        status_update = gr.update(value=status_msg)

        return (
            lora_name_update, trigger_word_update, resolution_update, num_repeats_update,
            sample_prompts_update, vram_update, max_epochs_update, base_model_update,
            train_script_update, train_config_update, total_steps_update, status_update
        )

    except Exception as e:
        print(f"Error loading project: {e}")
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(value=f"❌ Error loading project: {str(e)}")
        )

def load_project_captioning_ui(project_name):
    """
    Loads the captioning UI for a project (table with image + caption)
    """
    if not project_name or project_name == "Select a project...":
        return populate_original_captioning_ui(None)
    
    try:
        project_data = load_project_data(project_name)
        return populate_original_captioning_ui(project_data)
    except Exception as e:
        print(f"Error loading captioning UI: {e}")
        return populate_original_captioning_ui(None)

def refresh_projects_list():
    """
    Refreshes the list of available projects
    """
    projects = scan_existing_projects()
    choices = ["Select a project..."] + projects
    return gr.update(choices=choices, value="Select a project...")

def calculate_total_steps_for_loaded_project(project_name):
    """
    Calculates total_steps for a loaded project AFTER the UI update
    """
    if not project_name or project_name == "Select a project...":
        return gr.update()
        
    try:
        project_data = load_project_data(project_name)
        if project_data:
            num_images = len(project_data.get('images', []))
            num_repeats = project_data.get('settings', {}).get('num_repeats', 10)
            max_epochs = 16  # Default UI value
            
            total_steps = max_epochs * num_images * num_repeats
            print(f"📊 Final total steps calculation: {max_epochs} × {num_images} × {num_repeats} = {total_steps}")
            return gr.update(value=total_steps)
    except Exception as e:
        print(f"Error calculating total steps: {e}")
    
    return gr.update()

def load_sample_prompts_only(project_name):
    """
    Loads only the sample prompts for a project (fix for event conflict)
    """
    if not project_name or project_name == "Select a project...":
        return gr.update()
        
    prompts_path = os.path.join("outputs", project_name, "sample_prompts.txt")
    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                sample_prompts = f.read().strip()
            print(f"🔧 SECOND PASS: Loading sample prompts: {len(sample_prompts)} chars")
            return gr.update(value=sample_prompts)
        except Exception as e:
            print(f"Error in second pass loading sample prompts: {e}")
    
    return gr.update()
    """
    Clears the current project and resets UI to initial state
    Returns: Updates for all main UI components
    """
    print("🧹 Clearing project - resetting UI to initial state")
    
    # Default values for UI reset
    return (
        gr.update(value=""),                    # lora_name
        gr.update(value=""),                    # concept_sentence  
        gr.update(value=512),                   # resolution
        gr.update(value=10),                    # num_repeats
        gr.update(value=""),                    # sample_prompts
        gr.update(value="20G"),                 # vram
        gr.update(value=16),                    # max_train_epochs
        gr.update(),                            # base_model (keep current)
        gr.update(value=""),                    # train_script
        gr.update(value=""),                    # train_config
        gr.update(value=0),                     # total_steps
        gr.update(value="Select a project..."), # project_dropdown
        gr.update(value=""),                    # current_project state
        gr.update(value="🧹 Project cleared. Ready for new training setup.")  # load_status
    )

def clear_project_captioning_ui():
    """
    Clears the captioning UI (hides all image/caption rows)
    Returns: Updates for all captioning output_components
    """
    print("🧹 Clearing captioning UI")
    
    updates = []
    
    # Hide captioning_area
    updates.append(gr.update(visible=False))
    
    # Hide all captioning rows (1 to MAX_IMAGES)
    for i in range(1, MAX_IMAGES + 1):
        updates.append(gr.update(visible=False))    # captioning_row
        updates.append(gr.update(value=None, visible=False))   # image
        updates.append(gr.update(value="", visible=False))     # caption
    
    # Hide start button areas
    updates.append(gr.update(visible=False))    # start button area 1
    updates.append(gr.update(visible=False))    # start button area 2
    
    return updates

def clear_project():
    """
    Clears the current project and resets UI to initial state
    Returns: Updates for all main UI components
    """
    print("🧹 Clearing project - resetting UI to initial state")
    
    # Default values for UI reset
    return (
        gr.update(value=""),                    # lora_name
        gr.update(value=""),                    # concept_sentence  
        gr.update(value=512),                   # resolution
        gr.update(value=10),                    # num_repeats
        gr.update(value=""),                    # sample_prompts
        gr.update(value="20G"),                 # vram
        gr.update(value=16),                    # max_train_epochs
        gr.update(),                            # base_model (keep current)
        gr.update(value=""),                    # train_script
        gr.update(value=""),                    # train_config
        gr.update(value=""),                        # total_steps (empty string for Textbox)
        gr.update(value="Select a project..."), # project_dropdown
        gr.update(value=""),                    # current_project state
        gr.update(value="🔄 Form reset. Ready for new training setup.")  # load_status
    )

def clear_project_captioning_ui():
    """
    Clears the captioning UI (hides all image/caption rows)
    Returns: Updates for all captioning output_components
    """
    print("🧹 Clearing captioning UI")
    
    updates = []
    
    # Hide captioning_area
    updates.append(gr.update(visible=False))
    
    # Hide all captioning rows (1 to MAX_IMAGES)
    for i in range(1, MAX_IMAGES + 1):
        updates.append(gr.update(visible=False))    # captioning_row
        updates.append(gr.update(value=None, visible=False))   # image
        updates.append(gr.update(value="", visible=False))     # caption
    
    # Hide start button areas
    updates.append(gr.update(visible=False))    # start button area 1
    updates.append(gr.update(visible=False))    # start button area 2
    
    return updates

# =============================================================================
# GRADIO UI
# =============================================================================

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    # =============================================================================
                    # LOAD PROJECT SECTION - Added at the top
                    # =============================================================================
                    gr.Markdown("## 🔄 Load Existing Project")
                    with gr.Row():
                        with gr.Column(scale=3):
                            project_dropdown = gr.Dropdown(
                                label="Select Project",
                                choices=["Select a project..."],
                                value="Select a project...",
                                interactive=True
                            )
                        with gr.Column(scale=1):
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        with gr.Column(scale=1):
                            load_btn = gr.Button("📂 Load Project", variant="primary")

                    with gr.Row():
                        load_status = gr.Markdown("ℹ️ Select a project from the dropdown to load it.")

                    # No separate preview - we use the original captioning UI

                    # =============================================================================
                    # ORIGINAL STEP 1 - LoRA Info
                    # =============================================================================
                    gr.Markdown(
                        """# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="Base model (edit the models.yaml file to add more to this list)", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Textbox("", interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images")

                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        images = gr.File(
                            file_types=["image", ".txt"],
                            label="Upload your images",
                            #info="If you want, you can also manually upload caption files that match the image names (example: img0.png => img0.txt)",
                            file_count="multiple",
                            interactive=True,
                            visible=True,
                            scale=1,
                        )
                    with gr.Group(visible=False) as captioning_area:
                        do_captioning = gr.Button("Add AI captions with Florence-2")
                        output_components.append(captioning_area)
                        #output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    with gr.Row():
                        refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                        start = gr.Button("Start training", visible=False, elem_id="start_training")
                        clear_btn = gr.Button("🔄 Reset Form", variant="secondary")
                    output_components.append(start)
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="--learning_rate", info="Learning Rate", value="8e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=4, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(label="--network_dim", info="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                    advanced_components, advanced_component_ids = init_advanced()
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()
    current_project = gr.State()  # NEW: State for currently loaded project

    listeners = [
        base_model,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    
    # =============================================================================
    # ORIGINAL EVENT HANDLERS
    # =============================================================================
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    ).then(
        # Reset current project state on new upload
        fn=lambda: "",
        outputs=[current_project]
    )
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    ).then(
        # Reset current project state
        fn=lambda: "",
        outputs=[current_project]
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    ).then(
        # Reset current project state
        fn=lambda: "",
        outputs=[current_project]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images, current_project],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images, current_project],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images, current_project],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images, current_project],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images, current_project],
        outputs=[total_steps]
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])

    # =============================================================================
    # LOAD PROJECT EVENT HANDLERS - IMPROVED
    # =============================================================================
    refresh_btn.click(
        fn=refresh_projects_list,
        outputs=[project_dropdown]
    )

    # =============================================================================
    # LOAD PROJECT EVENT HANDLERS - FINAL VERSION
    # =============================================================================
    refresh_btn.click(
        fn=refresh_projects_list,
        outputs=[project_dropdown]
    )

    # Load project in 3 steps to avoid event conflicts
    load_btn.click(
        # Step 1: Load settings
        fn=on_load_project,
        inputs=[project_dropdown],
        outputs=[
            lora_name,              # LoRA Name
            concept_sentence,       # Trigger Word 
            resolution,             # Resolution
            num_repeats,            # Num Repeats
            sample_prompts,         # Sample Prompts
            vram,                   # VRAM Setting
            max_train_epochs,       # Max Train Epochs
            base_model,             # Base Model (no change)
            train_script,           # Train Script (train.bat content)
            train_config,           # Train Config (dataset.toml content)
            total_steps,            # Expected training steps (FIX!)
            load_status             # Status Message
        ]
    ).then(
        # Step 2: Set sample prompts again explicitly
        fn=lambda project_name: load_sample_prompts_only(project_name) if project_name != "Select a project..." else gr.update(),
        inputs=[project_dropdown],
        outputs=[sample_prompts]
    ).then(
        # Step 2.5: Set current project state for total_steps updates
        fn=lambda project_name: project_name if project_name != "Select a project..." else "",
        inputs=[project_dropdown],
        outputs=[current_project]
    ).then(
        # Step 3: Fill original captioning UI (the table!)
        fn=load_project_captioning_ui,
        inputs=[project_dropdown],
        outputs=output_components  # This is the original UI table
    ).then(
        # Step 4: Calculate total steps AFTER all UI values are set
        fn=calculate_total_steps_for_loaded_project,
        inputs=[project_dropdown],
        outputs=[total_steps]
    )

    # Clear project functionality - two steps to avoid timing issues
    clear_btn.click(
        # Step 1: Clear current project state first to prevent auto-recalculation
        fn=lambda: "",
        outputs=[current_project]
    ).then(
        # Step 2: Clear main UI components
        fn=clear_project,
        outputs=[
            lora_name,              # LoRA Name
            concept_sentence,       # Trigger Word 
            resolution,             # Resolution
            num_repeats,            # Num Repeats
            sample_prompts,         # Sample Prompts
            vram,                   # VRAM Setting
            max_train_epochs,       # Max Train Epochs
            base_model,             # Base Model (no change)
            train_script,           # Train Script
            train_config,           # Train Config
            total_steps,            # Expected training steps
            project_dropdown,       # Reset dropdown
            load_status             # Status Message
        ]
    ).then(
        # Step 3: Clear captioning UI (hide all image/caption rows)
        fn=clear_project_captioning_ui,
        outputs=output_components  # This clears the captioning table
    )

    # Initial loads
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    demo.load(fn=refresh_projects_list, outputs=[project_dropdown])

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(debug=True, show_error=True, allowed_paths=[cwd])
