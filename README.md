

# BRIA Background Removal v2.0 Model Card

RMBG v2.0 is our new state-of-the-art background removal model, designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. 
The accuracy, efficiency, and versatility currently rival leading source-available models. 
It is ideal where content safety, legally licensed datasets, and bias mitigation are paramount. 

Developed by BRIA AI, RMBG v2.0 is available as a source-available model for non-commercial use. 

[CLICK HERE FOR A DEMO](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0)


<img src="https://github.com/Efrat-Taig/RMBG-1.4/blob/main/t4.png" width="700">


## Model Details
#####
### Model Description

- **Developed by:** [BRIA AI](https://bria.ai/)
- **Model type:** Background Removal 
- **License:** [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/)
  - The model is released under a Creative Commons license for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA. [Contact Us](https://bria.ai/contact-us) for more information. 

- **Model Description:** BRIA RMBG-2.0 is a dichotomous image segmentation model trained exclusively on a professional-grade dataset.
- **BRIA:** Resources for more information: [BRIA AI](https://bria.ai/)



## Training data
Bria-RMBG model was trained with over 15,000 high-quality, high-resolution, manually labeled (pixel-wise accuracy), fully licensed images.
Our benchmark included balanced gender, balanced ethnicity, and people with different types of disabilities.
For clarity, we provide our data distribution according to different categories, demonstrating our model’s versatility.

### Distribution of images:

| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Objects only | 45.11% |
| People with objects/animals | 25.24% |
| People only | 17.35% |
| people/objects/animals with text | 8.52% |
| Text only | 2.52% |
| Animals only | 1.89% |

| Category | Distribution |
| -----------------------------------| -----------------------------------------:|
| Photorealistic | 87.70% |
| Non-Photorealistic | 12.30% |


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Non Solid Background | 52.05% |
| Solid Background | 47.95% 


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Single main foreground object | 51.42% |
| Multiple objects in the foreground | 48.58% |


## Qualitative Evaluation
Open source models comparison


<img src="https://github.com/Efrat-Taig/RMBG-2.0/blob/main/collage5.png" width="700">

<img src="https://github.com/Efrat-Taig/RMBG-2.0/blob/main/diagram1.png" width="700">


### Architecture
RMBG-2.0 is developed on the [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) architecture enhanced with our proprietary dataset and training scheme. This training data significantly improves the model’s accuracy and effectiveness for background-removal task.<br>
If you use this model in your research, please cite:

```
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

#### Requirements
```bash
torch
torchvision
pillow
kornia
transformers
```

### Usage

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->


```python
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(input_image_path)
input_images = transform_image(image).unsqueeze(0).to('cuda')

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")


```

# BRIA Background Removal v2.0 Benchmark

In this benchmark, I tested the new RMBG v2.0 model alongside  our earlier version, RMBG-1.4. The updated RMBG v2.0 showed impressive improvements in background removal, especially in handling complex scenes and preserving the finer details around edges.

Setting up this benchmark was quick and easy. I used a language model to create a set of 100 diverse images, then ran them through BRIA 2.3 Fast. In about 10 minutes, I had a solid benchmark ready to go. Feel free to try it out yourself and make your own benchmark that suits your needs and  usecase with the attahed code  [creat_generated_benchmark.py](https://github.com/Efrat-Taig/RMBG-2.0/blob/main/creat_generated_benchmark.py)


Our analysis confirms that while RMBG-1.4 delivers strong performance, RMBG v2.0 sets a new standard, achieving even cleaner separations and greater consistency across a range of scenarios. We’re excited to share this improved model!



## Download the Dataset

The dataset is available for download from Google Drive. You can access it using the following link:

[**Download RMBG Benchmark Dataset**](https://drive.google.com/drive/folders/1V7H0WzqgWU6RWVvvOntBriWSPXS35fwB?usp=sharing)


<img src="https://github.com/Efrat-Taig/RMBG-2.0/blob/main/benchmark.png" width="600">

BRIA's RMBG v2.0 model is ideal for applications where high-quality background removal is essential, particularly for content creators, e-commerce, and advertising. The model’s ability to handle various image types, including challenging ones with non-solid backgrounds, makes it a valuable asset for businesses focused on legally licensed, ethically sourced datasets.
