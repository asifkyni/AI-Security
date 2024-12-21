# AI-Security

## Physical Attacks on AI Systems

### Overview
Physical attacks exploit vulnerabilities in machine learning systems by altering the physical world to deceive models. This branch of the AI Security Tutorial focuses on:
- Analyzing methods for crafting physical adversarial attacks.
- Implementing examples to demonstrate these vulnerabilities.
- Proposing strategies for mitigation.

### Case Study: Adversarial Stickers on Stop Signs

#### Problem
One of the most studied examples of physical attacks involves placing adversarial stickers on stop signs. These attacks cause models in autonomous vehicles to misclassify stop signs as other objects, such as a 45 mph speed limit sign.

#### Reference
This vulnerability was detailed in "Robust Physical-World Attacks on Machine Learning Models" by Eykholt et al. ([Eykholt et al., 2018](https://arxiv.org/pdf/1707.08945)), Section 5.2, page 7.

### Experiment: Creating a Physical Attack

#### Objective
Simulate a physical attack on an image classifier and evaluate its impact.

#### Code Example
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

# Load a pre-trained image classifier
model = models.resnet50(pretrained=True)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the stop sign image
image = Image.open("stop_sign.jpg")
input_tensor = preprocess(image).unsqueeze(0)

# Add adversarial noise (e.g., simulate sticker effect)
noise = np.random.normal(0, 0.1, input_tensor.shape)
adversarial_image = input_tensor + torch.tensor(noise, dtype=torch.float32)
adversarial_image = torch.clamp(adversarial_image, 0, 1)

# Predict original and adversarial images
original_prediction = model(input_tensor).argmax(dim=1)
adversarial_prediction = model(adversarial_image).argmax(dim=1)

print("Original Prediction:", original_prediction)
print("Adversarial Prediction:", adversarial_prediction)

# Visualize the adversarial effect
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("Adversarial Image")
plt.imshow(adversarial_image.squeeze().permute(1, 2, 0).numpy())
plt.show()
```

### Defense Strategies

#### Adversarial Training
Enhance the robustness of models by including adversarial examples in the training process.

#### Input Preprocessing
Use techniques like Gaussian blurring or JPEG compression to reduce the effectiveness of adversarial patterns.

#### Real-Time Validation
Incorporate redundancy through multi-sensor data validation to detect anomalies in image classification.

#### Example Defense Code
```python
from art.defences.preprocessor import GaussianAugmentation

defense = GaussianAugmentation(sigma=0.1, apply_fit=True, apply_predict=True)
preprocessed_image, _ = defense(input_tensor)
defended_prediction = model(preprocessed_image).argmax(dim=1)

print("Defended Prediction:", defended_prediction)
```

### Conclusion
Understanding physical attacks and implementing defenses are critical for ensuring the reliability of AI systems in real-world scenarios. Continue exploring the repository for more examples and mitigation techniques.


