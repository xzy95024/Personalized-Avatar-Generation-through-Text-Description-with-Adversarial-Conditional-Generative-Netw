## Personalized Avatar Generation from Text Descriptions using ACGAN

### Overview
This project presents a method to generate personalized avatar images from textual descriptions using an Adversarial Conditional Generative Network (ACGAN). This approach leverages deep learning to convert text descriptions into visually detailed and realistic avatar images, facilitating enhanced user experiences in gaming, virtual environments, social media, and digital content creation.

### Project Description
The project employs the Auxiliary Classifier Generative Adversarial Network (ACGAN) architecture. It translates textual inputs specifying avatar attributes (such as hair color, eye color, accessories, etc.) into visually realistic avatar images. This model significantly streamlines avatar creation and customization in various applications.

### Features
- **Deep Learning-Based**: Uses advanced deep learning techniques to achieve high-quality avatar generation.
- **Text-to-Image Translation**: Converts detailed textual descriptions into corresponding avatar images.
- **Dataset**: Utilizes a high-quality, diverse dataset sourced from Getchu, containing a variety of illustrative styles.

### Model Architecture
- **Generator**: Adapted from the SRResNet architecture, featuring 16 residual blocks and 3 sub-pixel convolutional layers to upscale feature maps.
- **Discriminator**: Includes an attribute classifier extension that evaluates the authenticity of generated images while ensuring conditional accuracy.

### Dataset
The training dataset comprises anime-style images sourced from Getchu and annotated using the Illustration2Vec tool to estimate attributes from textual descriptions. The dataset includes consistent quality images, essential for reliable avatar generation.

### Requirements
- Python
- PyTorch / TensorFlow
- OpenCV
- lbpcascade_animeface
- Illustration2Vec

### Results
The initial experiments indicate promising results in aligning generated avatars closely with input descriptions. The model effectively handles simpler attributes such as colors, though performance with complex attributes (e.g., hats or glasses) requires further refinement.

### Dataset and Training
The model was trained on approximately 60,000 images from Getchu, augmented with tags estimated via Illustration2Vec. Optimal performance requires extensive training (recommended: 40,000–60,000 epochs).

### Evaluation
Qualitative and quantitative analyses demonstrate that the proposed method effectively generates personalized avatar images closely aligned with input descriptions, although further improvements are anticipated with prolonged training and dataset enrichment.

### Acknowledgments
Special thanks to the Northeastern University 5100 teaching staff for their invaluable guidance and support throughout this project.

### References
- Martin Arjovsky, Léon Bottou. "Towards principled methods for training generative adversarial networks."
- Martin Arjovsky, Soumith Chintala, Léon Bottou. "Wasserstein GAN."
- Jiakai Zhang, Minjun Li, Yingtao Tian, Huachun Zhu, Zhihao Fang. "Towards Automatic Anime Characters Creation with GANs."

