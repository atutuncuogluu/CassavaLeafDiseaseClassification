

# Project Report

**Author**: Ahmet Tütüncüoğlu  


## Step 1 - Problem Approach and Dataset Examination

- The dataset has 21,397 entries, but only 17,938 are labeled.
- The dataset is accessed from Kaggle.
- Class distribution in the dataset is unbalanced.
- Visualizations were created to show class distribution and sample images from the dataset.



## Step 2 - Literature Review

- Various preprocessing techniques like Cutmix, Cutout, and Mixup were studied to improve model accuracy.
- In the project, Cutout and Cutmix techniques were used.
- Data augmentation was performed using the Torchvision library with methods like random flip, random rotation, Gaussian blur, color jitter, random resized crop, and normalization based on ImageNet dataset values.



## Step 3 - Preparing Functions and Files for Training

- Instead of using paths directly in a dataframe, separate folders for each class were created for train and test sets.
- The `Shutil` library was used to organize the data.
- The `Dataset` class from `torch.utils` was inherited to create a custom `ImageFolderCustom` class.
- This setup allowed batching, improving speed, memory efficiency, and parallel processing.

## Step 4 - Researching Model Architectures

### GoogleNet
- The original paper was found and the architecture was implemented accordingly.



### AlexNet
- The architecture was implemented as per course materials.



### VGGNet
- Several versions were considered, and the chosen architecture was based on course materials.

### ResNet
- The original paper was consulted.
- Basic block and bottleneck block were used to create ResNet34 and ResNet152 variations.



## Step 5 - Creating Model, Training, and Testing Functions

- Each model architecture was created using the `torch.nn` module.
- Additional techniques like batch normalization and dropout were used.
- The Adam optimizer and ReduceLROnPlateau scheduler were employed for optimization and learning rate adjustment.
- `train_step` and `test_step` functions were developed to handle training and testing, respectively.
- Data augmentation techniques like cutout or cutmix were applied with a 0.5 probability in `train_step`.
- Each function performs standard steps like obtaining raw predictions, calculating loss using CrossEntropyLoss, resetting optimizer gradients, applying backpropagation, and updating the optimizer.

## Step 6 - Models and Average Metrics Table

| Model       | Epoch | Loss | Accuracy | Precision | Recall | F1-Score |
|-------------|-------|------|----------|-----------|--------|----------|
| VGGNet      | 35    | 0.71 | 0.74     | 0.61      | 0.58   | 0.57     |
| AlexNet     | 25    | 0.96 | 0.66     | 0.44      | 0.35   | 0.35     |
| GoogleNet   | 40    | 0.81 | 0.70     | 0.56      | 0.51   | 0.51     |
| ResNet34    | 40    | 0.74 | 0.73     | 0.60      | 0.53   | 0.54     |
| ResNet152   | 40    | 0.85 | 0.69     | 0.52      | 0.45   | 0.46     |

---

