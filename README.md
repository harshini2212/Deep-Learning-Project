# Deep-Learning-Project
 Modified ResNet for CIFAR-10: Achieving 94.74% training accuracy with 2.8M Parameters
 
 # Modified ResNet for CIFAR-10
This project implements a **modified ResNet architecture** optimized for the CIFAR-10 dataset, achieving **94.74% test accuracy** with only **2.8 million parameters** (56% of the 5M parameter limit). The model is designed to balance accuracy and computational efficiency, making it suitable for resource-constrained environments.

---

## **Key Features**
- **Progressive Channel Scaling**: Doubling feature channels at each downsampling stage (16→32→64) to balance accuracy and efficiency.
- **Lightweight Residual Blocks**: Basic blocks with two 3×3 convolutions reduce parameter inflation while maintaining performance.
- **Data Augmentation**: Random cropping, horizontal flipping, and normalization improve generalization.
- **Training Strategy**: ADAM optimizer with learning rate scheduling over 300 epochs ensures stable convergence.
- **Parameter Efficiency**: Achieves state-of-the-art performance using only 56% of the 5M parameter budget.

---

## **Results**
- **Test Accuracy**: 94.74%
- **Parameter Count**: 2,797,610 (56% of 5M limit)
- **Training Duration**: 300 epochs

### **Training Curves**
![training_curves_1](https://github.com/user-attachments/assets/ef20defc-2fbd-4657-99e6-c69fc24c3c19)
![training_curves_2](https://github.com/user-attachments/assets/51bdaf66-73df-4fe8-b576-20035b16fd5b)


### **Confusion Matrix**
![confusion_matrix](https://github.com/user-attachments/assets/bef6defa-0d4e-490b-b39f-e0c799dd9c7f)




---

## **Comparison with DenseNet**
We compared our modified ResNet with a DenseNet model trained under similar constraints. Key findings:
- **Accuracy**: ResNet outperformed DenseNet by **1.22%** (94.74% vs. 93.52%).
- **Parameter Efficiency**: ResNet used **9.7% fewer parameters** than DenseNet.
- **Generalization**: ResNet showed better generalization, with lower validation loss and higher test accuracy.

---

## **Repository Structure**
