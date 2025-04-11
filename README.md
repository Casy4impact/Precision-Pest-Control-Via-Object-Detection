# Precision-Pest-Control-Via-Object-Detection
EnviroPest Solutions, like many others in the pest control industry, faces significant inefficiencies using traditional methods. These challenges negatively affect their operational efficiency, environmental safety, and client satisfaction.
 ## Key Issues:
1.	Inefficient Inspections – Manual approaches are error-prone, time-consuming, and unsuitable for wide coverage.
2.	Suboptimal Resource Allocation – Lack of real-time pest data leads to poor targeting and waste.
3.	Environmental Impact – Excessive or misplaced pesticide application harms the ecosystem.
4.	Customer Dissatisfaction – Delayed pest identification frustrates clients and reduces retention.
________________________________________
## Project Rationale
The application of Computer Vision, specifically Convolutional Neural Networks (CNNs) and Transfer Learning, offers transformative potential:
- •	Enables automated and accurate pest detection.
- •	Facilitates real-time image processing for quicker responses.
- •	Supports targeted intervention, reducing chemical use and environmental harm.
________________________________________
## Project Aim
To develop an intelligent pest detection pipeline that:
- •	Accurately detects and classifies 12 pest types using fine-tuned deep learning models like MobileNetV2 and VGG16.
- •	Enables real-time monitoring for proactive pest control.
- •	Enhances resource allocation using data-driven insights.
________________________________________

## Data Description
### Dataset Highlights:
- •	12 pest species including: Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils.
- •	Sources: Real-world environments and public datasets.
- •	Preprocessing: All images resized to 300x300 pixels.
  
### Dataset Characteristics:
- •	Partially balanced—some classes like Earthworms and Slugs are underrepresented.
- •	Challenges: Risk of class imbalance and potential overfitting on dominant classes.
________________________________________
## Tech Stack Overview
- 🔹 Programming Language:
- •	Python – For its extensive libraries supporting AI and Computer Vision.
-
### 🔹 Core Libraries and Their Roles:

### Library,	Purpose, &	Why It Was Chosen
1. TensorFlow & Keras
- Purpose: Model training and high-level deep learning API
- Why It Was Chosen: Widely adopted, well-documented
  
2. MobileNetV2, VGG16		
- Purpose: Pre-trained models
- Why It Was Chosen: Lightweight, effective for transfer learning

3. NumPy & Pandas		
- Purpose: Numerical operations, data handling
- Why It Was Chosen: Crucial for data wrangling

4. Matplotlib, Seaborn		
- Purpose: Visualization
- Why It Was Chosen: Helps with EDA and training analysis

5. OpenCV		
- Purpose: Image operations
- Why It Was Chosen: Real-time image preprocessing

Scikit-learn	
- Purpose: Evaluation, splitting	
- Why It Was Chosen: Performance metrics and modeling utilities

6. ImageDataGenerator & Augmentation Layers		
- Purpose: Image augmentation
- Why It Was Chosen: Enhances diversity and mitigates overfitting
  
7. EarlyStopping, ModelCheckpoint		
- Purpose: Training optimization
- Why It Was Chosen: Improves efficiency, retains best models
  
8. Adam
- Purpose: Optimizer
- Why It Was Chosen: Adaptive learning rate ensures convergence

## Project Scope and Workflow
1. Data Collection
- •	Captured from varied environments: forests, farms, roadsides, beaches for model robustness.
2. Image Preprocessing
- •	Included resizing, normalization, and augmentation to prepare consistent and varied inputs.
3. Model Development
- •	Used both custom CNNs and pre-trained models (MobileNetV2, VGG16).
- •	Fine-tuning allowed the models to specialize in pest features without extensive training time.
  
### Why Fine-Tuning?
It leverages pre-learned image features (like edges and shapes) and adapts them to pest-specific patterns, reducing computational costs.
4. Model Evaluation
- •	Evaluation used metrics like accuracy
- •	Considered class imbalance in metric interpretation.
________________________________________

## Training Challenges & Bottlenecks
Observed Issues:
- •	Training halted unexpectedly due to memory overflow, overheating, and runtime crashes.
- •	A 100-epoch run finished in 2 hours but lost training history.
Workarounds:
- •	Reduced batch size and used early stopping and model checkpoints.
- •	Leveraged transfer learning to reduce training duration.
- •	Considered cloud-based training (but limited by budget).
________________________________________
## Pest Distribution Insights
- •	Snails, Bees, and Moths had higher representation.
- •	Underrepresented species like Earthworms and Slugs posed class imbalance risks.
  
- Solutions:
  - •	Applied data augmentation techniques.
  - •	Used class weights or oversampling to improve balance.

## Importance of Key Steps
### Step	    Value
- Augmentation: Tackles class imbalance and diversifies training data
- Train/Validation/Test Split:	Ensures unbiased performance evaluation
- Early Stopping:	Prevents overfitting and reduces computational cost
- Transfer Learning:	Boosts accuracy with smaller datasets
- Model Checkpoints:	Preserves optimal weights amidst instability
- Training Monitoring:	Informs better hyperparameter tuning

## Training Progress Summary
### Metric	          Epoch 1  	Epoch 36
- Training Accuracy  	9.48%	    65.65%
- Validation Accuracy	20.71%	  77.70%
- Training Loss	      6.3165	  1.1188
- Validation Loss	    2.3639	  0.7771

## Key Takeaways:
- •	Both accuracy and loss metrics showed steady improvement.
- •	Validation accuracy outperformed training due to:
- o	Training-specific dropout layers.
- o	More complex augmented images during training phase.
________________________________________

 ## Final Evaluation
 ### Performance on Test Set:
- •	Accuracy: 83.26%
- •	Loss: 0.6014
### Interpretation:
- •	Strong generalization to unseen data.
- •	Low loss reflects effective model learning and minimal overfitting

## Recommendations
  Task	                        Reason
- Adopt Cloud-Based Training: 	Greater stability and scalability
- Use More Lightweight Models:	Like EfficientNet or SqueezeNet
- Continue Dataset Expansion:  	Especially for minority classes
- Optimize Batch Size and Learning Rate:	Reduce training crashes and overfitting
- Maintain Checkpointing:      	Prevents data loss from system failures
- Tune Dropout Rate:           	Potential for better training performance
________________________________________

 ## Conclusion
This project is a vital leap toward precision agriculture and sustainable pest control. By integrating computer vision and deep learning, it enables:
- •	Early pest detection
- •	Eco-friendly interventions
- •	Efficient resource usage



