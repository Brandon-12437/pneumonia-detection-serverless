 # PNEUMONIA DETECTION: SERVERLESS DEEP LEARNING API
 

  <img width="851" height="664" alt="Screenshot from 2026-01-07 17-34-55" src="https://github.com/user-attachments/assets/0ab59d49-24e8-4507-a003-a4f93f92232f" />

## 1. PROBLE DESCRIPTION
Pneumonia is a life-threatening lung infection that causes the alveoli to fill with fluid. It is a leading cause of death globally, especially where access to specialist radiologists is limited.

The Challenge: Interpreting Chest X-Rays (CXR) requires high expertise. In over-burdened medical facilities, delays in diagnosis can be fatal.

The Solution: This project provides an automated triage tool using a Deep Learning model (Xception). It analyzes digital X-rays and returns a classification in seconds via a serverless AWS Lambda API, allowing for rapid screening and second opinions in rural or high-traffic clinics.

## 2. EXPLORATORY DATA ANALYSIS (EDA)

The dataset consists of 5,232 Chest X-Ray images categorized as Normal or Pneumonia.

  ### a. Visual Identification: Pneumonia is identified by "cloudy" white opacities in the lung fields compared to the clear black appearance of healthy lungs.
  ### b. Class Imbalance: The training set has significantly more pneumonia cases. I addressed this by using Class Weights during training to ensure the model doesn't ignore "Normal" cases.
  ### c. Preprocessing: All images were resized to 150×150 and normalized to the [0,1] range.

 ## DATA VISUALIZATION
 <img width="700" height="559" alt="Screenshot from 2026-01-07 17-47-32" src="https://github.com/user-attachments/assets/80473445-40d5-4b67-942b-5e236ceafa2a" />
 
