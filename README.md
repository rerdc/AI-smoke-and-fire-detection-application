# AI-smoke-and-fire-detection-application

Overview
This application integrates two previously trained AI models for smoke detection and fire classification to estimate the severity of potential forest wildfires.

The application processes a video source and performs real-time inference using both models. Their predictions are then combined to determine an associated wildfire danger level.

How It Works
The application takes a video source as input and processes its frames through two AI models:

Smoke Detection Model: Detects smoke in the video frames and generates bounding boxes around detected smoke regions.
Fire Classification Model: Analyzes the video frames to estimate the probability of fire.
The outputs of both models are combined using predefined threshold values to determine the overall danger level. This approach allows the application to classify the potential severity of a wildfire based on the relationship between the smoke detection and fire classification probabilities.

Application Features
The application displays the following information during inference:

Smoke Detection: Bounding boxes identifying detected smoke regions.
Model Predictions: The inference probabilities produced by both AI models.
Wildfire Danger Level: A severity classification determined by combining the two model outputs and applying predefined thresholds.
Performance Monitoring: The application's execution speed, displayed in frames per second (FPS).
AI Models
The two models were previously trained for their respective tasks. Their weights and training notebooks are available on Kaggle:

Smoke Detection Model: https://www.kaggle.com/code/rubcost/notebookyolov26-aug
Fire Classification Model: https://www.kaggle.com/code/rubcost/notebooka7da5cebb6
