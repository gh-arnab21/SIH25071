### 📊 Rockfall Prediction Ensemble Classifier

This folder contains the core implementation of our ensemble classifier, a sophisticated machine learning model designed to predict rockfall risk. By combining multiple predictive models, we achieve a more robust and accurate system capable of providing not just a risk level, but also confidence and uncertainty scores.

---

### 🧠 Model Architecture: An Ensemble Approach

Our system's strength lies in its **ensemble learning** approach. Instead of relying on a single model, we aggregate the predictions of three diverse classifiers. This strategy minimizes the weaknesses of individual models and capitalizes on their collective strengths, leading to enhanced predictive performance and reliability.

The ensemble workflow can be broken down into these key steps:

1.  **Individual Training**: Each of the three base models is trained independently on the same dataset.
2.  **Prediction**: The trained models all make a prediction on new, unseen data.
3.  **Aggregation**: The predictions from all three models are combined using a voting mechanism to determine the final output.
4.  **Confidence & Uncertainty**: Beyond a single prediction, we use the agreement or disagreement among the models to calculate a **confidence score** and an **uncertainty estimate**.

---

### 💻 Our Core Classifier Models

The diversity of our ensemble is a result of the unique characteristics of each base model:

#### 1. Random Forest Classifier

This model is an **ensemble of decision trees**. During training, it creates numerous trees and outputs the class that is the mode of the classes predicted by individual trees. We chose it for its:

* **Robustness**: It handles non-linear relationships and is less sensitive to outliers.
* **Interpretability**: It can provide a clear measure of **feature importance**, helping us understand which factors, like slope or rainfall, are most influential in predicting rockfall risk.

#### 2. XGBoost Classifier

**XGBoost** (Extreme Gradient Boosting) is a highly efficient and powerful machine learning algorithm. It is part of the boosting family, where models are built sequentially to correct the mistakes of previous models. Its benefits include:

* **High Performance**: It is known for its speed and accuracy, often winning machine learning competitions on structured data.
* **Scalability**: It is optimized to handle large datasets effectively.

#### 3. Neural Network Classifier

Our neural network is a **Multi-Layer Perceptron (MLP)**, a type of feed-forward neural network. It's designed to learn complex, non-linear patterns that other models might miss.

The MLP is built using the **Keras** high-level API, which runs on a **TensorFlow** backend.

* **Keras**: This library provides a user-friendly and modular interface for building deep learning models. It simplifies the process of defining layers and configuring the network, allowing us to focus on the model's architecture rather than low-level implementation details.
* **TensorFlow**: This is the underlying open-source machine learning framework that performs all the heavy computational work. It's a robust and flexible platform, especially well-suited for parallel processing on GPUs, which accelerates the training of our neural network.

By incorporating this MLP into our ensemble, we ensure the system can capture a wider range of predictive signals, from simple feature correlations to highly intricate data patterns.