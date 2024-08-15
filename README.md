# Credit Risk Assessment using Graph Neural Networks (GNNs) and K-Nearest Neighbors (KNN)

Introduction
This project aims to develop a sophisticated credit risk assessment model using Graph Neural Networks (GNNs) to predict loan default probabilities. The model leverages unsupervised deep learning techniques alongside K-Nearest Neighbors (KNN) to model relationships between loan applicants, ultimately helping financial institutions make more informed lending decisions.

### Project Overview
The goal of this project is to accurately assess the credit risk of loan applicants by analyzing various features, such as demographic information, financial history, and loan details. This is achieved by constructing a similarity graph of applicants and applying GNNs to learn meaningful embeddings, which are then used to predict the likelihood of loan default.

### Features
Data Preprocessing: Includes data cleaning, standard scaling, and one-hot encoding of categorical features.
Graph Construction: Builds a K-Nearest Neighbors (KNN) graph to represent relationships between applicants.
Model Training: Utilizes Graph Neural Networks (GNNs) for learning embeddings that predict loan default probabilities.
Clustering: Applies K-Means clustering on the learned embeddings to categorize applicants into different risk levels.
Visualization: Uses PCA and t-SNE for dimensionality reduction and visualization of the clustering results.

### Technology Stack
Programming Language: Python
Libraries and Frameworks:
PyTorch
PyTorch Geometric
Scikit-Learn
NetworkX
Pandas
Numpy
Matplotlib
Seaborn

### Tools:
Jupyter Notebooks
Google Colab (optional)
Docker (for containerization, optional)

### Data
The dataset used in this project consists of 1000+ entries of loan applicants, including features such as age, job, housing status, credit amount, and duration of the loan.
Data preprocessing steps include standard scaling of numerical features and one-hot encoding of categorical features.
The similarity graph is constructed using K-Nearest Neighbors (KNN) based on the processed features.
Installation
To set up the project on your local machine, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/credit-risk-assessment.git
cd credit-risk-assessment
Create and activate a virtual environment (optional but recommended):

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook or Python scripts:

bash
Copy code
jupyter notebook

### Usage
Data Preprocessing: Run the data preprocessing steps to clean and prepare the dataset.
Graph Construction: Create the K-Nearest Neighbors graph using the provided script.
Model Training: Train the Graph Neural Network (GNN) model to learn embeddings for credit risk assessment.
Clustering and Visualization: Apply K-Means clustering to the embeddings and visualize the results using PCA or t-SNE.

### Model Architecture
The model architecture is based on a two-layer Graph Convolutional Network (GCN):

Layer 1: GCN layer with ReLU activation to capture local graph structure.
Layer 2: GCN layer to refine embeddings and produce the final node representations.
Clustering: K-Means clustering is applied to the final embeddings to categorize applicants into different risk groups.

### Results
Model Performance: The GNN model successfully learned embeddings that could be used to predict loan default probabilities.
Visualization: The PCA and t-SNE visualizations effectively highlighted the clustering of applicants based on their risk levels.
Conclusion: The project demonstrates the feasibility and effectiveness of using GNNs combined with KNN for credit risk assessment.
