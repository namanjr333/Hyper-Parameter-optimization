# Hyper-Parameter-optimization
used many optimization techniques to get the best hyper-parameters 
NAME :- NAMAN SAINI
ENROLLMENT NO.- 21112074
Automated hyperparameter optimisation:


➢ Introduction
1.	Fine-tuning machine learning models is significantly enhanced by hyperparameter optimization.
2.	Hyperparameters are adjustable settings that control the model’s learning from data.
3.	These settings are fixed before training starts, unlike model parameters which are learned during training.
4.	Skilful hyperparameter tuning can greatly boost a model’s performance.
5.	The Bayesian Optimization method for hyperparameter refinement is the focus of this document.
6.	Additionally, the Tree-structured Parzen Estimator (TPE) method has also been utilized for hyperparameter optimization.
7.	A comparison has been made between Hyper opt, and Bayesian optimization techniques, including an analysis of their learning rates.
➢ Hyperparameters
1.	Hyperparameters are configuration settings for a machine learning algorithm.
2.	They are set before the training process begins.
3.	Hyperparameters guide the training algorithm.
4.	They significantly impact the model’s performance.
5.	Examples include learning rate, number of trees in a random forest, and number of layers in a neural network.
  I have used Random Forest Classifier as my base model over which I have applied many different HPO’s such as Bayesian Optimization, (TPE) Tree-Parzen Estimator and lastly Hyper-opt to compare above techniques with in built libraries


➢ Why  Random Forest Classifier is used as base model
•	Supervised Learning Algorithm: The Random Forest, also known as a Random Decision Forest, is a supervised machine learning algorithm that leverages multiple decision trees for tasks like classification and regression.
•	Versatile and Scalable: It is particularly effective for handling large and complex datasets, making it suitable for high-dimensional feature spaces.
•	Feature Importance Insights: This algorithm provides valuable insights into the significance of different features in the dataset.
•	High Predictive Accuracy: Random Forests are renowned for their ability to deliver high predictive accuracy while minimizing the risk of overfitting.
•	Broad Applicability: Its robustness and reliability make it a popular choice in various domains, including finance, healthcare, and image analysis.


➢ Key Hyperparameters for Optimization in Random Forest Classifier:
•	n_estimators:
o	Controls the number of decision trees in the forest.
o	A higher number of trees generally improves model accuracy but increases computational complexity.
o	Finding the optimal number of trees is crucial for balancing performance and training time.
•	max_depth:
o	Sets the maximum depth for each tree in the forest.
o	Crucial for enhancing model accuracy; deeper trees capture more complexity.
o	However, excessively deep trees can lead to overfitting, so setting an appropriate depth is vital to maintain generalization.
•	max_features:
o	Determines the number of features to consider when looking for the best split at each node.
o	Balancing this helps the model to avoid overfitting and improves its performance by ensuring diverse decision boundaries.
•	criterion:
o	Defines the function used to measure the quality of a split.
o	The choices are "gini" for Gini impurity and "entropy" for information gain.
o	Selecting the appropriate criterion can influence the effectiveness of the splits and overall model performance.

➢ Bayesian Optimization:
•	Purpose:
o	An iterative method to minimize or maximize an objective function, especially useful when evaluations are expensive.
•	Initialization:
o	Start with a small, randomly selected set of hyperparameter values.
o	Evaluate the objective function at these initial points to establish a starting dataset.
•	Surrogate Model:
o	Construct a probabilistic model, typically a Gaussian Process, based on the initial evaluations.
o	This model serves as an approximation of the objective function, providing estimates and uncertainty measures.
•	Acquisition Function:
o	Use the surrogate model to decide the next set of hyperparameters.
o	Optimize an acquisition function to balance exploring new areas and exploiting known promising regions.
•	Evaluation:
o	Assess the objective function with the hyperparameters chosen by the acquisition function.
o	This involves running the model and recording the performance metrics for these hyperparameters.
•	Update:
o	Integrate the new evaluation data into the surrogate model.
o	Refine the model’s approximation of the objective function with the updated information.
•	Iteration:
o	Repeat the steps of modelling, acquisition, and evaluation iteratively.
o	Continue the process until a stopping criterion, like a set number of iterations or a target performance level, is reached.

➢ Implementation
•	Step 1: Define the Objective Function:
o	Our goal for optimization is to minimize the negative mean accuracy of a Random Forest Classifier.
o	This means our objective function will measure and return the negative of the mean accuracy to align with the minimization process. Below is a code snippet illustrating the objective function

 
Step 2: Define the Hyperparameter Space:
•	We need to outline the range and possible values for the hyperparameters we want to optimize.
•	The following code snippet demonstrates the search space for various hyperparameters that will be used in the optimization process.
•	 
Step 3: Execute the Optimization Algorithm:
•	Use the optimization algorithm to search for the best possible hyperparameters within the defined search space.
•	The following code snippet illustrates how to run the optimization algorithm to identify the optimal hyperparameters.
•	 

Step 4: Evaluate the Results:
o	Once optimization is complete, assess the performance of the best-found model.
o	This involves calculating metrics like ROC-AUC scores and conducting cross-validation to ensure robust evaluation.
•	 

➢ Tree-structured Parzen Estimator (TPE) Optimization:
• Purpose:
•	TPE optimizes an objective function iteratively, aiming to maximize or minimize it efficiently, especially beneficial when function evaluations are costly.
• Initialization:
•	Initialize empty lists params and results to store sampled hyperparameters and their corresponding objective function scores.
• Iterations:
•	For n_calls iterations:
o	Sample hyperparameters (next_params) from the defined space using random choice.
o	Evaluate the objective function (objective_function) with next_params to obtain a score (score).
o	Store next_params and score in params and results, respectively.
• Best Hyperparameters:
•	Identify the index (best_index) of the highest score (np.argmax(results)), indicating the best-performing hyperparameters.
•	Retrieve and return the best hyperparameters (best_params) based on best_index.
• Output:
•	Print and return the best hyperparameters (best_params) found by the optimization process.

Below code snippet can be used to get the best parameters values:-
	For model creation and defining objective function  
	For hyper-parameter tuning  
This iterative approach efficiently explores the hyperparameter space, leveraging a surrogate model to guide the search towards optimal configurations, suitable for enhancing the performance of machine learning models and similar complex systems.
## Also used hyper-opt library and random forest classifier with default parameters to compare above techniques

