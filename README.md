# IPL-Match-winner-Prediction
This project uses historical Indian Premier League (IPL) match data to predict the winner of a cricket match using machine learning models. The pipeline includes data preprocessing, model training and evaluation, and winner prediction.

# Project Structure

The project is divided into three key Jupyter Notebooks:
	1.	1_Per_Processing_dataset_(IPL_match_winner).ipynb
	•	Cleans and prepares the raw IPL dataset
	•	Handles missing values, encodes categorical variables, and performs feature engineering
	2.	2_Testing_the_Models_(IPL_match_winner).ipynb
	•	Trains multiple machine learning models (e.g., Logistic Regression, Random Forest, etc.)
	•	Evaluates model performance using metrics like accuracy, precision, recall, and confusion matrix
	3.	3_Predictions_(_IPL_Match_Winner).ipynb
	•	Uses the best-performing model to predict the match winner based on input features
	•	Accepts team names, venue, toss result, etc., and returns the predicted winner

# Requirements
	•	Python 3.x
	•	Jupyter Notebook
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib / seaborn (for visualizations)

You can install dependencies using:

pip install pandas numpy scikit-learn matplotlib seaborn

# How to Use
	1.	Start with 1_Per_Processing_dataset_(IPL_match_winner).ipynb to preprocess the data.
	2.	Run 2_Testing_the_Models_(IPL_match_winner).ipynb to train and evaluate models.
	3.	Use 3_Predictions_(_IPL_Match_Winner).ipynb to make predictions with the trained model.

# Output
	•	Preprocessed dataset ready for model training
	•	Trained machine learning models with evaluation metrics
	•	Predicted winner for given match conditions

# License

This project is licensed under the MIT License.
