
Origin: @w3bwizart

---
### References
Course: Duke University https://www.coursera.org/learn/machine-learning-foundations-for-product-managers

A video with the results can be found at https://youtu.be/liNrFCxHfJ0

# Machine Learning - Predicting Electrical Energy Output of a Combined Cycle Power Plant

This is an assignment of the first course.

This is a prediction script that runs multiple algorithms on a data set.
The data set is in the repo and when you run the script with 
``` python assignment_1.py ``` 

- It will clean and inspect data
- do some data splitting
- run models
- creates plots which will be stored in the root directory
- the script will output all the result in the command-line


![CMD output](/image.png "CMD output")


# Results & Conclusion
## 1. Problem Definition and Approach

### Problem Type:
This is a regression problem aimed at predicting the continuous numerical value of Net hourly electrical energy output (PE) in MW.

### Evaluation Metric:
Root Mean Square Error (RMSE) is chosen as the primary metric because:
- It's interpretable in the same units as the target variable (MW)
- It penalizes larger errors more heavily, which is important for power output prediction

### Data Exploration:
- Dataset: 9568 hourly average ambient environmental readings
- No missing values found
- All variables converted to float type
- All values within specified ranges

## 2. Feature Selection and Algorithm Consideration

### Feature Analysis:
Based on correlation analysis and scatter plots:
1. Temperature (AT): Strong negative correlation (-0.948128) with PE
2. Exhaust Vacuum (V): Strong positive correlation (0.869780) with PE
3. Ambient Pressure (AP): Moderate positive correlation (0.518429) with PE
4. Relative Humidity (RH): Moderate positive correlation (0.389794) with PE

### Algorithms Considered:
1. Multiple Linear Regression
2. Random Forest Regression

Rationale: Comparing a simple linear model with a more complex ensemble method to capture potential non-linear relationships.

## 3. Model Development and Validation Strategy

### Data Splitting:
- Training set: 7654 samples (80%)
- Testing set: 1914 samples (20%)

### Validation Strategy:
5-fold cross-validation to ensure robust performance estimation

### Model Implementation and Comparison:
Both Linear Regression and Random Forest models were implemented and compared using:
1. Cross-validation RMSE scores
2. Test set RMSE
3. Predicted vs Actual plots
4. Residual plots

## 4. Results and Interpretation

### Model Performance:
Random Forest outperformed Linear Regression:
- Lower average RMSE in cross-validation (3.46 MW vs 4.57 MW)
- Lower RMSE on test set (3.25 MW vs 4.50 MW)
- Tighter clustering in Predicted vs Actual plot

### Feature Importance (Random Forest):
1. Temperature (AT): Highest importance
2. Exhaust Vacuum (V): Second highest
3. Ambient Pressure (AP) and Relative Humidity (RH): Lower importance

## 5. Results Based on our analysis: 

1. Model Performance: 
	- Random Forest outperformed Linear Regression: 
		- Lower average RMSE in cross-validation (3.46 MW vs 4.57 MW) 
		- Lower RMSE on test set (3.25 MW vs 4.50 MW) 
	- R-squared values: 
		- Linear Regression R-squared: 0.9301046431962188
		- Random Forest R-squared: 0.9636777774021675 
2. Feature Importance (Random Forest): 
	1. Temperature (AT): Highest importance 
	2. Exhaust Vacuum (V): Second highest 
	3. Ambient Pressure (AP) and Relative Humidity (RH): Lower importance 
3. Residual Analysis: 
	- Linear Regression: Residuals show a curved pattern, indicating unaccounted non-linear relationships. 
	- Random Forest: Residuals are more randomly scattered, suggesting better capture of data relationships. 

## 6. Conclusion 
The Random Forest model is superior for predicting the electrical energy output of the Combined Cycle Power Plant. Its ability to capture non-linear relationships between variables results in more accurate predictions compared to Linear Regression. 

### Key findings: 
1. Temperature and Exhaust Vacuum are the most critical factors influencing power output. 
2. The model's average prediction error is approximately 3.25 MW, about 0.7% of the average power output. 
3. Non-linear relationships in the data explain why Random Forest outperforms Linear Regression. 
 
### Practical implications: 
- Power plant operators should focus on monitoring and controlling Temperature and Exhaust Vacuum to optimize energy output. 

### Limitations and future work: 
1. Model performance in extreme conditions not represented in the dataset is uncertain. 
2. Future improvements could include: 
	- Collecting more diverse data 
	- Incorporating additional relevant features 
	- Exploring advanced algorithms (e.g., gradient boosting methods) 
	- Investigating feature interactions 

This project demonstrates the effectiveness of machine learning in predicting power plant output, potentially leading to improved efficiency and management of Combined Cycle Power Plants.
