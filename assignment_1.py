import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set the output directory to the script's directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Function to save plots
def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)

# Load the data
df = pd.read_csv('CCPP_data.csv')


print("\n### Predicting Electrical Energy Output of a Combined Cycle Power Plant")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check data types
print("\nData types:\n", df.dtypes)

# Check ranges for each column
print("\nRanges for each column min - max:")
for column in df.columns:
    print(f"{column}: {df[column].min()} to {df[column].max()}")

# Convert all columns to float
df = df.astype(float)

# Verify data types after conversion
print("\nData types after conversion:\n", df.dtypes)

# Calculate basic statistics
print("\nBasic statistics:\n", df.describe())

# Check correlations
print("\nCorrelations:\n", df.corr())

# Create a 2x2 grid of scatter plots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Scatter Plots of Variables vs Power Output (PE)', fontsize=16)

# Temperature vs PE
axs[0, 0].scatter(df['AT'], df['PE'])
axs[0, 0].set_xlabel('Temperature')
axs[0, 0].set_ylabel('PE')
axs[0, 0].set_title('Temperature vs PE')

# Ambient Pressure vs PE
axs[0, 1].scatter(df['AP'], df['PE'])
axs[0, 1].set_xlabel('Ambient Pressure')
axs[0, 1].set_ylabel('PE')
axs[0, 1].set_title('Ambient Pressure vs PE')

# Relative Humidity vs PE
axs[1, 0].scatter(df['RH'], df['PE'])
axs[1, 0].set_xlabel('Relative Humidity')
axs[1, 0].set_ylabel('PE')
axs[1, 0].set_title('Relative Humidity vs PE')

# Exhaust Vacuum vs PE
axs[1, 1].scatter(df['V'], df['PE'])
axs[1, 1].set_xlabel('Exhaust Vacuum')
axs[1, 1].set_ylabel('PE')
axs[1, 1].set_title('Exhaust Vacuum vs PE')

plt.tight_layout()
save_plot(fig, 'scatter_plots.png')

# Create a correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title('Correlation Heatmap')
save_plot(fig, 'correlation_heatmap.png')

# Separate features (X) and target variable (y)
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores_linear = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores_linear = np.sqrt(-cv_scores_linear)
rmse_scores_rf = np.sqrt(-cv_scores_rf)

print("\nResults:")
print("Linear Regression - Cross-validation RMSE scores:", rmse_scores_linear)
print("Linear Regression - Average RMSE:", rmse_scores_linear.mean())

print("\nRandom Forest - Cross-validation RMSE scores:", rmse_scores_rf)
print("Random Forest - Average RMSE:", rmse_scores_rf.mean())

# Train the models on the entire training set
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# For Linear Regression
r2_linear = linear_model.score(X_test, y_test)

# For Random Forest
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Calculate RMSE on the test set
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\nR-squared Results:")
print("Linear Regression R-squared:", r2_linear)
print("Random Forest R-squared:", r2_rf)

print("\nTest Set Results:")
print("Linear Regression - Test RMSE:", rmse_linear)
print("Random Forest - Test RMSE:", rmse_rf)

# Function to create scatter plot of predicted vs actual values
def plot_predictions(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    save_plot(fig, filename)

# Function to plot residuals
def plot_residuals(y_true, y_pred, title, filename):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.plot([y_pred.min(), y_pred.max()], [0, 0], 'r--', lw=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    save_plot(fig, filename)

# Generate predictions
lr_predictions = cross_val_predict(linear_model, X_train, y_train, cv=5)
rf_predictions = cross_val_predict(rf_model, X_train, y_train, cv=5)

# Create plots
plot_predictions(y_train, lr_predictions, 'Linear Regression: Predicted vs Actual', 'lr_predicted_vs_actual.png')
plot_predictions(y_train, rf_predictions, 'Random Forest: Predicted vs Actual', 'rf_predicted_vs_actual.png')
plot_residuals(y_train, lr_predictions, 'Linear Regression: Residuals', 'lr_residuals.png')
plot_residuals(y_train, rf_predictions, 'Random Forest: Residuals', 'rf_residuals.png')

# Plot feature importance for Random Forest
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(X_train.columns, rf_model.feature_importances_)
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Random Forest: Feature Importance')
save_plot(fig, 'rf_feature_importance.png')

# Plot RMSE comparison
models = ['Linear Regression', 'Random Forest']
rmse_scores = [
    np.sqrt(mean_squared_error(y_test, linear_model.predict(X_test))),
    np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(models, rmse_scores)
ax.set_ylabel('RMSE')
ax.set_title('Model Comparison: RMSE on Test Set')
save_plot(fig, 'model_comparison_rmse.png')

print("\nAll plots have been saved in the script's directory.")
