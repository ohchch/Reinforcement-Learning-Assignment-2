import pandas as pd # Import pandas library for data manipulation
import numpy as np # Import numpy library for numerical operations
from sklearn.model_selection import train_test_split # Import function for splitting data from scikit-learn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # Import classes for feature scaling and label encoding from scikit-learn
from tensorflow import keras as tf # Import TensorFlow's Keras API and alias it as tf
import matplotlib.pyplot as plt # Import matplotlib library for plotting
import seaborn as sns # Import seaborn library for statistical data visualization

# Task 1: Data Loading and Inspection
print("--- Task 1: Data Loading and Inspection ---")
try:
    # Attempt to load the dataset from the specified path
    df = pd.read_csv('/workspace/data/housing.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # If the file is not found, print an error message and exit
    print("Error: 'Housing.csv' not found. Please download the dataset and place it in the correct directory.")
    exit()

print("\nInitial Data Overview (First 5 rows):")
# Display the first 5 rows of the dataset
print(df.head())

print("\nDataset Structure and Info:")
# Display the structure and information of the dataset, including column names, non-null counts, and data types
df.info()

print("\nSummary Statistics:")
# Display descriptive statistics of the dataset, such as mean, standard deviation, etc.
print(df.describe())
print("-" * 50)

# Task 2: Data Preprocessing
print("\n--- Task 2: Data Preprocessing ---")

# 2.1 Handling Missing Values
# Check the number of missing values per column
print("\nMissing values before handling:")
print(df.isnull().sum())

# Iterate through all columns to handle missing values
for column in df.columns:
    # If there are missing values in the column
    if df[column].isnull().any():
        # If it's a categorical data type, fill missing values with the mode
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column].dtype):
            df[column].fillna(df[column].mode()[0], inplace=True)
            print(f"Filled missing values in categorical column '{column}' with mode.")
        # If it's a numerical data type, fill missing values with the mean
        elif pd.api.types.is_numeric_dtype(df[column].dtype):
            df[column].fillna(df[column].mean(), inplace=True)
            print(f"Filled missing values in numerical column '{column}' with mean.")

# Print the number of missing values after handling (should be 0)
print("\nMissing values after handling (should be 0):")
print(df.isnull().sum())


# 2.2 Feature Scaling (Using MinMaxScaler to scale numerical features to a 0-1 range)
# Define numerical columns to be scaled
numerical_cols_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Scale the selected numerical columns
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])
# Print message after applying scaling and show the first 5 rows of data
print(f"\nApplied MinMaxScaler to columns: {numerical_cols_to_scale}")
print(df[numerical_cols_to_scale].head())

# 2.3 Label Encoding (Using LabelEncoder to convert categorical features to numerical values)
# Automatically identify and process all 'object' type columns (i.e., string-type categorical features)
for col in df.select_dtypes(include=['object']).columns:
    # Ensure not to mistakenly process the target variable 'price' (if it happens to be an object type)
    if col != 'price':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Applied LabelEncoder to categorical column: {col}")
    else:
        print(f"Warning: Skipping label encoding for 'price' column as it's the target variable.")

print("\nData after preprocessing (First 5 rows):")
print(df.head())
print("-" * 50)

# Task 3: Exploratory Data Analysis (EDA)
print("\n--- Task 3: Exploratory Data Analysis (EDA) ---\n")

# 3.1 Generate summary statistics again (after preprocessing)
print("\nSummary Statistics (after preprocessing):")
print(df.describe())

# 3.2 Visualize important features
# Plot house price distribution
plt.figure(figsize=(10, 6)) # Set figure size
sns.histplot(df['price'], kde=True) # Use histplot to draw price distribution with KDE curve
plt.title('Distribution of House Prices') # Set plot title
plt.xlabel('Price') # Set X-axis label
plt.ylabel('Frequency') # Set Y-axis label
plt.savefig('eda_price_distribution.png') # Save plot as PNG file
plt.show() # Display plot

# Plot scatter plot of Area vs. Price
plt.figure(figsize=(10, 6)) # Set figure size
sns.scatterplot(x='area', y='price', data=df) # Use scatterplot to draw scatter plot
plt.title('Area vs. Price') # Set plot title
plt.xlabel('Area (scaled)') # Set X-axis label
plt.ylabel('Price') # Set Y-axis label
plt.savefig('eda_area_vs_price.png') # Save plot as PNG file
plt.show() # Display plot

# Plot box plot to show the effect of 'Bedrooms' on 'Price'
plt.figure(figsize=(10, 6)) # Set figure size
sns.boxplot(x='bedrooms', y='price', data=df) # Use boxplot to draw box plot (bedrooms count has been scaled, but trends can still be observed)
plt.title('Bedrooms vs. Price') # Set plot title
plt.xlabel('Bedrooms (scaled)') # Set X-axis label
plt.ylabel('Price') # Set Y-axis label
plt.savefig('eda_bedrooms_vs_price.png') # Save plot as PNG file
plt.show() # Display plot

# Plot correlation heatmap of features
plt.figure(figsize=(12, 10)) # Set figure size
correlation_matrix = df.corr() # Calculate the correlation matrix of all features in the DataFrame
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Use heatmap to draw heatmap, display correlation values and use coolwarm colormap
plt.title('Correlation Matrix of Features') # Set plot title
plt.savefig('eda_correlation_heatmap.png') # Save plot as PNG file
plt.show() # Display plot
print("EDA visualizations generated and saved.")
print("-" * 50)

# Task 4: Data Splitting
print("\n--- Task 4: Data Splitting ---")
# Separate features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets (30% for testing, random state 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the training and testing sets
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
print("-" * 50)

# Task 5: Model Architecture
print("\n--- Task 5: Model Architecture ---")
# Create a Sequential model, which is the simplest type of model in Keras (a linear stack of layers)
model = tf.models.Sequential()
# Input Layer: Add the first Dense layer. input_dim automatically matches the number of input features.
model.add(tf.layers.Dense(100, activation='relu', input_dim=X_train.shape[1])) # Use ReLU activation function
# Second Hidden Layer: Again use ReLU activation function
model.add(tf.layers.Dense(100, activation='relu'))
# Output Layer: Single neuron, using linear activation function, suitable for regression tasks
model.add(tf.layers.Dense(1, activation='linear'))

print("\nModel Summary:")
# Print the model summary, showing output shape and number of parameters for each layer
model.summary()
print("-" * 50)

# Task 6: Model Compilation
print("\n--- Task 6: Model Compilation ---")
# Initialize RMSprop optimizer (default learning rate)
optimizer = tf.optimizers.RMSprop()
# Compile the model, specifying the optimizer, loss function (mean squared error), and evaluation metric (mean absolute error)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
print("Model compiled with 'rmsprop' optimizer, 'mse' loss, and 'mae' metric.")
print("-" * 50)

# Task 7: Model Training and Visualization
print("\n--- Task 7: Model Training and Visualization ---")
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100, # epochs: Number of training iterations
    batch_size=32, # batch_size: Number of samples per gradient update
    validation_split=0.3, # validation_split: 30% of training data split for validation set
    verbose=1 # verbose: Display training progress (1 for progress bar)
)

print("\nModel training completed.") # Print model training completion message

# Visualize training and validation loss
plt.figure(figsize=(12, 6)) # Create a figure with two subplots
plt.subplot(1, 2, 1) # First subplot: Training loss vs. Validation loss
plt.plot(history.history['loss'], label='Training Loss') # Plot training loss curve
plt.plot(history.history['val_loss'], label='Validation Loss') # Plot validation loss curve
plt.title('Training and Validation Loss') # Set subplot title
plt.xlabel('Epoch') # Set X-axis label
plt.ylabel('Loss (MSE)') # Set Y-axis label
plt.legend() # Set legend

# Visualize training and validation MAE
plt.subplot(1, 2, 2) # Second subplot: Training MAE vs. Validation MAE
plt.plot(history.history['mean_absolute_error'], label='Training MAE') # Plot training MAE curve
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE') # Plot validation MAE curve
plt.title('Training and Validation MAE') # Set subplot title
plt.xlabel('Epoch') # Set X-axis label
plt.ylabel('Mean Absolute Error') # Set Y-axis label
plt.legend() # Set legend

plt.tight_layout() # Automatically adjust subplot parameters for a tight layout
plt.savefig('training_performance_plots.png') # Save training performance plots
plt.show() # Display plot
print("Training performance plots generated and saved.")
print("-" * 50)

# Task 8: Model Evaluation
print("\n--- Task 8: Model Evaluation ---")
# Evaluate model performance on the test set (verbose=0 for no progress display)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
# Print loss (MSE) and mean absolute error (MAE) on the test set
print(f"\nEvaluation on Test Set:")
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")

# Print model performance discussion and improvement suggestions
print("\nDiscussion of Model Performance and Improvements:")
# The model's performance on the test set indicates its generalization ability.
print("The model's performance on the test set gives an indication of its generalization ability.")
# Print the achieved Test MSE and MAE
print(f"A Test MSE of {loss:.2f} and MAE of {mae:.2f} were achieved.")