import pandas as pd # 導入用於數據處理的 pandas 庫
import numpy as np # 導入用於數值運算的 numpy 庫
from sklearn.model_selection import train_test_split # 從 scikit-learn 導入用於數據分割的函式
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # 從 scikit-learn 導入用於特徵縮放和標籤編碼的類
from tensorflow import keras as tf # 導入 TensorFlow 的 Keras API，並簡寫為 tf
import matplotlib.pyplot as plt # 導入用於繪圖的 matplotlib 庫
import seaborn as sns # 導入用於統計圖形可視化的 seaborn 庫

# 任務 1: 數據加載與檢查
print("--- Task 1: Data Loading and Inspection ---")
try:
    # 嘗試從指定路徑加載數據集
    df = pd.read_csv('/workspace/data/housing.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # 如果文件未找到，則打印錯誤訊息並退出
    print("Error: 'Housing.csv' not found. Please download the dataset and place it in the correct directory.")
    exit()

print("\nInitial Data Overview (First 5 rows):")
# 顯示數據集的前 5 行
print(df.head())

print("\nDataset Structure and Info:")
# 顯示數據集的結構和信息，包括列名、非空值數量和數據類型
df.info()

print("\nSummary Statistics:")
# 顯示數據集的描述性統計信息，如均值、標準差等
print(df.describe())
print("-" * 50)

# 任務 2: 數據預處理
print("\n--- Task 2: Data Preprocessing ---")

# 2.1 處理缺失值
# 檢查每列的缺失值數量
print("\nMissing values before handling:")
print(df.isnull().sum())

# 遍歷所有列以處理缺失值
for column in df.columns:
    # 如果列中有缺失值
    if df[column].isnull().any():
        # 如果是類別型數據，用眾數填充缺失值
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column].dtype):
            df[column].fillna(df[column].mode()[0], inplace=True)
            print(f"Filled missing values in categorical column '{column}' with mode.")
        # 如果是數值型數據，用均值填充缺失值
        elif pd.api.types.is_numeric_dtype(df[column].dtype):
            df[column].fillna(df[column].mean(), inplace=True)
            print(f"Filled missing values in numerical column '{column}' with mean.")

# 打印處理缺失值後的數量 (應為 0)
print("\nMissing values after handling (should be 0):")
print(df.isnull().sum())


# 2.2 特徵縮放 (使用 MinMaxScaler 將數值特徵縮放到 0-1 範圍)
# 定義需要進行縮放的數值列
numerical_cols_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
# 初始化 MinMaxScaler
scaler = MinMaxScaler()
# 對選定的數值列進行縮放
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])
# 打印應用縮放後的訊息和前 5 行數據
print(f"\nApplied MinMaxScaler to columns: {numerical_cols_to_scale}")
print(df[numerical_cols_to_scale].head())

# 2.3 標籤編碼 (使用 LabelEncoder 將類別特徵轉換為數值)
# 自動識別並處理所有 'object' 類型的列（即字串類型的類別特徵）
for col in df.select_dtypes(include=['object']).columns:
    # 確保不會錯誤地處理目標變數 'price' (如果它碰巧是 object 類型)
    if col != 'price':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Applied LabelEncoder to categorical column: {col}")
    else:
        print(f"Warning: Skipping label encoding for 'price' column as it's the target variable.")

print("\nData after preprocessing (First 5 rows):")
print(df.head())
print("-" * 50)

# 任務 3: 探索性數據分析 (EDA)
print("\n--- Task 3: Exploratory Data Analysis (EDA) ---")

# 3.1 再次生成總結統計 (預處理後)
print("\nSummary Statistics (after preprocessing):")
print(df.describe())

# 3.2 可視化重要特徵
# 繪製房價分佈圖
plt.figure(figsize=(10, 6)) # 設置圖形大小
sns.histplot(df['price'], kde=True) # 使用 histplot 繪製價格分佈，並添加 KDE 曲線
plt.title('Distribution of House Prices') # 設置圖形標題
plt.xlabel('Price') # 設置 X 軸標籤
plt.ylabel('Frequency') # 設置 Y 軸標籤
plt.savefig('eda_price_distribution.png') # 保存圖形為 PNG 文件
plt.show() # 顯示圖形

# 繪製面積 (Area) 與價格 (Price) 的散點圖
plt.figure(figsize=(10, 6)) # 設置圖形大小
sns.scatterplot(x='area', y='price', data=df) # 使用 scatterplot 繪製散點圖
plt.title('Area vs. Price') # 設置圖形標題
plt.xlabel('Area (scaled)') # 設置 X 軸標籤
plt.ylabel('Price') # 設置 Y 軸標籤
plt.savefig('eda_area_vs_price.png') # 保存圖形為 PNG 文件
plt.show() # 顯示圖形

# 繪製臥室數量 (Bedrooms) 對價格 (Price) 影響的箱線圖
plt.figure(figsize=(10, 6)) # 設置圖形大小
sns.boxplot(x='bedrooms', y='price', data=df) # 使用 boxplot 繪製箱線圖 (臥室數量已被縮放，但仍可觀察趨勢)
plt.title('Bedrooms vs. Price') # 設置圖形標題
plt.xlabel('Bedrooms (scaled)') # 設置 X 軸標籤
plt.ylabel('Price') # 設置 Y 軸標籤
plt.savefig('eda_bedrooms_vs_price.png') # 保存圖形為 PNG 文件
plt.show() # 顯示圖形

# 繪製特徵相關性熱圖
plt.figure(figsize=(12, 10)) # 設置圖形大小
correlation_matrix = df.corr() # 計算數據框中所有特徵的相關性矩陣
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # 使用 heatmap 繪製熱圖，顯示相關性值並使用 coolwarm 調色板
plt.title('Correlation Matrix of Features') # 設置圖形標題
plt.savefig('eda_correlation_heatmap.png') # 保存圖形為 PNG 文件
plt.show() # 顯示圖形
print("EDA visualizations generated and saved.")
print("-" * 50)

# 任務 4: 數據分割
print("\n--- Task 4: Data Splitting ---")
# 分離特徵 (X) 和目標變數 (y)
X = df.drop('price', axis=1)
y = df['price']

# 將數據集分割為訓練集和測試集 (測試集佔 30%，隨機狀態為 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 打印訓練集和測試集的形狀
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
print("-" * 50)

# 任務 5: 模型架構
print("\n--- Task 5: Model Architecture ---")
# 創建一個 Sequential 模型，這是 Keras 中最簡單的模型類型 (層的線性堆疊)
model = tf.models.Sequential()
# 輸入層: 添加第一個 Dense 層。input_dim 自動匹配輸入特徵數量。
model.add(tf.layers.Dense(100, activation='relu', input_dim=X_train.shape[1])) # 使用 ReLU 激活函數
# 第二個隱藏層: 再次使用 ReLU 激活函數
model.add(tf.layers.Dense(100, activation='relu'))
# 輸出層: 單個神經元，使用線性激活函數，適用於回歸任務
model.add(tf.layers.Dense(1, activation='linear'))

print("\nModel Summary:")
# 打印模型的摘要，顯示各層的輸出形狀和參數數量
model.summary()
print("-" * 50)

# 任務 6: 模型編譯
print("\n--- Task 6: Model Compilation ---")
# 初始化 RMSprop 優化器 (默認學習率)
optimizer = tf.optimizers.RMSprop() # 默认学习率为 0.001
# 編譯模型，指定優化器、損失函數 (均方誤差) 和評估指標 (平均絕對誤差)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
print("Model compiled with 'rmsprop' optimizer, 'mse' loss, and 'mae' metric.")
print("-" * 50)

# 任務 7: 模型訓練與可視化
print("\n--- Task 7: Model Training and Visualization ---")
# 訓練模型
history = model.fit(
    X_train, y_train,
    epochs=100, # epochs: 訓練迭代次數
    batch_size=32, # batch_size: 每次梯度更新使用的樣本數
    validation_split=0.3, # validation_split: 從訓練數據中劃分 30% 作為驗證集
    verbose=1 # verbose: 顯示訓練進度 (1 為顯示進度條)
)

print("\nModel training completed.") # 打印模型訓練完成訊息

# 可視化訓練和驗證損失
plt.figure(figsize=(12, 6)) # 創建一個圖形，包含兩個子圖
plt.subplot(1, 2, 1) # 第一個子圖：訓練損失與驗證損失
plt.plot(history.history['loss'], label='Training Loss') # 繪製訓練損失曲線
plt.plot(history.history['val_loss'], label='Validation Loss') # 繪製驗證損失曲線
plt.title('Training and Validation Loss') # 設置子圖標題
plt.xlabel('Epoch') # 設置 X 軸標籤
plt.ylabel('Loss (MSE)') # 設置 Y 軸標籤
plt.legend() # 設置圖例

# 可視化訓練和驗證 MAE
plt.subplot(1, 2, 2) # 第二個子圖：訓練 MAE 與驗證 MAE
plt.plot(history.history['mean_absolute_error'], label='Training MAE') # 繪製訓練 MAE 曲線
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE') # 繪製驗證 MAE 曲線
plt.title('Training and Validation MAE') # 設置子圖標題
plt.xlabel('Epoch') # 設置 X 軸標籤
plt.ylabel('Mean Absolute Error') # 設置 Y 軸標籤
plt.legend() # 設置圖例

plt.tight_layout() # 自動調整子圖參數，使之填充整個圖形區域
plt.savefig('training_performance_plots.png') # 保存訓練性能圖形
plt.show() # 顯示圖形
print("Training performance plots generated and saved.")
print("-" * 50)

# 任務 8: 模型評估
print("\n--- Task 8: Model Evaluation ---")
# 在測試集上評估模型性能 (verbose=0 不顯示進度)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
# 打印測試集上的損失 (MSE) 和平均絕對誤差 (MAE)
print(f"\nEvaluation on Test Set:")
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test Mean Absolute Error (MAE): {mae:.4f}")