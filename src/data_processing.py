import numpy as np
import os

def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Not found file")

        return None
    
    try:
        data = np.genfromtxt(file_path, delimiter=',', dtype=None, names=True, encoding='utf-8')
        print(f"Load successfully!")

        return data
    except Exception as e:
        print(f"Error reading file: {e}")

        return None

def get_data_dictionary(data):
    column_desc = {
    'enrollee_id': ['Unique ID for candidate', 'ID'],
    'city': ['City code', 'Code'],
    'city_development_index': ['Development index of the city (scaled)', 'Index (Float)'],
    'gender': ['Gender of candidate', 'Category'],
    'relevent_experience': ['Relevant experience of candidate', 'Category'],
    'enrolled_university': ['Type of University course enrolled', 'Category'],
    'education_level': ['Education level of candidate', 'Ordinal'],
    'major_discipline': ['Education major discipline', 'Category'],
    'experience': ['Candidate total experience in years', 'Years'],
    'company_size': ["No of employees in employer's company", 'Ordinal'],
    'company_type': ['Type of current employer', 'Category'],
    'last_new_job': ['Years between previous and current job', 'Years'],
    'training_hours': ['Training hours completed', 'Hours'],
    'target': ['0: Not looking, 1: Looking for change', 'Binary Target']
    }   

    row_format = "{:<25} | {:<45} | {:<20} | {:<15}"
    
    # In Header
    print("-" * 115)
    print(row_format.format("Feature Name", "Description", "Unit", "Data Type"))
    print("-" * 115)

    column_names = data.dtype.names

    for col in column_names:
        # Lấy thông tin từ dictionary mô tả, nếu không có thì để N/A
        desc_info = column_desc.get(col, ['N/A', 'N/A'])
        description = desc_info[0]
        unit = desc_info[1]
        
        np_dtype = str(data.dtype[col])
        
        print(row_format.format(col, description, unit, np_dtype))
    
    print("-" * 115)

def get_dataset_info(data):
    if data is None: return

    print(f"Samples (rows): {data.shape[0]}")
    print(f"\nNumber of Features (columns): {len(data.dtype.names)}")
    print(f"\nFeatures: {list(data.dtype.names)}")
    
    print("\nFirst 3 lines:")

    for i in range(3):
        print(f"Line {i}: {data[i]}")

def check_datatype(data):
    print("✅Data Types")
    print("-" * 50)
    print(f"{'Feature':<25} | {'Type':<10}")
    print("-" * 50)

    for name in data.dtype.names:
        print(f"{name:<25} | {data.dtype[name]}")

def check_duplicated(data): 
    count = len(data) - len(np.unique(data['enrollee_id']))
    print(f"\n✅Number of Duplicates: {count}")

def check_missing_stats(data):
    print("\n✅Missing State")
    print("-" * 50)
    print(f"{'Feature':<25} | {'Count':<10}")
    print("-" * 50)

    total = data.shape[0]
    has_missing = False

    for col in data.dtype.names:
        col_data = data[col]
        missing_count = 0
        
        if np.issubdtype(col_data.dtype, np.number):
            missing_count = np.isnan(col_data).sum()
        else:
            temp_str = col_data.astype(str)
            missing_count = np.sum((temp_str == 'nan') | (temp_str == '') | (temp_str == 'NaN'))
            
        if missing_count > 0:
            has_missing = True
            print(f"{col:<25} | {missing_count:<10}")

def check_valid(data):
    print("\n✅Valid")
    print("-" * 50)
    print(f"{'Feature':<25} | {'Count':<10}")
    print("-" * 50)

    col = data['city_development_index']
    mask_city_index = (col < 0.0) | (col > 1.0)
    count_city_index = col[mask_city_index].shape[0]
    print(f"{'city_development_index':<25} | {count_city_index}")

    col = data['training_hours']
    mask_training_hours = (col <= 0)
    count_training_hours = col[mask_training_hours].shape[0]
    print(f"{'training_hours':<25} | {count_training_hours}")

    col = data['target']
    mask_target = (col != 0.0) & (col != 1.0)
    count_target = col[mask_city_index].shape[0]
    print(f"{'target':<25} | {count_target}")
            
def satistics (data):
    print("\n✅Satistics")
    print("-" * 80)
    print(f"{'Feature':<25} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Median':<10}")
    print("-" * 80)

    for col in ['city_development_index', 'training_hours']:
        print(f"{col:<25} | {np.min(data[col]):<10.2f} | {np.max(data[col]):<10.2f} | {np.mean(data[col]):<10.2f} | {np.median(data[col]):<10.2f}")

# ==========================================
# MISSING VALUE HANDLING 
# ==========================================
def impute_numerical(data, col_name, strategy='median'):
    col_data = data[col_name]
    
    mask = np.isnan(col_data)
    
    if strategy == 'mean':
        fill_val = np.nanmean(col_data)
    else: 
        fill_val = np.nanmedian(col_data)
        
    data[col_name][mask] = fill_val

    return data

def impute_categorical(data, col_name, fill_value='Unknown'):
    col_data = data[col_name]
    
    mask = (col_data == '') | (col_data == 'nan') | (col_data == 'NaN')
    
    data[col_name][mask] = fill_value
    return data

# ==========================================
# ENCODING 
# ==========================================

def encode_ordinal(data, col_name, mapping):
    col_data = data[col_name]

    encoded = np.zeros(len(col_data), dtype=int)
    
    for key, val in mapping.items():
        mask = (col_data == key)
        encoded[mask] = val
        
    return encoded

def encode_frequency(data, col_name):
    col_data = data[col_name]
    
    uniques, counts = np.unique(col_data, return_counts=True)
    
    freqs = counts / len(col_data)
    
    encoded = np.zeros(len(col_data), dtype=float)
    
    for u, f in zip(uniques, freqs):
        mask = (col_data == u)
        encoded[mask] = f
        
    return encoded

def encode_one_hot(data, col_name):
    col_data = data[col_name]
    uniques, inverse_indices = np.unique(col_data, return_inverse=True)
    
    one_hot_matrix = np.eye(len(uniques))[inverse_indices]
    
    return one_hot_matrix, uniques

# ==========================================
# SCALING & OUTLIER REMOVAL
# ==========================================

# def remove_outliers_zscore(X, y, feature_idx = 1, threshold=3.0):
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     std[std == 0] = 1.0 
    
#     z_scores = np.abs((X - mean) / std)
    
#     mask = np.all(z_scores < threshold, axis=1)
    
#     return X[mask], y[mask]

def remove_outliers_zscore(X, y, feature_index=1, threshold=3.0):
    feature_data = X[:, feature_index]
    mean = np.mean(feature_data)
    std = np.std(feature_data)
    
    if std == 0:
        std = 1.0 
    
    z_scores = np.abs((feature_data - mean) / std)
    
    mask = z_scores < threshold
    
    return X[mask], y[mask]

def standard_scaling(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    
    X_scaled = (X - mean) / std

    return X_scaled
# ==========================================
# STATISTICAL HYPOTHESIS TESTING
# ==========================================

def t_test_manual(group1, group2):
    """
    Perform Independent T-test using np.einsum for Variance calculation.
    """
    n1 = len(group1)
    n2 = len(group2)
    
    m1 = np.mean(group1)
    m2 = np.mean(group2)
    
    diff1 = group1 - m1
    ss1 = np.einsum('i,i->', diff1, diff1) 
    var1 = ss1 / (n1 - 1)
    
    diff2 = group2 - m2
    ss2 = np.einsum('i,i->', diff2, diff2)
    var2 = ss2 / (n2 - 1)
    
    # Standard Error
    se = np.sqrt(var1/n1 + var2/n2)
    
    # T-statistic
    t_stat = (m1 - m2) / se
    
    return t_stat, m1, m2