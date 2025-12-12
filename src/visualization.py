# File: src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def format_labels(arr):
    if arr.dtype.kind in {'S', 'U'} and isinstance(arr[0], (bytes, np.bytes_)):
        return np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in arr])
    
    return arr

def plot_target_distribution(data):
    vals = data['target']
    unique, counts = np.unique(vals, return_counts=True)

    plt.figure(figsize=(6, 4))

    plt.pie(counts, labels=['No change job', 'Change job'], autopct='%1.1f%%', startangle=90, colors=['#6495ED', '#FF7F50'], 
            explode=(0, 0.1), shadow=True, textprops={'color': 'black', 'fontsize':8, 'fontweight': 'bold'})
    
    plt.title('Proportion of Target', fontsize=12, fontweight='bold', color='red')
    plt.show()

def plot_categorical_vs_target(data):
    cols = ['gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_type']

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle('Correlation categorical vs. target', fontsize=12, fontweight='bold', color='red')

    for i, col in enumerate(cols):
        ax = axes[i//2, i%2]

        feature_vals = format_labels(data[col]) 
        target_vals = data['target']

        mask = (feature_vals.astype(str) != '')

        filtered_features = feature_vals[mask]
        filtered_targets = target_vals[mask]
        
        sns.countplot(x=filtered_features, hue=filtered_targets, palette={0: "#6495ED", 1: "#FF7F50"}, ax=ax)   

        ax.set_yscale('log')

        ax.set_title(f'{col}', fontsize=10, color='blue')
        ax.set_ylabel('Frequency', fontsize=10)

        ax.tick_params(axis='x', rotation=45, labelsize= 10)

        if i == 0:
            ax.legend(labels=['No change', 'Change'])
        else:
            if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_categorical_vs_target_stacked(data):
    cols = ['experience', 'company_size', 'last_new_job']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) 
    fig.suptitle('Correlation categorical vs. target', fontsize=14, fontweight='bold', color='red')

    for i, col_name in enumerate(cols):
        ax = axes[i]
        
        feature_vals = data[col_name]

        if feature_vals.dtype.kind in {'S', 'U'} and isinstance(feature_vals[0], bytes):
            feature_vals = np.array([x.decode('utf-8') for x in feature_vals])
        else:
            feature_vals = feature_vals.astype(str)
            
        target_vals = data['target'].astype(int)

        mask = (feature_vals != 'nan') & (feature_vals != 'NaN') & (feature_vals != '') & (feature_vals != 'Unknown')
        feature_vals = feature_vals[mask]
        target_vals = target_vals[mask]

        unique_cats = np.unique(feature_vals)

        pct_stay = []
        pct_leave = []
        
        for cat in unique_cats:
            cat_mask = (feature_vals == cat)
            total = np.sum(cat_mask)
            
            if total == 0:
                pct_stay.append(0)
                pct_leave.append(0)
                continue
                
            count_leave = np.sum(target_vals[cat_mask] == 1)
            count_stay = total - count_leave
            
            pct_stay.append((count_stay / total) * 100)
            pct_leave.append((count_leave / total) * 100)

        ax.bar(unique_cats, pct_stay, label='No change', color="#6495ED", alpha=0.9, width=0.6)
        ax.bar(unique_cats, pct_leave, bottom=pct_stay, label='Change', color="#FF7F50", alpha=0.9, width=0.6)

        ax.set_title(f'{col_name}', fontsize=12, color='blue')
        ax.set_ylabel('Percentage (%)')
        
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

        if i == 0:
            ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9) 
    plt.show()
    
def plot_numerical_vs_target(data):
    cols = ['city_development_index', 'training_hours']

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle('Correlation numerical vs. target', fontsize=12, fontweight='bold', color='red')

    for i, col in enumerate(cols):
        ax_kde = axes[i, 0]
        ax_box = axes[i, 1]

        targets_int = data['target'].astype(int)
    
        val_stay = data[cols[i]][data['target'] == 0]
        val_leave = data[cols[i]][data['target'] == 1]

        sns.kdeplot(val_stay, fill=True, color="#4c72b0", ax=ax_kde)
        sns.kdeplot(val_leave, fill=True, color="#c44e52", ax=ax_kde)

        ax_kde.set_title(f'{cols[i]}', fontsize=10, color='blue')
        
        targets_str = targets_int.astype(str)
        sns.boxplot(x=targets_str, y=data[col], hue=targets_str, legend=False, palette={'0': "#4c72b0", '1': "#c44e52"}, ax=ax_box, order=['0', '1'])
        
        ax_box.set_xticks([0, 1])
        ax_box.set_xticklabels(['No change', 'Change'])
        ax_box.set_title(f'{cols[i]}', fontsize=10, color='blue')

        if i == 0:
            ax_kde.legend(labels=['No change', 'Change'])
        else:
            if ax_kde.get_legend(): ax_kde.get_legend().remove()        

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_correlation_heatmap(X_matrix, y_vector, feature_names_list):
    if y_vector.ndim == 1:
        y_vector = y_vector.reshape(-1, 1)
        
    data_all = np.column_stack([X_matrix, y_vector])
    
    labels = feature_names_list + ['Target']
    
    corr_matrix = np.corrcoef(data_all, rowvar=False)
    
    plt.figure(figsize=(6, 6)) 
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, 
                linewidths=0.5, linecolor='gray')
    
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', color='red')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_most_city(data):
    candidates = data[data['target'] == 1]

    unique_cities, counts = np.unique(candidates['city'], return_counts=True)

    sorted_indices = np.argsort(counts)[::-1]

    top_5_indices = sorted_indices[:5]

    top_cities = unique_cities[top_5_indices]
    top_counts = counts[top_5_indices]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=top_counts, y=top_cities, hue=top_cities, legend=False, palette='viridis')

    plt.title('Top 5 Cities with the Most Potential Candidates (Target=1)', fontsize=14, fontweight='bold', color='red')
    plt.xlabel('Candidates')
    plt.ylabel('City')

    for i, v in enumerate(top_counts):
        plt.text(v, i, f' {v}', va='center')

    plt.tight_layout()
    plt.show()

def plot_STEM_vs_nonSTEM(data):
    is_stem = data['major_discipline'] == 'STEM'

    stem_group = data['target'][is_stem]

    non_stem_group = data['target'][~is_stem]

    stem_rate = np.mean(stem_group) * 100
    non_stem_rate = np.mean(non_stem_group) * 100

    categories = ['STEM', 'Non-STEM']
    values = [stem_rate, non_stem_rate]

    plt.figure(figsize=(6, 4))

    ax = sns.barplot(x=categories, y=values, hue=values, legend=False, palette=['#3498db', '#e74c3c'])

    plt.title('Conversion Rate\nbetween STEM and Non-STEM', fontsize=14, fontweight='bold', color='red')
    plt.ylabel('Percentage of Potential Candidates (%)')
    plt.xlabel('Group')
    plt.ylim(0, 100) 

    for i, v in enumerate(values):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    avg_total = np.mean(data['target']) * 100
    plt.axhline(avg_total, color='gray', linestyle='--', label=f'Overall Average ({avg_total:.1f}%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_most_characteristic(data):
    features_to_scan = [
        'gender', 'relevent_experience', 'enrolled_university', 
        'education_level', 'major_discipline', 'experience', 
        'company_size', 'company_type', 'last_new_job'
    ]

    all_categories = []

    for col in features_to_scan:
        unique_vals = np.unique(data[col])
        
        for val in unique_vals:
            if str(val) == 'nan' or str(val) == '': continue

            mask = data[col] == val
            targets_in_group = data['target'][mask]
            
            if len(targets_in_group) > 20: 
                rate = np.mean(targets_in_group) * 100
                
                val_str = str(val)
                if isinstance(val, bytes):
                    val_str = val.decode('utf-8')
                
                display_name = f"{col}: {val_str}"
                
                all_categories.append((display_name, rate))

    result_arr = np.array(all_categories, dtype=[('name', 'U50'), ('rate', 'f4')])

    sorted_indices = np.argsort(result_arr['rate'])[::-1]

    top_7 = result_arr[sorted_indices][:7]

    plt.figure(figsize=(10, 4))
    
    ax = sns.barplot(x=top_7['rate'], y=top_7['name'], palette='deep', hue=top_7['name'], legend=False, dodge=False)

    plt.title('Top 7 Employee Characteristics with the Most Potential Candidates', fontsize=14, fontweight='bold', color='red')
    plt.xlabel('Percentage of Potential Candidates (%)', fontsize=12)
    plt.ylabel('Employee Characteristics', fontsize=12)

    for i, v in enumerate(top_7['rate']):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')

    avg_rate = np.mean(data['target']) * 100
    plt.axvline(x=avg_rate, color='red', linestyle='--', linewidth=1.5)
    plt.text(avg_rate, len(top_7)-0.5, f' Average: {avg_rate:.1f}%', color='red', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_compare_education_value(data):
    mask_low = data['city_development_index'] < 0.7
    mask_high = data['city_development_index'] >= 0.7
    
    unique_levels = np.unique(data['education_level'])
    
    comparison_data = []

    for level in unique_levels:
        level_str = str(level)
        if isinstance(level, bytes):
            level_str = level.decode('utf-8')
        if level_str == 'nan' or level_str == '': continue
        
        mask_level_low = (data['education_level'] == level) & mask_low
        targets_low = data['target'][mask_level_low]
        rate_low = np.mean(targets_low) * 100 if len(targets_low) > 10 else 0
            
        mask_level_high = (data['education_level'] == level) & mask_high
        targets_high = data['target'][mask_level_high]
        rate_high = np.mean(targets_high) * 100 if len(targets_high) > 10 else 0
            
        comparison_data.append((level_str, rate_low, 'Low CDI'))
        comparison_data.append((level_str, rate_high, 'High CDI'))

    plot_data = np.array(comparison_data, dtype=[('level', 'U50'), ('rate', 'f4'), ('city_type', 'U50')])

    plt.figure(figsize=(11, 6))
    
    ax = sns.barplot(
        x=plot_data['level'], y=plot_data['rate'], hue=plot_data['city_type'], palette={'Low CDI': '#e74c3c', 'High CDI': '#2c3e50'})

    plt.title('Potential Candidates by Education Level and City Development Index', fontsize=14, fontweight='bold', color='red')
    plt.ylabel('Percentage of Potential Candidates (%)', fontsize=12)
    plt.xlabel('Education Level', fontsize=12)
    plt.legend(title='City Type')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

    plt.tight_layout()
    plt.show()

