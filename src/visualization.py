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
