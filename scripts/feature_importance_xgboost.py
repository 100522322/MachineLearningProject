import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv('./data/vehicles.csv') 

drop_cols = ['url', 'region_url', 'image_url', 'VIN', 'posting_date', 
             'description', 'model', 'region', 'lat', 'long', 'id']

df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
df_clean = df_clean.dropna(subset=['price', 'year', 'odometer'])
df_clean = df_clean[(df_clean['price'] > 500) & (df_clean['price'] < 100000)]

numeric_cols = ['year', 'odometer']
categorical_cols = ['manufacturer', 'fuel', 'condition', 'cylinders', 
                    'transmission', 'drive', 'type', 'paint_color', 'size', 'state']

for col in categorical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('unknown')

print("Preprocessing...")
X = df_clean[numeric_cols + [c for c in categorical_cols if c in df_clean.columns]]
y = df_clean['price']

X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=True)

print("Fast XGBoost...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
model.fit(X, y)

print("Generating graphic...")

importances = model.feature_importances_
feature_names = X.columns


df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_imp = df_imp.sort_values('Importance', ascending=False).head(10) # TOP 10

def clean_name(name):
    name = name.replace('_', ' ').title()
    return name

df_imp['Feature'] = df_imp['Feature'].apply(clean_name)

# PLOT
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5)) 


barplot = sns.barplot(x='Importance', y='Feature', data=df_imp, palette='mako')

plt.title('Top 10 Features of Used Car Prices (XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance (Gain)', fontsize=12)
plt.ylabel('')
plt.tick_params(axis='y', labelsize=11)

# Save
plt.tight_layout()
filename = './metrics/statistical_test/feature_importance.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')

print(f"Image saved: {filename}")
plt.show()