import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('telecom_data.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with NaN in TotalCharges (usually new customers with 0 tenure)
df = df.dropna(subset=['TotalCharges'])

#plt setup
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# 3. Visualization 1: Churn by Payment Method

payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).rename('Percentage').reset_index()
payment_churn['Percentage'] *= 100

sns.barplot(
    data=payment_churn[payment_churn['Churn'] == 'Yes'], 
    x='PaymentMethod', 
    y='Percentage', 
    ax=axes[0], 
    palette="Reds_r"
)
axes[0].set_title('Churn Rate by Payment Method', fontsize=14)
axes[0].set_ylabel('Churn Rate (%)')
axes[0].tick_params(axis='x', rotation=45)

#visualization 2: Tenure Distribution by Churn
sns.histplot(
    data=df, 
    x='tenure', 
    hue='Churn', 
    multiple="stack", 
    bins=30, 
    ax=axes[1], 
    palette="Set2"
)
axes[1].set_title('Customer Tenure by Churn Status', fontsize=14)
axes[1].set_xlabel('Tenure (Months)')

#Data visualization 3: Monthly Charges vs. Churn
sns.boxplot(
    data=df, 
    x='Churn', 
    y='MonthlyCharges', 
    ax=axes[2], 
    palette="Set3"
)
axes[2].set_title('Monthly Charges vs. Churn', fontsize=14)
axes[2].set_ylabel('Monthly Charges ($)')

#plots
plt.tight_layout()
plt.savefig("telecom_churn_visuals.png")
plt.show()


#summary
print("--- SUMMARY DATA FOR VISUALIZATIONS ---")
print("\n1. Churn Rate by Payment Method:")
print(payment_churn[payment_churn['Churn'] == 'Yes'].to_string(index=False))

print("\n2. Tenure Stats by Churn:")
print(df.groupby('Churn')['tenure'].describe()[['count', 'mean', '50%', 'min', 'max']].round(1).to_string())

print("\n3. Monthly Charges Stats by Churn:")
print(df.groupby('Churn')['MonthlyCharges'].describe()[['mean', '50%', 'min', 'max']].round(1).to_string())
