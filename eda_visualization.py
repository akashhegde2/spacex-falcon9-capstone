import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spacex_data_step2_enriched.csv')
print("Loaded shape:", df.shape)
sns.set_style('whitegrid')

# 1. FlightNumber vs PayloadMass, colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='FlightNumber', y='PayloadMass', hue='Class', data=df, palette=['#d62728','#2ca02c'], s=80)
plt.xlabel('Flight Number', fontsize=14)
plt.ylabel('Payload Mass (kg)', fontsize=14)
plt.title('Flight Number vs Payload Mass, colored by landing outcome')
plt.tight_layout()
plt.savefig('chart1_flightnumber_vs_payloadmass.png', dpi=150)
plt.close()

# 2. FlightNumber vs LaunchSite, colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='FlightNumber', y='LaunchSite', hue='Class', data=df, palette=['#d62728','#2ca02c'], s=80)
plt.xlabel('Flight Number', fontsize=14)
plt.ylabel('Launch Site', fontsize=14)
plt.title('Flight Number vs Launch Site, colored by landing outcome')
plt.tight_layout()
plt.savefig('chart2_flightnumber_vs_launchsite.png', dpi=150)
plt.close()

# 3. PayloadMass vs LaunchSite, colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PayloadMass', y='LaunchSite', hue='Class', data=df, palette=['#d62728','#2ca02c'], s=80)
plt.xlabel('Payload Mass (kg)', fontsize=14)
plt.ylabel('Launch Site', fontsize=14)
plt.title('Payload Mass vs Launch Site, colored by landing outcome')
plt.tight_layout()
plt.savefig('chart3_payloadmass_vs_launchsite.png', dpi=150)
plt.close()

# 4. Success rate by Orbit type
orbit_success = df.groupby('Orbit')['Class'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=orbit_success.index, y=orbit_success.values, hue=orbit_success.index, palette='viridis', legend=False)
plt.xlabel('Orbit Type', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.title('Landing Success Rate by Orbit Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart4_success_rate_by_orbit.png', dpi=150)
plt.close()

# 5. FlightNumber vs Orbit, colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='FlightNumber', y='Orbit', hue='Class', data=df, palette=['#d62728','#2ca02c'], s=80)
plt.xlabel('Flight Number', fontsize=14)
plt.ylabel('Orbit Type', fontsize=14)
plt.title('Flight Number vs Orbit Type, colored by landing outcome')
plt.tight_layout()
plt.savefig('chart5_flightnumber_vs_orbit.png', dpi=150)
plt.close()

# 6. PayloadMass vs Orbit, colored by Class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PayloadMass', y='Orbit', hue='Class', data=df, palette=['#d62728','#2ca02c'], s=80)
plt.xlabel('Payload Mass (kg)', fontsize=14)
plt.ylabel('Orbit Type', fontsize=14)
plt.title('Payload Mass vs Orbit Type, colored by landing outcome')
plt.tight_layout()
plt.savefig('chart6_payloadmass_vs_orbit.png', dpi=150)
plt.close()

# 7. Yearly success trend
df['Year'] = pd.to_datetime(df['Date']).dt.year
yearly_success = df.groupby('Year')['Class'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=yearly_success.index, y=yearly_success.values, marker='o', linewidth=2.5)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Success Rate', fontsize=14)
plt.title('Launch Success Rate Trend by Year')
plt.tight_layout()
plt.savefig('chart7_yearly_success_trend.png', dpi=150)
plt.close()

print("All 7 EDA visualization charts saved.")
print("\nOrbit success rates:\n", orbit_success)
print("\nYearly success rates:\n", yearly_success)
