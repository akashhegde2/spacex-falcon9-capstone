import pandas as pd
import folium
from folium.plugins import MarkerCluster

df = pd.read_csv('spacex_data_step2_enriched.csv')

launch_sites_df = df.groupby('LaunchSite', as_index=False).agg(
    Latitude=('Latitude', 'first'),
    Longitude=('Longitude', 'first')
)
print(launch_sites_df)

site_map = folium.Map(
    location=[launch_sites_df['Latitude'].mean(), launch_sites_df['Longitude'].mean()],
    zoom_start=4
)

for _, row in launch_sites_df.iterrows():
    folium.Circle(
        location=[row['Latitude'], row['Longitude']],
        radius=1000,
        color='#000000',
        fill=True
    ).add_child(folium.Popup(row['LaunchSite'])).add_to(site_map)

    folium.map.Marker(
        [row['Latitude'], row['Longitude']],
        icon=folium.DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color:#d35400;"><b>{row["LaunchSite"]}</b></div>'
        )
    ).add_to(site_map)

marker_cluster = MarkerCluster().add_to(site_map)

for _, row in df.iterrows():
    marker_color = 'green' if row['Class'] == 1 else 'red'
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='white', icon_color=marker_color)
    ).add_to(marker_cluster)

site_map.save('spacex_launch_sites_map.html')
print("Folium map saved.")
