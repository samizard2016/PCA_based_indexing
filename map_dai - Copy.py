import folium
import pandas as pd

def plot_villages_on_map(villages):
    """
    Plots villages on a map with a color legend based on digital accessibility.

    Parameters:
    villages (list of dicts): A list where each dictionary contains 'lat', 'lon', and 'digital_accessibility' keys.
                                   Example: [{'lat': 12.34, 'lon': 56.78, 'digital_accessibility': 'High'}, ...]
    """
    # Create a base map
    map_center = [villages[0]['lat'], villages[0]['lon']]
    village_map = folium.Map(location=map_center, zoom_start=10)

    # Define a color mapping for the digital accessibility levels
    color_mapping = {
        'High': 'green',
        'Medium': 'orange',
        'Low': 'red'
    }

    # Add villages to the map
    for village in villages:
        lat = village['lat']
        lon = village['lon']
        accessibility = village['digital_accessibility']
        color = color_mapping.get(accessibility, 'gray')  # Default to gray if accessibility level is not recognized

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Digital Accessibility: {accessibility}"
        ).add_to(village_map)

    # Add a legend to the map
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 150px; height: 90px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white;
                     ">
             <strong>Digital Accessibility</strong><br>
             <i class="fa fa-circle" style="color:green"></i> High<br>
             <i class="fa fa-circle" style="color:orange"></i> Medium<br>
             <i class="fa fa-circle" style="color:red"></i> Low<br>
         </div>
     '''
    village_map.get_root().html.add_child(folium.Element(legend_html))

    # Save the map to an HTML file
    village_map.save('villages_digital_accessibility_map.html')

    # Display the map
    return village_map
if __name__=="__main__":
    village_data = pd.read_excel("Weights and DAI v1.0.xlsx",sheet_name="Data with DAI")
    village_data = village_data[["village_latitude","village_longitude","DAI_normalized"]]
    village_data["digital_accessibility"] = pd.qcut(village_data["DAI_normalized"],q=3, labels=['Low', 'Medium', 'High'])
    village_data.rename(columns={"village_latitude":"lat","village_longitude":"lon"},inplace=True)
    village_data = village_data.to_dict(orient='records') 
    plot_villages_on_map(village_data)