import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/single_ray_informations.csv')

# Daten nach ray_id gruppieren
grouped = df.groupby('ray_id')

# Für jede ray_id einen Plot erstellen
for ray_id, group in grouped:
   if ray_id > 10:
      break
   # Plot erstellen
   fig, ax = plt.subplots()
   plt.figure(figsize=(100, 50), dpi=80)
   ax.plot(group['distance'], group['density'], label='Density', color='blue')
   
   # Punkt finden, der der Distance von 1 am nächsten ist
   # closest_idx = (group['distance'] - 1).abs().idxmin()
   # closest_point = group.loc[closest_idx]
   
   # # find distance not larger than 10
   # closest_idx_10 = (group['distance'] - 10).abs().idxmin()   

   # Speziellen Punkt plotten
   plt.plot()
   
   # Achsen beschriften und Titel setzen
   plt.xlabel('Distance')
   plt.ylabel('Density')
   plt.title(f'Ray ID: {ray_id}')

   # Plot speichern oder anzeigen
   save_dir = f'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/Data/Plots/'
   plt.savefig(f'{save_dir}ray_id_{ray_id}.png')  # Zum Speichern
   # plt.show()  # Zum Anzeigen

# Hinweis: Achten Sie darauf, den Pfad zur CSV-Datei anzupassen
