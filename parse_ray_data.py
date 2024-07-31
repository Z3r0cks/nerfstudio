import pandas as pd

# CSV-Datei einlesen
df = pd.read_csv('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/single_ray_informations.csv')
out_dir = 'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/'
file_name = 'result_single_ray_informations.csv'

# Funktion zur Berechnung der nächsten Distance zu 1 und der zugehörigen Density
def find_closest_to_one(group):
   # Finde die Zeile, deren Distance am nächsten an 1 ist
   closest_row = group.iloc[(group['distance'] - 1).abs().argsort()[:1]]
   closest_distance = closest_row['distance'].values[0]
   closest_density = closest_row['density'].values[0]
    
   closest_index = closest_row.index[0] 
   # Berechne die Steigerung der Density bis zu diesem Punkt
   cumulative_density = group[group['distance'] <= closest_distance]['density'].sum()
   cumulative_density_increase = group.loc[:closest_index, 'density'].diff().fillna(0).sum()
   density_increases = group.loc[:closest_index, 'density'].diff().fillna(0).tolist()
    
   return pd.Series({
      'closest_distance': closest_distance,
      'closest_density': closest_density,
      'cumulative_density': cumulative_density,
      'cumulative_density_increase': cumulative_density_increase,
      'density_increases': density_increases
   }) 

# Für jede ray_id die benötigten Werte berechnen
result = df.groupby('ray_id').apply(find_closest_to_one).reset_index()

# save the result
result.to_csv(out_dir + file_name, index=False)
