import pandas as pd

# CSV-Datei einlesen
df = pd.read_csv('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/single_ray_informations.csv')
out_dir = 'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/'
file_name = 'single_ray_density_absolut_result.csv'

# Funktion zur Berechnung der nächsten Distance zu 1 und der zugehörigen Density
def parse_ray_absolut_data(group):
    closest_row = group.iloc[(group['distance'] - 1).abs().argsort()[:1]]
    closest_distance = closest_row['distance'].values[0]
    closest_density = closest_row['density'].values[0]

    rgb = closest_row['rgb'].values[0]
    closest_index = closest_row.index[0]
    density_increases = group.loc[:closest_index, 'density'].fillna(0).tolist()

    # Result dictionary erstellen
    result_dict = {
        'closest_distance': closest_distance,
        'closest_density': closest_density,
        'rgb': rgb
    }

    # Density increases hinzufügen
    for i in range(len(density_increases)):
        result_dict[f'density_increase_{i}'] = density_increases[i]

    return result_dict

# Anwendung der Funktion auf die Gruppen
result = df.groupby('ray_id').apply(parse_ray_absolut_data).reset_index()

# Konvertierung des Ergebnisses in ein DataFrame
result_df = pd.json_normalize(result[0]) #type: ignore

# Spalten von 'result' und 'result_df' zusammenführen
final_result = pd.concat([result.drop(columns=0), result_df], axis=1)

# Ergebnis speichern
final_result.to_csv(out_dir + file_name, index=False)
