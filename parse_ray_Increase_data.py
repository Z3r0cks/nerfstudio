import pandas as pd

df = pd.read_csv('C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/single_ray_informations.csv')
out_dir = 'C:/Users/free3D/Desktop/Patrick_Kaserer/Masterthesis/'
file_name ='single_ray_density_increase_result.csv'

def parse_ray_increase_data(group):
    closest_row = group.iloc[(group['distance'] - 1).abs().argsort()[:1]]
    closest_distance = closest_row['distance'].values[0]
    closest_density = closest_row['density'].values[0]

    rgb = closest_row['rgb'].values[0]
    closest_index = closest_row.index[0]
    cumulative_density = group[group['distance'] <= closest_distance]['density'].sum()
    cumulative_density_increase = group.loc[:closest_index, 'density'].diff().fillna(0).sum()
    density_increases = group.loc[:closest_index, 'density'].diff().fillna(0).tolist()

    result_dict = {
        'closest_distance': closest_distance,
        'closest_density': closest_density,
        'cumulative_density': cumulative_density,
        'cumulative_density_increase': cumulative_density_increase,
        'rgb': rgb
    }

    for i in range(len(density_increases)):
        result_dict[f'density_increase_{i}'] = density_increases[i]

    return result_dict

result = df.groupby('ray_id').apply(parse_ray_increase_data).reset_index()
result_df = pd.json_normalize(result[0]) #type: ignore
final_result = pd.concat([result.drop(columns=0), result_df], axis=1)
final_result.to_csv(out_dir + file_name, index=False)
