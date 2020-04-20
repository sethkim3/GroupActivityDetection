from generate_samples import *
from learn_activities import *


# Example of generating sample data.
n_samples = 100
n_individuals = 50
# activities = ['sports']
# grid_size = [100,100]
# activities = ['traveling']
# grid_size = [100,100]
# activities = ['media']
# grid_size = [100,5]
# activities = ['eating']
# grid_size = [100,100]
activities = ['sports', 'traveling', 'media', 'eating', 'line']
grid_size = [100,100]
noise = 0.1
labeled = True

print('GENERATING SAMPLES...')
samples = generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
print(samples)

features = ['x_pos_high',
            'x_pos_low',
            'y_pos_high',
            'y_pos_low',
            'mean_distance',
            'total_distance',
            'x_vel_sum_of_squares',
            'y_vel_sum_of_squares',
            'x_vel_mean',
            'y_vel_mean',
            'mean_speed',
            'mean_graph_clustering',
            'mean_graph_closeness_centrality',
            'min_graph_closeness_centrality',
            'max_graph_closeness_centrality']

print("ENGINEERING FEATURES...")
features_df = engineer_features(samples, n_samples, n_individuals, features)
print(features_df)

# Example of generating a single group's sample data and plotting the positions/velocities.
group_df = generate_group_activity_df(1, 'line', n_individuals, grid_size, noise, 1)
plot_group(1, group_df)
visualize_distance_network(1, group_df, n_individuals)