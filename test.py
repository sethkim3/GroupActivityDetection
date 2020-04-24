from generate_samples import *
from learn_activities import *
import csv

# Example of generating sample data.
n_samples = 25
n_individuals = 50
activities = ['sports', 'traveling', 'media', 'eating', 'line']
grid_size = [100,100]
noise = 0.1
labeled = True
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


print('GENERATING SAMPLES...')
samples = generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
print(samples)

print("ENGINEERING FEATURES...")
features_df = engineer_features(samples, n_samples, n_individuals, features)
print(features_df)

print("PREPARING ML MODEL...")
model, scores, returned_features = learn_activities(features_df, 'NaiveBayes')

print("GENERATING NEW DATA...")
new_samples = generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
print(new_samples)

print("ENGINEERING FEATURES ON NEW DATA...")
new_features_df = engineer_features(new_samples, n_samples, n_individuals, features)
print(new_features_df)

print("PREDICTING ON NEW DATA...")
predictions, class_accuracies, class_precisions, class_recalls = predict_activities(new_features_df, model, activities)
print(predictions)

# Example of generating a single group's sample data and plotting the positions/velocities.
# group_df = generate_group_activity_df(1, 'line', n_individuals, grid_size, noise, 1)
# plot_group(1, group_df)
# visualize_distance_network(1, group_df, n_individuals)

# take some variations and save to csv
# with open('classification_model_tests.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Model Type', 'Noise', 'Cross Val Mean Score', 'Cross Val Std Score', 'Top-5 Selected Features',
#                      'Accuracies By Class', 'Precision By Class', 'Recalls By Class'])
#
# for noise in [0.0, 0.1, 0.3, 0.5, 1.0]:
#     samples = generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
#     features_df = engineer_features(samples, n_samples, n_individuals, features)
#     new_samples = generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
#     new_features_df = engineer_features(new_samples, n_samples, n_individuals, features)
#     for model_type in ['NaiveBayes', 'RandomForest', 'GradientBoost']:
#         model, scores, features = learn_activities(features_df, model_type)
#         predictions, class_accuracies, class_precisions, class_recalls = predict_activities(new_features_df, model, activities)
#         class_accuracies_dict = {}
#         class_precisions_dict = {}
#         class_recalls_dict = {}
#         for i in range(len(activities)):
#             class_accuracies_dict[activities[i]] = class_accuracies[i]
#             class_precisions_dict[activities[i]] = class_precisions[i]
#             class_recalls_dict[activities[i]] = class_recalls[i]
#
#         with open('classification_model_tests.csv', 'a') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([model_type, str(noise), str(scores.mean()), str(scores.std()), str(features),
#                              str(class_accuracies_dict), str(class_precisions_dict), str(class_recalls_dict)])