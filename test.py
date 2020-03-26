from generate_samples import *
from learn_activities import *


# Example of generating sample data.
n_samples = 100
n_individuals = 50
activities = ['test_activity_1']
grid_size = [100,100]
noise = 0.1
labeled = True

generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)

# Example of generating a single group's sample data and plotting the positions/velocities.
group_df = generate_group_activity_df(1, 'test_activity_1', 50, [100,100], 0.1, 1)
plot_group(1, group_df)
