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
activities = ['line']
grid_size = [100,100]
noise = 0.1
labeled = True

generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)

# Example of generating a single group's sample data and plotting the positions/velocities.
group_df = generate_group_activity_df(1, 'line', n_individuals, grid_size, noise, 1)
plot_group(1, group_df)
