import numpy as np
from numpy.random import *
import pandas as pd
import matplotlib.pyplot as plt

def generate_group_activity_df(group_num, activity, n_individuals, grid_size, noise, label=None):
    """
    Generates a df for a single group of individuals depending on the assigned activity.
    :param group_num: The group number.
    :param activity: The activity that has been assigned to the group.
    :param n_individuals: The number of individuals in the group.
    :param grid_size: Size of the grid. Default value is 100x100
    :param noise: Noise to add to position and velocity assignments. Default value is 0.1.
    :param label: The label to assign to the label column if set True.
    :return: A dataframe of size n_individuals rows and 7 columns.
    """

    group_df = pd.DataFrame(columns=['group_num', 'individual_num', 'x_pos', 'y_pos', 'x_vel', 'y_vel', 'activity_label'])

    if(activity == 'test_activity_1'):
        for row in range(n_individuals):
            x_pos = np.random.uniform(0,1) * grid_size[0]
            y_pos = np.random.uniform(0,1) * grid_size[1]
            x_vel = np.random.uniform(-1,1) * 5
            y_vel = np.random.uniform(-1,1) * 5
            group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)

    return group_df

def plot_group(group_num, group_df):
    """
    Plots the individual points from a group plot with directional arrows for the velocities.
    :param group_df: A group_df generated from generate_group_activity_df.
    :return: N/A
    """

    x = group_df['x_pos'].values
    y = group_df['y_pos'].values

    u = group_df['x_vel'].values
    v = group_df['y_vel'].values
    pos_x = x + u / 2
    pos_y = y + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    plt.title('Positions and Velocities of Individuals in Group ' + str(group_num))
    plt.show()

def generate_samples(n_samples,n_individuals,activities,grid_size=[100,100],noise=0.1,labeled=True):
    """
    This function will generate the group activity samples and return them in a pandas dataframe.
    The dataframe will contain n_samples*n_individuals rows.
    The dataframe will contain 7 columns (group_num,individual_num,x_pos,y_pos,x_vel,y_vel,activity_label).
    Each individual from the group will contain the same label as the group.
    :param n_samples: Total number of samples to generate
    :param n_individuals: Number of individuals per sample
    :param activities: A list of activities to generate samples on. The number of samples will be divided evenly among
    the activities.
    :param grid_size: The grid size to place the individuals on. Default size is 100x100.
    :param noise: The noise to add in the position and velocity assignments. Default value is 0.1.
    :param labeled: Whether or not to label the individuals depending on the group activity. Default True.
    :return: A pandas dataframe of n_samples*n_individuals rows, and 7 columns.
    """

    df = pd.DataFrame(columns=['group_num', 'individual_num', 'x_pos', 'y_pos', 'x_vel', 'y_vel', 'activity_label'])

    # assign labels to activities and activities to groups
    groups_per_activity = round(n_samples/len(activities))
    activity_label = {}
    group_assignment = {}
    activity_count = 0
    group_count = 0
    for activity in activities:
        for group in range(groups_per_activity):
            group_assignment[group_count] = activity
            group_count += 1
        activity_label[activity] = int(activity_count)
        activity_count += 1

    for group_num in range(n_samples):
        activity = group_assignment[group_num]
        label = activity_label[activity]
        if(labeled == False):
            label = None
        group_df = generate_group_activity_df(group_num, activity, n_individuals, grid_size, noise, label)

        df = df.append(group_df, ignore_index=True)

    print(df.head)


n_samples = 100
n_individuals = 50
activities = ['test_activity_1']
grid_size = [100,100]
noise = 0.1
labeled = True

# generate_samples(n_samples, n_individuals, activities, grid_size, noise, labeled)
group_df = generate_group_activity_df(1, 'test_activity_1', 50, [100,100], 0.1, 1)
plot_group(1, group_df)