import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

def generate_group_activity_df(group_num, activity, n_individuals, grid_size, noise=0.0, label=None):
    """
    Generates a df for a single group of individuals depending on the assigned activity.
    :param group_num: The group number.
    :param activity: The activity that has been assigned to the group.
    :param n_individuals: The number of individuals in the group.
    :param grid_size: Size of the grid. Default value is 100x100
    :param noise: Noise to add to position and velocity assignments. Default value is 0.
    :param label: The label to assign to the label column if set True.
    :return: A dataframe of size n_individuals rows and 7 columns.
    """

    group_df = pd.DataFrame(columns=['group_num', 'individual_num', 'x_pos', 'y_pos', 'x_vel', 'y_vel', 'activity_label'])

    if(activity == 'sports'):
        for row in range(n_individuals):
            x_pos = np.random.uniform(0,1) * grid_size[0] * (1 + np.random.uniform(-1*noise, noise))
            y_pos = np.random.uniform(0,1) * grid_size[1] * (1 + np.random.uniform(-1*noise, noise))
            x_vel = np.random.uniform(-1,1) * 5 * (1 + np.random.uniform(-1*noise, noise))
            y_vel = np.random.uniform(-1,1) * 5 * (1 + np.random.uniform(-1*noise, noise))
            group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)


    #Sitting in a car or bus moving in one direction
    if(activity == 'traveling'):
        x_vel = np.random.uniform(-1,1) * 5 * (1 + np.random.uniform(-1*noise, noise))
        y_vel = np.random.uniform(-1,1) * 5 * (1 + np.random.uniform(-1*noise, noise))
        for row in range(n_individuals):
            x_pos = np.random.uniform(0,1) * grid_size[0] * (1 + np.random.uniform(-1*noise, noise))
            y_pos = np.random.uniform(0,1) * grid_size[1] * (1 + np.random.uniform(-1*noise, noise))
            group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)

    #Sitting in a row or multiple rows watching media
    if(activity == 'media'):
        rows = 10 #create 10 rows
        row_size = grid_size[0]/rows
        num_per_row = int(round(n_individuals/rows))
        for row in range(rows):
            # get positions
            x_pos = row*row_size * (1 + np.random.uniform(-1*noise, noise))
            for individual in range(num_per_row):
                x_vel = np.random.uniform(-1 * noise, noise)
                y_vel = np.random.uniform(-1 * noise, noise)
                y_pos = round(np.random.uniform(0,1) * grid_size[1]) * (1 + np.random.uniform(-1*noise, noise))
                group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)

    #Sitting around a circular table facing inwards
    if(activity == 'eating'):
        angles = np.linspace(0, 2*np.pi, n_individuals)
        center_x,center_y = [grid_size[0]/2,grid_size[1]/2]
        radius = min(grid_size[0]/2,grid_size[1]/2)
        row = 0
        for ang in angles:
            x_pos = center_x + radius*np.cos(ang) * (1 + np.random.uniform(-1*noise, noise))
            y_pos = center_y + radius*np.sin(ang) * (1 + np.random.uniform(-1*noise, noise))
            x_vel = np.random.uniform(-1 * noise, noise)
            y_vel = np.random.uniform(-1 * noise, noise)
            group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)      
            row = row + 1
    #Line of people following eachother
    if(activity == 'line'):
        last_x_pos = np.random.uniform(0,1) * grid_size[0]
        last_y_pos = np.random.uniform(0,1) * grid_size[1]
        num = np.random.randint(2)
        
        if(num):
            for row in range(n_individuals):
                # print("---------X-------")
                x_pos = last_x_pos + 1 * (1 + np.random.uniform(-1*noise, noise))
                y_pos = last_y_pos + np.random.uniform(-0.5,0.5) * (1 + np.random.uniform(-1*noise, noise))
                
                x_vel = ((x_pos - last_x_pos) / grid_size[0]) * (1 + np.random.uniform(-1*noise, noise))
                y_vel = ((y_pos - last_y_pos) / grid_size[1]) * (1 + np.random.uniform(-1*noise, noise))

                last_x_pos = x_pos
                last_y_pos = y_pos
                group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)
        else:
            for row in range(n_individuals):
                # print("---------Y-------")
                x_pos = (last_x_pos + np.random.uniform(-0.5,0.5)) * (1 + np.random.uniform(-1*noise, noise))
                y_pos = (last_y_pos + 1) * (1 + np.random.uniform(-1*noise, noise))
                
                x_vel = ((x_pos - last_x_pos) / grid_size[0]) * (1 + np.random.uniform(-1*noise, noise))
                y_vel = ((y_pos - last_y_pos) / grid_size[1]) * (1 + np.random.uniform(-1*noise, noise))

                last_x_pos = x_pos
                last_y_pos = y_pos
                group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)
    
    
    # #Sitting in a small row or rows all facing the same direction
    # if(activity == 'Media'):
    #     for row in range(n_individuals):
    #         x_pos = np.random.uniform(0,1) * grid_size[0]
    #         y_pos = np.random.uniform(0,1) * grid_size[1]
    #         x_vel = np.random.uniform(-1,1) * 5
    #         y_vel = np.random.uniform(-1,1) * 5
    #         group_df = group_df.append(pd.Series([group_num, row, x_pos, y_pos, x_vel, y_vel, label], index=group_df.columns), ignore_index=True)
            

    return group_df

def plot_group(group_num, group_df):
    """
    Plots the individual points from a group plot with directional arrows for the velocities.
    :param group_num: The group number (for title).
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

def visualize_distance_network(group_num, group_df, n_individuals):
    """
    Visualize a group df as a distance network (for feature engineering).
    :param group_num: The group number (for title).
    :param group_df: A group_df generated from generate_group_activity_df.
    :param n_individuals: Number of individuals in the group.
    :return: N/A
    """
    G = nx.Graph()
    edges_with_weights = []
    individual_combos = combinations(list(range(n_individuals)), 2)
    for i, j in individual_combos:
        x_dist = group_df.iloc[i]['x_pos'] - group_df.iloc[j]['x_pos']
        y_dist = group_df.iloc[i]['y_pos'] - group_df.iloc[j]['y_pos']
        tot_distance = pow(pow(x_dist, 2) + pow(y_dist, 2), 0.5)
        edges_with_weights.append((i, j, tot_distance))
    G.add_weighted_edges_from(edges_with_weights)
    pos = nx.spring_layout(G, weight='weight')
    nx.draw(G, pos, node_color='b', node_size=50, with_labels=False)
    plt.title('Distance Network for Group ' + str(group_num))
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
        label = int(activity_label[activity])
        if(labeled == False):
            label = None
        group_df = generate_group_activity_df(group_num, activity, n_individuals, grid_size, noise, label)

        df = df.append(group_df, ignore_index=True)

    return df