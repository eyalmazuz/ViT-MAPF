from functools import partial
import glob
from multiprocessing import Pool
import os
from pathlib import Path

from tqdm import tqdm

from graph_utils import rotate_positions_90_clockwise, from_2d_to_1d, from_1d_to_2d
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mapfgraph import MapfGraph
from PIL import Image
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import to_rgb


def count_conflicts(x):
    count = np.bincount(x)
    if len(count) == 1:
        return 0

    if np.count_nonzero(count[1:]) == 1:  # For one conflict at least
        return 0

    max_count = np.max(count[1:])
    return max_count


class VizMapfGraph(MapfGraph):
    def __init__(self, **args):
        MapfGraph.__init__(self, **args)
        self.masks_for_agents = 0
        self.masks = []
        self.map_representation = []

    # def color_to_value(self, color):
    # if color == 'white':
    #     return [255, 255, 255]
    # if color == 'red':
    #     return [255, 0, 0]
    # if color == 'green':
    #     return [0, 255, 0]
    # return [0, 0, 0]

    def draw_graph_to(self, filename):
        normal_pos = dict((n, (n // self.grid_size[1], n % self.grid_size[1])) for n in self.G.nodes())
        # 90 degree rotation of positions is done because Networkx way of visualization
        pos = {k: rotate_positions_90_clockwise(*v) for k, v in normal_pos.items()}

        xy = np.asarray([normal_pos[v] for v in list(self.G)])
        color_map = [to_rgb(n[1]['color']) for n in self.G.nodes(data=True)]
        mapf_image = np.zeros(shape=(self.grid_size[0], self.grid_size[1], 3))
        # for d in range(3):
        mapf_image[xy[:, 0], xy[:, 1], :] = np.array(color_map)
        # The row above takes the relevant color value (by each dimension) and assign it to the mapf image
        # at the relevant position, which defined by the positions of the nodes

        # plt.imshow(mapf_image)
        # plt.axis('off')
        # plt.savefig(filename + '.jpg')
        # plt.close('all')
        np.savez_compressed(filename, mapf_image)

    def draw_2d_mapf_representation(self, filename):
        nd_mapf_file = Path('nd_mapf') / filename
        if os.path.exists(nd_mapf_file):
            return
        mapf_representation = np.zeros((2, self.grid_size[0], self.grid_size[1]))

        if len(self.map_representation) == 0:
            self.map_representation = self.create_map_channel()

        mapf_representation[0] = self.map_representation

        # Make a mask for each path where each point on the mask is given value 1...t where t is the end time
        # then for each point count number of times it has the same value with threshold T, and take the max of values.
        # The ouput of that will be a HxW map, where each point describes the maximum number of agents
        # conflicts at this point on their shortest paths

        if len(self.masks) == 0:
            self.masks = np.zeros((self.grid_size[0], self.grid_size[1], len(self.paths)), dtype=int)
        else:
            self.masks = np.concatenate(
                (self.masks, np.zeros((self.grid_size[0], self.grid_size[1], 1), dtype=int)), axis=2)

        for agent_i, path in enumerate(self.paths):
            if agent_i < self.masks_for_agents:
                continue
            locations = [(loc // self.grid_size[1], loc % self.grid_size[1]) for loc in path]
            rows, cols = zip(*locations)
            mask = np.zeros(tuple(self.grid_size), dtype=int)  # Array of zeros, such as all free
            mask[rows, cols] = range(1, len(locations) + 1)
            self.masks[:, :, agent_i] = mask
            self.masks_for_agents += 1

        mapf_representation[1] = np.apply_along_axis(count_conflicts, 2, self.masks)
        # mapf_representation[1] *= (255.0) / (mapf_representation[1].max())
        # img = Image.fromarray(np.uint8(mapf_representation[1]) * 255)
        # img.show()

        np.savez_compressed(nd_mapf_file, mapf_representation)

    def create_map_channel(self):
        nodes = self.G.nodes(data=True)
        map_channel = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=float)  # self.grid_size[1] == width
        for node in nodes:
            if node[1]['color'] == 'white':  # all free places
                x, y = from_1d_to_2d(node[0], self.grid_size[1])
                map_channel[x, y, :] = [1, 1, 1]  # False means free, True means occupied
                # division by the width gives the row index and the modulo is the column index

        return map_channel

    def draw_nd_image_to(self, filename):
        normal_pos = dict((n, (n // self.grid_size[1], n % self.grid_size[1])) for n in self.G.nodes())
        pos = {k: rotate_positions_90_clockwise(*v) for k, v in normal_pos.items()}
        # 90 degree rotation of positions is done because Networkx way of visualization

        # Setup an ndarray of shape (W, H, N+1) and init the first channel to be the map itself
        mapf_representation = np.zeros((self.num_agents + 1, self.grid_size[0], self.grid_size[1], 3))

        print("Creating MAPF representation of:", mapf_representation.shape, ", Instance", self.instanceid)

        # mapf_representation[0] = 0  # Set 0 where no obstacle, 1 where there is
        map_channel = self.create_map_channel()
        mapf_representation[0] = map_channel
        # Compute all shortestpaths and add each shortest path to the array
        for index, agent_location in enumerate(zip(self.agents_start_pos, self.agents_goal_pos)):
            start_1d = from_2d_to_1d(agent_location[0], self.grid_size[1])
            goal_1d = from_2d_to_1d(agent_location[1], self.grid_size[1])
            agent_astar_path = nx.astar_path(self.G, start_1d, goal_1d)
            agent_astar_locations = [from_1d_to_2d(loc, self.grid_size[1]) for loc in
                                     agent_astar_path]
            # print(agent_astar_locations)
            locations_as_tuple = list(zip(*agent_astar_locations))  # Make it a tuple of pairs
            
            # mask = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=bool)  # Array of False, such as all free
            # mask[locations_as_tuple] = [0, 0, 1]  # This Agent occupies those locations
            # TODO: Change the values of the mask to be the time the agent arrives to this location
            mapf_representation[index + 1] = map_channel
            mapf_representation[index + 1, locations_as_tuple[0], locations_as_tuple[1], :] = [0, 0, 1]
            mapf_representation[index + 1, agent_location[0][0], agent_location[0][1], :] = [1, 0, 0]
            mapf_representation[index + 1, agent_location[1][0], agent_location[1][1], :] = [0, 1, 0]

        # Write the ndarray to the file given in the function
        # summed_img = np.sum(mapf_representation, axis=0) #Used for visualization
        # img = Image.fromarray(np.uint8(summed_img * 255), 'L')
        # img.show()
        np.savez_compressed(filename, mapf_representation)
        # hkl.dump(mapf_representation, filename + '.hkl')

    def draw_nd_agg_image_to(self, filename):
        normal_pos = dict((n, (n // self.grid_size[1], n % self.grid_size[1])) for n in self.G.nodes())
        pos = {k: rotate_positions_90_clockwise(*v) for k, v in normal_pos.items()}
        # 90 degree rotation of positions is done because Networkx way of visualization

        # Setup an ndarray of shape (W, H, N+1) and init the first channel to be the map itself
        mapf_representation = np.zeros((self.num_agents + 1, self.grid_size[0], self.grid_size[1], 3))

        print("Creating MAPF representation of:", mapf_representation.shape, ", Instance", self.instanceid)

        # mapf_representation[0] = 0  # Set 0 where no obstacle, 1 where there is
        map_channel = self.create_map_channel()
        mapf_representation[0] = map_channel
        # Compute all shortestpaths and add each shortest path to the array
        last_locations_as_tuple = [[], []]
        last_agent_location = [[], []]
        for index, agent_location in enumerate(zip(self.agents_start_pos, self.agents_goal_pos)):
            start_1d = from_2d_to_1d(agent_location[0], self.grid_size[1])
            goal_1d = from_2d_to_1d(agent_location[1], self.grid_size[1])
            agent_astar_path = nx.astar_path(self.G, start_1d, goal_1d)
            agent_astar_locations = [from_1d_to_2d(loc, self.grid_size[1]) for loc in
                                     agent_astar_path]
            # print(agent_astar_locations)
            locations_as_tuple = list(zip(*agent_astar_locations))  # Make it a tuple of pairs
            
            # mask = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=bool)  # Array of False, such as all free
            # mask[locations_as_tuple] = [0, 0, 1]  # This Agent occupies those locations
            # TODO: Change the values of the mask to be the time the agent arrives to this location

            

            mapf_representation[index + 1] = mapf_representation[index]
            mapf_representation[index + 1, locations_as_tuple[0], locations_as_tuple[1], :] = [0, 0, 1]
            mapf_representation[index + 1, agent_location[0][0], agent_location[0][1], :] = [1, 0, 0]
            mapf_representation[index + 1, agent_location[1][0], agent_location[1][1], :] = [0, 1, 0]

            if index != 0:
                mapf_representation[index + 1, last_locations_as_tuple[0], last_locations_as_tuple[1], :] = [1, 0.5, 0]
                mapf_representation[index + 1, last_agent_location[0][0], last_agent_location[0][1], :] = [1, 0, 0]
                mapf_representation[index + 1, last_agent_location[1][0], last_agent_location[1][1], :] = [0, 1, 0]

            last_locations_as_tuple = locations_as_tuple
            last_agent_location = agent_location
        # Write the ndarray to the file given in the function
        # summed_img = np.sum(mapf_representation, axis=0) #Used for visualization
        # img = Image.fromarray(np.uint8(summed_img * 255), 'L')
        # img.show()
        np.savez_compressed(filename, mapf_representation)
        # hkl.dump(mapf_representation, filename + '.hkl')


tqdm.pandas()
scen_suffix = 'custom'
base_dir = Path('./')
maps_dir = base_dir / 'mapf-map'
scen_dir = base_dir / 'scen-all'


def visualize_mapf_problem(row, graph, grid_name=None, instance_id=None, problem_type=None):
    _, row = row
    n_agents = row['NumOfAgents']
    curr_scen = str(scen_dir / row['scen'])
    print(curr_scen, os.path.exists(curr_scen))
    if instance_id is None:
        instance_id = row['InstanceId']
    if problem_type is None:
        problem_type = row['problem_type']
    filename = row['map'].split('.map')[0]
    # For custom:
    mapf_image_path = './data_frames_map_paths_start_goal/{f}-{s}-{i}-{n}.npz'.format(f=filename,
                                                                   s=problem_type,
                                                                   i=instance_id,
                                                                   n=n_agents)
    # mapf_image_path = '../mapf_images/{f}-{s}-{i}-{n}.npz'.format(f=filename, s=problem_type, i=instance_id,
    #                                                               n=n_agents)
    # if os.path.exists(mapf_image_path):
    #     print("Skipping already existing mapf file {f}".format(f=mapf_image_path))
    #     return
    # print("Visualizing {f}".format(f=mapf_image_path))
    graph.load_agents_from_scen(curr_scen, n_agents, instance_id)
    print(graph.num_agents)
    # graph.draw_graph_to(mapf_image_path)
    graph.draw_nd_image_to(mapf_image_path)
    # graph.draw_nd_agg_image_to(mapf_image_path)


def visualize_mapf_scen(scen):
    grid_name, instance_id, problem_type = scen.name
    curr_map = str(maps_dir / scen['map'].iloc[0])
    if not os.path.isfile(curr_map):
        print("Skipping missing map: {m}".format(m=curr_map))
        return scen
    print("Visualizing map: {m}".format(m=curr_map))
    graph = VizMapfGraph(map_filename=curr_map)
    graph.create_graph()

    partial_func = partial(visualize_mapf_problem, graph=graph, grid_name=grid_name,
                           instance_id=instance_id, problem_type=problem_type)

    with Pool(8) as p:
        N = list(tqdm(p.imap(partial_func, scen.iterrows()), total=len(scen)))

    # scen.apply(lambda x: visualize_mapf_problem(x, graph, grid_name, instance_id, problem_type), axis=1)

def main():

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        file_number = int(os.environ['SLURM_ARRAY_TASK_ID'])

        df = pd.read_csv(f'./frames/{file_number}.csv')
    else:
        df = pd.read_csv('MovingAIData-labelled-with-features.csv')

    print(df.shape)

    print(df.head())

    if 'map' not in df.columns or df['map'].isna().any():
        df['map'] = df.GridName + '.map'
    if 'scen' not in df.columns or df['scen'].isna().any():
        df['scen'] = df.apply(lambda x: x['GridName'] + '-' + x['problem_type'] + '-' + str(x['InstanceId']) + '.scen',
                              axis=1)
    df.groupby(['GridName', 'InstanceId', 'problem_type']).progress_apply(visualize_mapf_scen)


if __name__ == "__main__":
    main()
