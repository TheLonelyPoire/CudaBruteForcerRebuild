import numpy as np
from matplotlib import colors as clrs

def computeColorMapFromColorList(color_list, name=''):
    step = 1/(len(color_list) - 1)
    half_step = step / 2
    last_col_index = len(color_list) - 1

    red_list = []
    green_list = []
    blue_list = []

    red_list.append([0, color_list[0][0], color_list[0][0]])
    green_list.append([0, color_list[0][1], color_list[0][1]])
    blue_list.append([0, color_list[0][2], color_list[0][2]])

    for i in range(last_col_index):
        t = step*i + half_step
        red_list.append([t, color_list[i][0], color_list[i + 1][0]])
        green_list.append([t, color_list[i][1], color_list[i + 1][1]])
        blue_list.append([t, color_list[i][2], color_list[i + 1][2]])
    
    red_list.append([1, color_list[last_col_index][0], color_list[last_col_index][0]])
    green_list.append([1, color_list[last_col_index][1], color_list[last_col_index][1]])
    blue_list.append([1, color_list[last_col_index][2], color_list[last_col_index][2]])

    color_dict = { 'red' : red_list, 
                   'green': green_list,
                   'blue': blue_list
                 }

    return clrs.LinearSegmentedColormap(name, color_dict)



CM_FRACTAL = computeColorMapFromColorList([(0,0,0),
                                            (0,0,0.5),
                                            (0,0,1),
                                            (0.5,0,1),
                                            (1,0,1),
                                            (1,0.6,1),
                                            (1,1,1)])

CM_FRACTAL_PARA = computeColorMapFromColorList([(1,0.5,0),
                                            (0,0,0),
                                            (0,0,0.5),
                                            (0,0,1),
                                            (0.5,0,1),
                                            (1,0,1),
                                            (1,0.6,1),
                                            (1,1,1)])

CM_EXTRA_STAGES = computeColorMapFromColorList([(0,0,0),
                                                (0,0,0.5),
                                                (0,0,0.5),
                                                (0.5,0,1),
                                                (0.5,0,1),
                                                (0,1,1),
                                                (0,0.75,0.1),
                                                (0.5,1,0),
                                                (1,1,0),
                                                (1,1,1)])

CM_EXTRA_STAGES_PARA = computeColorMapFromColorList([ (1,0,0),
                                                    (0,0,0),
                                                    (0,0,0.5),
                                                    (0,0,0.5),
                                                    (0.5,0,1),
                                                    (0.5,0,1),
                                                    (0,1,1),
                                                    (0,0.75,0.1),
                                                    (0.5,1,0),
                                                    (1,1,0),
                                                    (1,1,1)])

CM_EXTRA_STAGES_ORIG = computeColorMapFromColorList([(0,0,0),
                                                    (0,0,0.5),
                                                    (0,0,1),
                                                    (0.5,0,1),
                                                    (1,0,1),
                                                    (1,0.6,1),
                                                    (1,0,0),
                                                    (1,0.5,0),
                                                    (1,1,0),
                                                    (1,1,1)])

CM_HWR_BANDS = clrs.LinearSegmentedColormap('hwrbands', {'red':  [[0.0,  0.0, 0.0],
                                                                  [0.3,  0.0, 1.0],
                                                                  [1.0,  1.0, 1.0]],
                                                        'green': [[0.0,  0.0, 0.0],
                                                                  [0.1,  0.0, 1.0],
                                                                  [0.4, 1.0, 0.0],
                                                                  [0.8,  0.0, 1.0],
                                                                  [1.0,  1.0, 1.0]],
                                                        'blue':  [[0.0,  0.0, 0.0],
                                                                  [0.01,  0.0, 1.0],
                                                                  [0.2,  1.0, 0.0],
                                                                  [0.6,  0.0, 1.0],
                                                                  [1.0,  1.0, 1.0]]})

CM_UPWARP_SPEED = clrs.LinearSegmentedColormap('minSpeedColors', {'red':   [[0.0,  0.0, 0.0],
                                                                        [0.0001, 0.0, 1.0],
                                                                        [0.08,  1.0, 1.0],
                                                                        [0.33, 1.0, 1.0],
                                                                        [0.67, 0.0, 0.0],
                                                                        [1.0,  0.0, 0.0]],
                                                                'green': [[0.0,  0.0, 0.0],
                                                                        [0.0001, 0.0, 1.0],
                                                                        [0.08,  1.0, 0.0],
                                                                        [0.33, 1.0, 1.0],
                                                                        [0.67, 1.0, 1.0],
                                                                        [1.0,  0.0, 0.0]],
                                                                'blue':  [[0.0,  0.0, 0.0],
                                                                        [0.0001, 0.0, 1.0],
                                                                        [0.08,  1.0, 0.0],
                                                                        [0.33, 0.0, 0.0],
                                                                        [0.67, 1.0, 1.0],
                                                                        [1.0,  1.0, 1.0]]})

CM_DEFAULT = 'viridis'