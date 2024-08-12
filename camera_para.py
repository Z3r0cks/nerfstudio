import numpy as np

def testbed_camera_json(testbed, j_trans, type=1):
	if type==1:
		# my tiral
		cx, cy = j_trans["cx"], j_trans["cy"]
		h, w = j_trans["h"], j_trans["w"]
		fx, fy = j_trans["fl_x"], j_trans["fl_y"]
		fov_x = np.arctan2(w/2, fx) * 2 * 180 / np.pi
		fov_y = np.arctan2(h/2, fy) * 2 * 180 / np.pi

		testbed.screen_center = np.array([cx/w, cy/h])
		testbed.fov_xy = np.array([fov_x, fov_y])
	else:
		# demo 
		testbed.fov_axis = 1
		testbed.fov = np.arctan2(j_trans['h']/2, j_trans['fl_y']) * 2 * 180 / np.pi
		# testbed.screen_center = np.array([j_trans['cx']/j_trans['w'], j_trans['cy']/j_trans['h']])

