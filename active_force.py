import numpy as np
from numba import jit

class ActiveForce:
    """
    Active force class
    ------------------

    Calculates the active forces acting on a cell centroid.
    v0 can be:
        - a float (same motility for all cells),
        - a dict {ctype: v0_value},
        - a list/tuple/array of values (indexed by cell type),
        - or a full-length array of length n_cells.
    """

    def __init__(self, tissue, active_params=None):
        assert active_params is not None, "Specify active params"
        self.t = tissue
        self.active_params = active_params
        self.aF = None
        self.orientation = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)

        # --- handle v0 flexible input ---
        v0_param = self.active_params["v0"]

        if isinstance(v0_param, (float, int)):
            # scalar → same motility for all cells
            v0_arr = float(v0_param) * np.ones(self.t.mesh.n_c)

        elif isinstance(v0_param, dict):
            # dictionary mapping type index → motility
            v0_arr = np.zeros(self.t.mesh.n_c)
            for ctype, v0_val in v0_param.items():
                if isinstance(v0_val, (list, tuple, np.ndarray)):
                    # a given cell type can have a range of motility values
                    motilities = np.random.rand(self.t.c_typeN[ctype]) * (float(np.max(v0_val)) - float(np.min(v0_val))) + float(np.min(v0_val))
                    v0_arr[self.t.c_types == int(ctype)] = motilities
                else: 
                    # assign single motility value to all cells of this type
                    v0_arr[self.t.c_types == int(ctype)] = float(v0_val)

        elif isinstance(v0_param, (list, tuple, np.ndarray)):
            # sequence indexed by cell type
            v0_arr = np.zeros(self.t.mesh.n_c)
            # assume ctype indices start at 0, and it is int-like by enumeration
            for ctype, v0_val in enumerate(v0_param):  
                # boolean mask of specifying all indices in c_types equal to ctype, 
                # then assigning the corresponding v0_val
                v0_arr[self.t.c_types == ctype] = float(v0_val)  

        else:
            # assume already full-length array
            v0_arr = np.asarray(v0_param, dtype=np.float64)

        # ensure float64 numpy array
        self.active_params["v0"] = np.asarray(v0_arr, dtype=np.float64)

        # defaults for other active params
        if "angle0" not in self.active_params:
            self.active_params["angle0"] = 0
        if "alpha_dir" not in self.active_params:
            self.active_params["alpha_dir"] = 0

        # initialize first active force
        self.get_active_force()

    def update_active_param(self, param_name, val):
        self.active_params[param_name] = val

    def update_orientation(self, dt):
        """
        Time-step the orientation (angle of velocity).
        """
        self.orientation = _update_persistent_directional_orientation(
            self.orientation,
            self.active_params["Dr"],
            dt,
            self.t.mesh.n_c,
            self.active_params["alpha_dir"],
            self.active_params["angle0"],
        )

    @property
    def orientation_vector(self):
        """
        Convert angle to unit vector (cosθ, sinθ).
        """
        return _vec_from_angle(self.orientation)

    def get_active_force(self):
        """
        Compute active force per cell from orientation vectors and motility v0.
        """
        orientation = np.asarray(self.orientation_vector, dtype=np.float64)
        v0 = np.asarray(self.active_params["v0"], dtype=np.float64)
        self.aF = _get_active_force(orientation, v0)

    def update_active_force(self, dt):
        self.update_orientation(dt)
        self.get_active_force()
        return self.aF


@jit(nopython=True)
def _get_active_force(orientation, v0):
    """
    Compute active forces: v0[i] * orientation[i].
    orientation: (n_cells, 2), v0: (n_cells,)
    """
    n = orientation.shape[0]
    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        out[i, 0] = v0[i] * orientation[i, 0]
        out[i, 1] = v0[i] * orientation[i, 1]
    return out


@jit(nopython=True)
def _update_persistent_random_orientation(orientation, Dr, dt, n_c):
    return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c)) % (np.pi * 2)


@jit(nopython=True)
def _update_persistent_directional_orientation(orientation, Dr, dt, n_c, alpha_dir, angle0):
    return (
        (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c)) % (np.pi * 2)
        - dt * alpha_dir * (((orientation - angle0 + np.pi) % (2 * np.pi)) - np.pi)
    )


@jit(nopython=True)
def _vec_from_angle(vec):
    return np.column_stack((np.cos(vec), np.sin(vec)))

# class ActiveForce:
#     """
#     Active force class
#     ------------------

#     Calculates the active forces acting on a cell centroid. This is traditionally phrased in terms of v0 and Dr, being the fixed velocity and the rotational diffusion of the direction.
#     """

#     def __init__(self, tissue, active_params=None):
#         assert active_params is not None, "Specify active params"
#         self.t = tissue
#         self.active_params = active_params
#         self.aF = None
#         self.orientation = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)
        
#         self.get_active_force()
#         # inside ActiveForce.__init__, after self.get_active_force() call
#         if isinstance(self.active_params["v0"], (float, int)):
#             # same motility for all cells
#             self.active_params["v0"] = float(self.active_params["v0"]) * np.ones(self.t.mesh.n_c)

#         elif isinstance(self.active_params["v0"], dict):
#             # dictionary mapping type index -> motility
#             v0_arr = np.zeros(self.t.mesh.n_c)
#             for ctype, v0_val in self.active_params["v0"].items():
#                 v0_arr[self.t.c_types == int(ctype)] = float(v0_val)
#             self.active_params["v0"] = v0_arr

#         elif isinstance(self.active_params["v0"], (list, tuple, np.ndarray)):
#             # sequence of length = number of cell types
#             v0_arr = np.zeros(self.t.mesh.n_c)
#             for ctype, v0_val in enumerate(self.active_params["v0"]):
#                 v0_arr[self.t.c_types == ctype] = float(v0_val)
#             self.active_params["v0"] = v0_arr
#         self.active_params["v0"] = np.asarray(self.active_params["v0"], dtype=np.float64)

#         # if type(self.active_params["v0"]) is float:
#         #     self.active_params["v0"] = self.active_params["v0"] * np.ones(self.t.mesh.n_c)
#         if "angle0" not in self.active_params:
#             self.active_params["angle0"] = 0
#         if "alpha_dir" not in self.active_params:
#             self.active_params["alpha_dir"] = 0

#     def update_active_param(self, param_name, val):
#         self.active_params[param_name] = val

#     def update_orientation(self, dt):
#         """
#         Time-steps the orientation (angle of velocity) according to the equation outlined in Bi et al PRX.
#         :param dt:
#         :return:
#         """
#         # self.orientation = _update_persistent_random_orientation(self.orientation,
#         #                                                    self.active_params["Dr"],
#         #                                                    dt,
#         #                                                    self.t.mesh.n_c)
#         self.orientation = _update_persistent_directional_orientation(self.orientation,
#                                                            self.active_params["Dr"],
#                                                            dt,
#                                                            self.t.mesh.n_c,
#                                                            self.active_params["alpha_dir"],
#                                                            self.active_params["angle0"],)


#     @property
#     def orientation_vector(self):
#         """
#         Property. Converts angle to a unit vector
#         :return: Unit vector
#         """
#         return _vec_from_angle(self.orientation)

#     # def get_active_force(self):
#     #     """
#     #     Standard SPV model
#     #     :return:
#     #     """
#     #     self.aF = _get_active_force(self.orientation_vector,
#     #                                 self.active_params["v0"])
#     def get_active_force(self):
#         """
#         Compute active force per cell from orientation vectors and motility v0.
#         """
#         orientation = np.asarray(self.orientation_vector, dtype=np.float64)
#         v0 = np.asarray(self.active_params["v0"], dtype=np.float64)
#         self.aF = _get_active_force(orientation, v0)



#     def update_active_force(self, dt):
#         self.update_orientation(dt)
#         self.get_active_force()
#         return self.aF

#     ##but could include other options here...


# @jit(nopython=True)
# def _get_active_force(orientation, v0):
#     return (v0 * orientation.T).T


# @jit(nopython=True)
# def _update_persistent_random_orientation(orientation, Dr, dt, n_c):
#     return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c))%(np.pi*2)

# @jit(nopython=True)
# def _update_persistent_directional_orientation(orientation, Dr, dt, n_c,alpha_dir,angle0):
#     return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c))%(np.pi*2) \
#            - dt*alpha_dir*((orientation-angle0 + np.pi)%(np.pi*2) - np.pi)



# @jit(nopython=True)
# def _vec_from_angle(vec):
#     return np.column_stack((np.cos(vec), np.sin(vec)))
