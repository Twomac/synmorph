import _pickle as cPickle
import bz2
import pickle

import numpy as np
from numba import jit

import synmorph.tri_functions as trf
from synmorph.active_force import ActiveForce
from synmorph.force import Force
from synmorph.mesh import Mesh
from synmorph import utils


class Tissue:
    """
    Tissue class
    ------------

    This class sits above the mesh, force, active_force,grn classes and integrates information about geometry and other properties of cells to determine forces.

    This class is used to initialize the mesh, force and active_force classes.

    """

    def __init__(self, tissue_params=None, active_params=None, init_params=None, initialize=True, calc_force=True, meshfile=None,
                 run_options=None,tissue_file=None):

        if tissue_file is None:
            assert tissue_params is not None, "Specify tissue params"
            assert active_params is not None, "Specify active params"
            assert init_params is not None, "Specify init params"
            assert run_options is not None, "Specify run options"
 
            self.tissue_params = tissue_params

            self.init_params = init_params
            self.mesh = None

            self.c_types = None
            self.nc_types = None
            self.c_typeN = None
            self.tc_types, self.tc_typesp, self.tc_typesm = None, None, None

            self.boundary_MEP = None
            self.boundary_LEP = None
            self.boundary_fraction = None

            # boolean if an individual cell is surrounded by only ECM
            self.ejected = None

            if meshfile is None:
                assert init_params is not None, "Must provide initialization parameters unless a previous mesh is parsed"
                if initialize:
                    self.initialize(run_options)
            else:
                self.mesh = Mesh(load=meshfile, run_options=run_options)
                assert self.L == self.mesh.L, "The L provided in the params dict and the mesh file are not the same"

            for par in ["A0", "P0", "kappa_A", "kappa_P"]:
                self.tissue_params[par] = _vectorify(self.tissue_params[par], self.mesh.n_c)

            self.active = ActiveForce(self, active_params)

            if calc_force:
                self.get_forces()
            else:
                self.F = None

            self.time = None

            self.name = None
            self.id = None

        else:
            self.load(tissue_file)

    def set_time(self, time):
        """
        Set the time and date at which the simulation was performed. For logging.
        :param time:
        :return:
        """
        self.time = time

    def update_tissue_param(self, param_name, val):
        """
        Short-cut for updating a tissue parameter
        :param param_name: dictionary key
        :param val: corresponding value
        :return:
        """
        self.tissue_params[param_name] = val

    def initialize(self, run_options=None):
        """
        Initialize the tissue. Here, the mesh is initialized, and cell types are assigned.
        In the future, this may want to be generalized.

        :param run_options:
        :return:
        """
        self.initialize_mesh(run_options=run_options)
        self.assign_ctypes()

    def initialize_mesh(self, run_options=None):
        """
        Make initial condition. Currently, this is a hexagonal lattice + noise

        Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

        If x is supplied, this is over-ridden

        :param run_options:
        :param L: Domain size/length (np.float32)
        :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
        """

        x = trf.hexagonal_lattice(int(self.L), int(np.ceil(self.L)), noise=self.init_noise, A=self.A0)
        x += 1e-3
        np.argsort(x.max(axis=1))

        x = x[np.argsort(x.max(axis=1))[:int(self.L ** 2 / self.A0)]]
        self.mesh = Mesh(x, self.L, run_options=run_options)

    # def assign_ctypes(self):
    #     assert sum(self.c_type_proportions) == 1.0, "c_type_proportions must sum to 1.0"
    #     assert (np.array(self.c_type_proportions) >= 0).all(), "c_type_proportions values must all be >=0"
    #     self.nc_types = len(self.c_type_proportions)
    #     self.c_typeN = [int(pr * self.mesh.n_c) for pr in self.c_type_proportions[:-1]]
    #     self.c_typeN += [self.mesh.n_c - sum(self.c_typeN)]

    #     c_types = np.zeros(self.mesh.n_c, dtype=np.int32)
    #     j = 0
    #     for k, ctN in enumerate(self.c_typeN):
    #         c_types[j:j + ctN] = k
    #         j += ctN
    #     np.random.shuffle(c_types)
    #     self.c_types = c_types
    #     self.c_type_tri_form()
    
    def assign_ctypes(self):
        """
        Assign cell types.
        Modes:
        - "random" (default): assign by c_type_proportions (existing behavior)
        - "ball_by_radius": all cells within a radius r of domain center are assigned
            among 'ball_non_ecm_types' by 'ball_non_ecm_proportions'; all others are 'ecm_type'.
        """
        init_mode = self.init_params.get("init_mode", "random")

        if init_mode != "ball_by_radius":
            # --- Original random assignment ---
            assert sum(self.c_type_proportions) == 1.0, "c_type_proportions must sum to 1.0"
            assert (np.array(self.c_type_proportions) >= 0).all(), "c_type_proportions values must all be >= 0"
            self.nc_types = len(self.c_type_proportions)
            # integer counts except last gets the remainder to ensure total
            self.c_typeN = [int(pr * self.mesh.n_c) for pr in self.c_type_proportions[:-1]]
            self.c_typeN += [self.mesh.n_c - sum(self.c_typeN)]

            c_types = np.zeros(self.mesh.n_c, dtype=np.int32)
            j = 0
            for k, ctN in enumerate(self.c_typeN):
                c_types[j:j + ctN] = k
                j += ctN
            np.random.shuffle(c_types)
            self.c_types = c_types
            self.c_type_tri_form()
            return

        # --- New: radius-based ball initializer ---
        # Number of declared types comes from proportions length (kept for consistency)
        self.nc_types = len(self.c_type_proportions)
        assert self.nc_types >= 2, "Need at least two cell types for ball initialization."
        L = self.L

        # Parameters (all optional; sensible defaults provided)
        ball_radius = float(self.init_params.get("ball_radius", L/4))
        # By default, treat the last type index as ECM (e.g., 2 for three types 0,1,2)
        ecm_type = int(self.init_params.get("ecm_type", self.nc_types - 1))
        assert 0 <= ecm_type < self.nc_types, "ecm_type must be a valid type index"

        # Non-ECM types used inside the ball: by default, all types except ECM
        default_non_ecm = tuple(i for i in range(self.nc_types) if i != ecm_type)
        ball_non_ecm_types = tuple(self.init_params.get("ball_non_ecm_types", default_non_ecm))  # also just 0 & 1 as of now to distinguish LEPs and MEPs
        assert len(ball_non_ecm_types) >= 1, "ball_non_ecm_types must include at least one non-ECM type"
        for t in ball_non_ecm_types:
            assert 0 <= t < self.nc_types and t != ecm_type, "ball_non_ecm_types must be valid, non-ECM indices"

        # Proportions for non-ECM split inside the ball (defaults to equal split)
        if "ball_non_ecm_proportions" in self.init_params:
            ball_props = np.array(self.init_params["ball_non_ecm_proportions"], dtype=float)
            assert len(ball_props) == len(ball_non_ecm_types), \
                "ball_non_ecm_proportions length must match ball_non_ecm_types"
            assert (ball_props >= 0).all() and np.isclose(ball_props.sum(), 1.0), \
                "ball_non_ecm_proportions must be non-negative and sum to 1"
        else:
            ball_props = np.ones(len(ball_non_ecm_types), dtype=float) / len(ball_non_ecm_types)  # equal split if not specified

        # Compute distances from center and mask inside the ball
        coords = self.mesh.x
        center = np.array([L/2, L/2], dtype=coords.dtype)
        dists = np.linalg.norm(coords - center, axis=1)  # numpy automatically subtracts [L/2,L/2] from each row of coordinates resepctively
        inside = dists < ball_radius
        n_inside = int(inside.sum())

        c_types = np.full(self.mesh.n_c, ecm_type, dtype=np.int32)  # ECM everywhere by default

        if n_inside > 0:
            # Assign inside cells according to ball_props over ball_non_ecm_types
            inside_idx = np.where(inside)[0]
            np.random.shuffle(inside_idx)

            # turn proportions into integer counts that sum to n_inside
            counts = (ball_props * n_inside).astype(int)
            # give any remainder to the largest-proportion bins
            remainder = n_inside - counts.sum()
            if remainder > 0:
                # distribute remainder to types with largest fractional parts
                fracs = ball_props * n_inside - (ball_props * n_inside).astype(int)
                order = np.argsort(-fracs)  # descending fractional part
                for j in range(remainder):
                    counts[order[j % len(order)]] += 1

            start = 0
            for t, cnt in zip(ball_non_ecm_types, counts):
                if cnt > 0:
                    sel = inside_idx[start:start + cnt]
                    c_types[sel] = int(t)
                    start += cnt

        # Save and record counts per type
        self.c_types = c_types
        # c_typeN matches histogram order 0..nc_types-1
        self.c_typeN = [int((self.c_types == k).sum()) for k in range(self.nc_types)]

        # Update triangulated form
        self.c_type_tri_form()

    def c_type_tri_form(self):
        """
        Convert the nc x 1 c_type array to a nv x 3 array -- triangulated form.
        Here, the CW and CCW (p,m) cell types can be easily deduced by the roll function.
        :return:
        """
        self.tc_types = trf.tri_call(self.c_types, self.mesh.tri)
        self.tc_typesp = trf.roll(self.tc_types, -1)
        self.tc_typesm = trf.roll(self.tc_types, 1)

    def get_forces(self):
        """
        Calculate the forces by calling the Force class.
        :return:
        """
        self.F = Force(self).F
        return sum_forces(self.F, self.active.aF)

    def update(self, dt):
        """
        Wrapper for update functions.
        :param dt: time-step.
        :return:
        """
        self.update_active(dt)
        self.update_mechanics()

    def update_active(self, dt):
        """
        Wrapper for update of active forces
        :param dt: time-step
        :return:
        """
        self.active.update_active_force(dt)

    def update_mechanics(self):
        """
        Wrapper of update of the mesh. The mesh is retriangulated and the geometric properties are recalculated.

        Then the triangulated form of the cell types are reassigned.
        :return:
        """
        self.mesh.update()
        self.c_type_tri_form()

    def update_x_mechanics(self, x):
        """
        Like update_mechanics, apart from x is explicitly provided.
        :param x:
        :return:
        """
        self.mesh.x = x
        self.update_mechanics()

    def get_boundary_interface_length(self, ctype):

        """
        Get the number of cells of a given type on the boundary with ECM.
        :param ctype: cell type to check
        :return:
        """
        assert ctype != 2, "ECM type cannot be on boundary against itself."

        c_types = self.c_types
        # find the IDs of each cell type
        CELL_IDs = np.where(c_types == ctype)[0]
        ECM_IDs = np.where(c_types == 2)[0]

        # extract the interface length adjacency matrix from the mesh
        int_weighted_adj = self.mesh.get_l_interface()
        # Boolean masks for edges touching the given cell type and ECM
        ECM_mask = np.isin(int_weighted_adj.col, ECM_IDs)
        CELL_mask = np.isin(int_weighted_adj.row, CELL_IDs)
        # Mask for edges connecting cell type <-> ECM
        boundary_edges_mask = ECM_mask & CELL_mask
        # Sum the interface lengths for those edges
        total_interface_length = np.sum(int_weighted_adj.data[boundary_edges_mask])

        return total_interface_length

    def get_boundary_LEP(self):
        self.boundary_LEP = self.get_boundary_interface_length(0)
        return self.boundary_LEP

    def get_boundary_MEP(self):
        self.boundary_MEP = self.get_boundary_interface_length(1)
        return self.boundary_MEP

    def get_boundary_fraction(self):
        """
        Get the fraction of LEP interface in the boundary.
        :return:
        """
        assert self.c_types is not None, "Cell types have not been assigned"
        
        MEP_ECM = self.get_boundary_MEP()
        LEP_ECM = self.get_boundary_LEP()

        self.boundary_fraction = LEP_ECM / (MEP_ECM + LEP_ECM)

        return self.boundary_fraction
    
    import numpy as np

    def find_ejected_cells(self):
        """
        Identify LEP/MEP cells (type 0 or 1) that are entirely surrounded by ECM (type 2),
        using the sparse COO interface adjacency matrix.
        Vectorized: avoids Python loops.

        Returns
        -------
        ejected : np.ndarray[bool]
            Boolean mask, True where a real cell is surrounded only by ECM.
        """
        c_types = self.c_types
        n_cells = len(c_types)
        int_adj = self.mesh.get_l_interface()

        # Identify all edges where at least one end is NOT ECM
        # (because ECM-ECM edges don't matter)
        non_ecm_mask = (c_types[int_adj.row] != 2) | (c_types[int_adj.col] != 2)
        rows = int_adj.row[non_ecm_mask]
        cols = int_adj.col[non_ecm_mask]

        # For each edge, mark cells that have a non-ECM neighbor
        # A real cell will be marked if it touches another real cell
        has_non_ecm_neighbor = np.zeros(n_cells, dtype=bool)
        has_non_ecm_neighbor[rows[c_types[cols] != 2]] = True
        has_non_ecm_neighbor[cols[c_types[rows] != 2]] = True

        # Ejected if:
        # - It's a real cell (type 0 or 1)
        # - It has NO non-ECM neighbor
        ejected = (c_types < 2) & (~has_non_ecm_neighbor)

        self.ejected = ejected
        return ejected

    @property
    def init_noise(self):
        return self.init_params["init_noise"]

    @property
    def c_type_proportions(self):
        return self.init_params["c_type_proportions"]

    @property
    def L(self):
        return self.tissue_params["L"]

    @property
    def A0(self):
        return self.tissue_params["A0"]

    @property
    def P0(self):
        return self.tissue_params["P0"]

    @property
    def kappa_A(self):
        return self.tissue_params["kappa_A"]

    @property
    def kappa_P(self):
        return self.tissue_params["kappa_P"]

    @property
    def W(self):
        return self.tissue_params["W"]

    @property
    def a(self):
        return self.tissue_params["a"]

    @property
    def k(self):
        return self.tissue_params["k"]

    ###More properties, for plotting primarily.

    @property
    def dA(self):
        return self.mesh.A - self.A0

    @property
    def dP(self):
        return self.mesh.P - self.P0

    def get_latex(self, val):
        if val in utils._latex:
            return utils._latex[val]
        else:
            print("No latex conversion in the dictionary.")
            return val

    def save(self, name, id=None, dir_path="", compressed=False):
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_tissue" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_tissue" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))

        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        self.__dict__ = fdict


@jit(nopython=True)
def sum_forces(F, aF):
    return F + aF


@jit(nopython=True)
def _vectorify(x, n):
    return x * np.ones(n)
