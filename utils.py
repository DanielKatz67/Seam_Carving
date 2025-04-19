import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from typing import List
import copy

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        h, w, _ = np_img.shape

        # Pad the image with 0.5 to prevent outliers
        rgb_padded = np.pad(np_img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)

        # Convert the image to grayscale
        grey_padded = np.dot(rgb_padded[..., :3], self.gs_weights)

        # Back to the original shape
        grey_image = grey_padded[1:-1, 1:-1].reshape(h, w, 1).astype(np.float32)

        return grey_image

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image """
        # pad the current grayscale
        gs = self.resized_gs[..., 0]  # shape (h, w)
        pad = np.pad(gs, ((1, 1), (1, 1)), mode='constant', constant_values=0.5)  # shape (h+2, w+2)

        # Compute horizontal and vertical gradients
        hor_grad = (pad[1:-1, 2:] - pad[1:-1, :-2])
        vert_grad = (pad[2:, 1:-1] - pad[:-2, 1:-1])

        # gradient magnitude and clip to [0,1]
        grad = np.sqrt(hor_grad ** 2 + vert_grad ** 2)

        grad = np.clip(grad, 0, 1).astype(np.float32)

        return grad


    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map[i, s:] = np.roll(self.idx_map[i, s:], -1)

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    # def paint_seams(self):
    #     for s in self.seam_history:
    #         for i, s_i in enumerate(s):
    #             self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False
    #     cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
    #     self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

    def update_cumm_mask(self):
        """ Updates cumm_mask with current seam """
        seam = self.seam_history[-1]
        for row, col in enumerate(seam):
            col_idx = self.idx_map_h[row, col]
            row_idx = self.idx_map_v[row, col]
            self.cumm_mask[row_idx, col_idx] = False

    def paint_seams(self):
        """ Paints seams according to cumm_mask """
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1, 0, 0])

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)

            if self.vis_seams:
                self.update_cumm_mask()
                self.update_ref_mat()

            self.remove_seam(seam)

        if self.vis_seams:
            self.paint_seams()


    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")

    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        self.w -= 1

        for row, col in enumerate(seam):
            self.mask[row, col] = False  # Mark seam pixel for removal

        self.resized_gs = self.resized_gs[self.mask].reshape((self.h, self.w, 1))
        mask_3d = np.stack([self.mask] * 3, axis=2)
        self.resized_rgb = self.resized_rgb[mask_3d].reshape((self.h, self.w, 3))

    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        rotate = -1 if clockwise else 1  # np.rot90 rotates counter-clockwise by default
        self.resized_gs = np.rot90(self.resized_gs, k=rotate)
        self.resized_rgb = np.rot90(self.resized_rgb, k=rotate)
        self.cumm_mask = np.rot90(self.cumm_mask, k=rotate)
        self.seams_rgb = np.rot90(self.seams_rgb, k=rotate)
        self.E = np.rot90(self.E, rotate)
        self.idx_map_h = np.rot90(self.idx_map_h, rotate)
        self.idx_map_v = np.rot90(self.idx_map_v, -rotate)
        self.h, self.w = self.w, self.h
        self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h


    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.idx_map = self.idx_map_h
        self.seams_removal(num_remove)  # Remove vertical seams

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(clockwise=True)  # Rotate to change horizontal to vertical
        self.seams_removal_vertical(num_remove)  # Remove vertical seams
        self.rotate_mats(clockwise=False)  # Rotate back to original orientation

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")


class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""
    def __init__(self, *args, **kwargs):
        """ GreedySeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.cumm_mask = np.ones(self.gs.shape[:2], dtype=bool)  # (h, w) shape
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        h, w = self.E.shape
        seam = []

        # Start from the top row, choose the column with the smallest energy
        col = np.argmin(self.E[0])
        seam.append(col)

        # Iterate from top to bottom row
        for row in range(1, h):
            col_range = []

            # Add valid neighbors: left, center, right
            if col > 0:
                col_range.append(col - 1)  # left

            col_range.append(col)  # center

            if col < w - 1:
                col_range.append(col + 1)  # right

            # Choose the neighbor with minimum energy
            col = min(col_range, key=lambda c: self.E[row, c])
            seam.append(col)

        return seam


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """
    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.cumm_mask = np.ones(self.gs.shape[:2], dtype=bool)  # (h, w) shape
            self.backtrack_mat = np.zeros_like(self.E, dtype=int)
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        # Step 1: Make sure energy matrix M and the backtracking matrix are up-to-date
        self.init_mats()

        height, width = self.M.shape
        seam_path = np.zeros(height, dtype=np.int32)

        # Step 2: Start from the bottom row and find the column with the minimum energy
        seam_path[-1] = np.argmin(self.M[-1])

        # Step 3: Backtrack the seam path from bottom to top
        # At each step, use the backtracking matrix to find which column in the row above led to the current pixel
        for row in range(height - 2, -1, -1):
            next_col = seam_path[row + 1]                          # column in the row below
            seam_path[row] = self.backtrack_mat[row + 1, next_col]

        return seam_path.tolist()

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        # Initialize
        M = np.zeros_like(self.E)
        m_height, m_width = M.shape
        column_indices = np.arange(m_width)
        C_l, C_v, C_r = self.calc_C()
        M[0] = self.E[0]

        # Calculate the M matrix from top to bottom
        for row in range(1, m_height):
            # Each row of the M matrix is calculated based on the previous row
            M_v = M[row - 1]
            M_left = np.roll(M_v, 1)
            M_right = np.roll(M_v, -1)

            M_l_v_r = np.array([M_left, M_v, M_right])
            C_l_v_r = np.array([C_l[row], C_v[row], C_r[row]])
            sum_M_C_l_v_r = M_l_v_r + C_l_v_r

            # Calculate the minimum cost indices and values
            min_cost_idx = np.argmin(sum_M_C_l_v_r, axis=0)
            min_cost = np.choose(min_cost_idx, sum_M_C_l_v_r)
            # Update the M matrix
            M[row] = self.E[row] + min_cost

            # Update the backtracking matrix : actual column numbers + Ensure columns are within bounds
            actual_col_num = column_indices + min_cost_idx - 1
            actual_col_num = np.clip(actual_col_num, 0, m_width - 1)
            self.backtrack_mat[row] = actual_col_num

        return M

    @NI_decor
    def calc_C(self):
        """ Calculates the matrices C_L, C_V, C_R (forward-looking cost) for the M matrix
        Returns:
            C_L, C_V, C_R matrices (float32) of shape (h, w)
        """
        # Squeeze the greyscale image from (h, w, 1) to (h, w)
        gray_image = self.resized_gs.squeeze()

        # Calculate the cost of the new edges
        left_cost = np.roll(gray_image, 1, axis=1)
        middle_cost = np.roll(gray_image, 1, axis=0)
        right_cost = np.roll(gray_image, -1, axis=1)

        c_v = np.abs(right_cost - left_cost)
        c_l = c_v + np.abs(middle_cost - left_cost)
        c_r = c_v + np.abs(middle_cost - right_cost)

        # Edge Handling (Inf at Borders) - no up-left pixel and no up-right pixel
        c_l[:, 0] = c_r[:, -1] = np.inf

        return c_l, c_v, c_r

    def init_mats(self):
        """  Calculates backtrack_mat, M """
        self.backtrack_mat = np.zeros_like(self.E, dtype=int)
        self.M = self.calc_M()

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
        h, w = M.shape


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    if len(orig_shape) != 2 or len(scale_factors) != 2:
        raise ValueError("orig_shape and scale_factors must be 2D arrays")

    # new height, new width
    return int(orig_shape[0] * scale_factors[0]), int(orig_shape[1] * scale_factors[1])


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    # Check that shapes is 2D
    if len(shapes) != 2:
        raise ValueError("shapes must be a 2D array")

    orig_shape, new_shape = shapes
    if len(orig_shape) != 2 or len(new_shape) != 2:
        raise ValueError("shapes must be a 2D array")

    current_height, current_width = orig_shape
    shape_height, shape_width = new_shape
    vertical_seams_count = current_width - shape_width
    horizontal_seams_count = current_height - shape_height

    vertical_seam_image = copy.deepcopy(seam_img)

    # Width Resizing - Remove Vertical Seams
    if vertical_seams_count > 0:
        vertical_seam_image.seams_removal_vertical(vertical_seams_count)

    # Height Resizing - remove Horizontal Seams
    if horizontal_seams_count > 0:
        vertical_seam_image.seams_removal_horizontal(horizontal_seams_count)

    return vertical_seam_image.resized_rgb


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape

    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]

    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1

    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))

    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)

    return new_image


