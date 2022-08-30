import torch
import torch.nn as nn
import torch.nn.functional as F
import geomloss


class Whitening(nn.Module):
    # Adapted from https://cs231n.github.io/neural-networks-2/
    def __init__(self, num_pcs=None):
        super().__init__()
        self.num_pcs = num_pcs

    def forward(self, x):
        x = x - x.mean(0)
        cov = x.T.matmul(x) / x.shape[0]
        u, s, v = torch.svd(cov)

        # Keep only the first num_components PCs if specified
        num_components = u.shape[1] if self.num_pcs is None else self.num_pcs

        x_rot = x.matmul(u[:, :num_components])
        x_white = x_rot / torch.sqrt(s + 1e-5)

        return x_white


def _compute_dist_mat(a, b=None):
    if b is None:
        return _compute_dist_mat(a, a)
    dist_mat = a[:, None] - b[None]
    # Perform squaring and summation in one step
    return torch.einsum('ijk,ijk->ij', dist_mat, dist_mat)


def _get_mat_upper_triangle(mat):
    rows, cols = torch.triu_indices(*mat.shape)
    return mat[rows, cols]


class ClosestPointLoss(nn.Module):
    def __init__(self, batch_size=10_000, reduce_method="mean"):
        super().__init__()
        self.batch_size = batch_size
        self.reduce_method = reduce_method

    @staticmethod
    def nearest_dists(preds, targs):
        dist = _compute_dist_mat(preds, targs)
        return dist.min(-1)[0]

    def combine(self, dists):
        if self.reduce_method == "mean":
            return dists.mean()
        if self.reduce_method == "sum":
            return dists.sum()
        raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def forward(self, outputs, targets):
        lower, upper = 0, self.batch_size
        dists = []
        while lower < len(outputs):
            dists.append(self.nearest_dists(outputs[lower:upper], targets))
            upper += self.batch_size
            lower += self.batch_size
        return self.combine(torch.cat(dists))


class RegLoss(nn.Module):
    def __init__(self, initial_outputs, whitening_num_pcs=None):
        super().__init__()
        self.whiten = Whitening(num_pcs=whitening_num_pcs)
        self.original_dist_mat_whitened = _compute_dist_mat(self.whiten(initial_outputs))

    def whitened_interpoint_distance_change(self, outputs):
        new_dist_mat_normed = _compute_dist_mat(self.whiten(outputs))
        return torch.nn.SmoothL1Loss()(new_dist_mat_normed, self.original_dist_mat_whitened)

    def forward(self, outputs):
        return self.whitened_interpoint_distance_change(outputs) * 1e4


class PerSurfaceRegLoss(nn.Module):
    """
    For each surface, MSE between the self->self distance mat of the original control points and the
    learnt control points.
    """

    def __init__(self, initial_surfaces, reduce_method="mean"):
        super().__init__()
        self.original_distance_matrices = self.compute_distance_matrices(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    def compute_surf_dist_mat(self, surface):
        all_points = surface.control_points.reshape(-1, 3)
        dists = _get_mat_upper_triangle(_compute_dist_mat(all_points))
        dists = dists - dists.min()
        dists = dists / dists.max()
        return dists

    def compute_distance_matrices(self, surfaces):
        return torch.cat([self.compute_surf_dist_mat(surf) for surf in surfaces])

    def forward(self, surfaces):
        new_dist_mats = self.compute_distance_matrices(surfaces)
        return self.loss_func(new_dist_mats, self.original_distance_matrices)


class SurfaceCentreDistanceLoss(nn.Module):
    def __init__(self, initial_surfaces, reduce_method="mean"):
        super().__init__()
        # self.original_centre_dist_mats = self.compute_surface_centres(initial_surfaces).detach()
        self.original_centres = self.compute_surface_centres(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    def compute_surface_centres(self, surfaces):
        centres = torch.stack([surf().mean() for surf in surfaces], 0)
        return centres
        centres = centres[:, None]
        return _get_mat_upper_triangle(_compute_dist_mat(centres))

    def forward(self, surfaces):
        new_centres = self.compute_surface_centres(surfaces)
        return self.loss_func(new_centres, self.original_centres)


class AltRegLoss(nn.Module):
    def __init__(self, initial_outputs, whitening_num_pcs=None):
        super().__init__()
        self.whiten = Whitening(num_pcs=whitening_num_pcs)
        original_dist_mat_whitened = _compute_dist_mat(self.whiten(initial_outputs))
        self.original_stats = torch.stack((original_dist_mat_whitened.mean(),
                                           original_dist_mat_whitened.std()), 0)

    def statistics_loss(self, outputs):
        new_dist_mat_normed = _compute_dist_mat(self.whiten(outputs))
        new_stats = torch.stack((new_dist_mat_normed.mean(),
                                 new_dist_mat_normed.std()), 0)
        return torch.nn.SmoothL1Loss()(new_stats, self.original_stats)

    def forward(self, outputs):
        return self.statistics_loss(outputs) * 1e4


class AreaLoss(nn.Module):
    def __init__(self, initial_surfaces):
        super().__init__()
        self.initial_area = self.compute_areas(initial_surfaces)

    @staticmethod
    def compute_area(surface):
        eval_pts = surface().detach()
        dist_mat = _compute_dist_mat(eval_pts)
        mean_dist = dist_mat.mean()
        return mean_dist

    @staticmethod
    def compute_areas(surfaces):
        all_areas = torch.stack([AreaLoss.compute_area(surf) for surf in surfaces])
        proportional_areas = all_areas / all_areas.max()
        return proportional_areas

    def forward(self, surfaces):
        current_areas = self.compute_areas(surfaces)
        diff = self.initial_area - current_areas
        return (diff ** 2).mean()


class ParameterRegLoss(nn.Module):
    """
    The initial parameters (control points) are normalised, then concatenated and stored.
    The parameters of the surfaces provided in forward have the same done, then the loss is
    MSE between the initial parameters and these new ones.
    """

    def __init__(self, initial_surfaces, reduce_method="mean"):
        super().__init__()
        self.initial_params = self.get_normalised_params(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    def get_normalised_params(self, surfaces):
        out = []
        for surf in surfaces:
            surf_params = surf.control_points.reshape(-1, 3)
            out.append(surf_params)
        out = torch.cat(out, 0)
        out.sub_(out.mean(0))
        out.div_(out.std(0))
        return out

    def forward(self, surfaces):
        params = self.get_normalised_params(surfaces)
        return self.loss_func(params, self.initial_params)


class NeighbourLoss(nn.Module):
    def __init__(self, initial_surfaces, num_neighbours=3, reduce_method="mean"):
        super().__init__()
        self.neighbour_dist_indices = self.get_neighbour_indices(
            initial_surfaces, num_neighbours).detach()
        self.original_neighbour_distances = self.get_neighbour_distances(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    def get_neighbour_indices(self, surfaces, num_neighbours):
        points = torch.cat([surf() for surf in surfaces], 0)
        # points = torch.cat([surf.control_points.reshape(-1, 3) for surf in surfaces], 0)
        # print(points.shape)
        dists = _compute_dist_mat(points)
        # Make the diagonal the greatest distance, so we can disregard self->self distances
        dists.add_(torch.eye(len(dists), device=dists.device)).mul_(dists.max() + 1)
        return torch.topk(dists, num_neighbours, sorted=False, largest=False)[1]

    def get_neighbour_distances(self, surfaces):
        points = torch.cat([surf() for surf in surfaces], 0)
        dists = _compute_dist_mat(points)
        return torch.gather(dists, 1, self.neighbour_dist_indices).reshape(-1)

    def forward(self, surfaces):
        curr_neighbour_distances = self.get_neighbour_distances(surfaces)
        return self.loss_func(curr_neighbour_distances, self.original_neighbour_distances) * 1e5


class TangentialLoss(nn.Module):
    """
    At initialisation, approximate surface normal vectors by computing the cross-product of the
    vectors made from three corner control points of each surface. Use these as a set of basis
    vectors, so points can be decomposed into 2 tangential components and one normal component. To
    compute this, we build a change of basis transform matrix for each surface. The initial surface
    control points are transformed into this space, then the normal (z) component is discarded,
    retaining only the tangential coordinates.
    The loss is the the MSE between the original tangential coordinates and those of the control
    points during optimisation.
    """

    def __init__(self, initial_surfaces, reduce_method="mean"):
        super().__init__()
        self.mats = tuple(mat.detach() for mat in self.get_change_of_basis_mats(initial_surfaces))
        self.initial_tangential_coords = self.change_bases(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    @staticmethod
    def change_of_basis_mat(tri):
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = v0.cross(v1)
        return F.normalize(torch.stack((v0, v1, v2), 1), dim=0)

    @staticmethod
    def get_change_of_basis_mat(surf):
        tri = (
            surf.control_points[0, 0].detach(),
            surf.control_points[0, -1].detach(),
            surf.control_points[-1, 0].detach(),
        )
        return TangentialLoss.change_of_basis_mat(tri)

    def get_change_of_basis_mats(self, surfaces):
        return [self.get_change_of_basis_mat(surf) for surf in surfaces]

    def change_bases(self, surfaces):
        out_points = []
        for surf, mat in zip(surfaces, self.mats):
            points = surf.control_points.reshape(-1, 3)
            cob_points = torch.einsum("ij,jk->ik", points, mat)
            out_points.append(cob_points[..., :2])
        return torch.cat(out_points, 0)

    def forward(self, surfaces):
        curr_tangential_coords = self.change_bases(surfaces)
        return self.loss_func(curr_tangential_coords, self.initial_tangential_coords)


class TriangleNormalLoss(nn.Module):
    def __init__(self, initial_surfaces, ignore_rotations=False, reduce_method="mean"):
        super().__init__()
        self.ignore_rotations = ignore_rotations
        self.initial_tri_normals = self.compute_tri_normals(initial_surfaces).detach()
        self.loss_func = nn.MSELoss(reduction=reduce_method)

    def compute_tri_normals(self, surfaces):
        surf_normals = []
        for surf in surfaces:
            normals = self.compute_tri_normals_single(surf)
            if self.ignore_rotations:
                mat = TangentialLoss.get_change_of_basis_mat(surf)
                normals = torch.einsum("ij,jk->ik", normals, mat)
            surf_normals.append(normals)
        surf_normals = torch.cat(surf_normals)
        return F.normalize(surf_normals, dim=-1)

    def compute_tri_normals_single(self, surf):
        import numpy as np
        data_grid = surf.control_points
        data = data_grid.reshape(-1, 3)
        rows, cols = data_grid.shape[: 2]

        all_indices = np.arange(rows * cols)
        all_indices_grid = all_indices.reshape(rows, cols)

        first_row_indices = all_indices_grid[0, :].reshape(-1)
        last_col_indices = all_indices_grid[:, -1].reshape(-1)
        last_row_indices = all_indices_grid[-1, :].reshape(-1)
        top_left_indices = np.setdiff1d(np.setdiff1d(all_indices, last_col_indices),
                                        last_row_indices)
        top_right_indices = top_left_indices + 1
        bot_left_indices = np.setdiff1d(np.setdiff1d(all_indices, last_col_indices),
                                        first_row_indices)
        bot_right_indices = bot_left_indices + 1

        left_tri_indices = np.stack((top_left_indices, top_right_indices, bot_left_indices), 1)
        right_tri_indices = np.stack((top_right_indices, bot_right_indices, bot_left_indices), 1)

        left_tri_indices = left_tri_indices.reshape(-1)
        left_tris = data[left_tri_indices].reshape(-1, 3, 3)
        right_tri_indices = right_tri_indices.reshape(-1)
        right_tris = data[right_tri_indices].reshape(-1, 3, 3)

        def get_surface_normals(tris):
            b0 = tris[:, 2] - tris[:, 0]
            b1 = tris[:, 2] - tris[:, 1]
            b2 = b0.cross(b1)
            normalised = F.normalize(torch.stack((b0, b1, b2), 1), dim=-1)
            return normalised[:, 2]

        left_normals = get_surface_normals(left_tris)
        right_normals = get_surface_normals(right_tris)
        return torch.cat((left_normals, right_normals))

    def forward(self, surfaces):
        curr_tri_normals = self.compute_tri_normals(surfaces)
        return self.loss_func(curr_tri_normals, self.initial_tri_normals)


class SimpleDistanceLoss(nn.Module):
    """The summed L2 distance from each point to its nearest"""
    """Not possible due to memory constraints"""

    def forward(self, from_points, to_points):
        dist_mat = _compute_dist_mat(from_points, to_points)
        dist_mat = dist_mat.min(1)[0]
        dist_mat = dist_mat.sum()
        return dist_mat


class DistLoss(nn.Module):
    """BAD"""

    def __init__(self, direction):
        super().__init__()
        if direction == "cloud2nearest":
            self.dim = 0
        else:
            self.dim = 1

    def forward(self, surfaces, targets):
        nearest_dists = []
        for surf in surfaces:
            surf_pts = surf()
            dist = _compute_dist_mat(surf_pts, targets)
            dist = dist.min(self.dim)[0]
            nearest_dists.append(dist)
        nearest_dists = torch.stack(nearest_dists)
        nearest_dists = nearest_dists.min(self.dim)[0]
        return nearest_dists.sum()


SamplesLoss = geomloss.SamplesLoss


class ControlPointsSamplesloss(nn.Module):
    def __init__(self, surfaces):
        super().__init__()
        self.initial_control_points = self.get_control_points(surfaces).detach()
        self.loss_func = SamplesLoss(blur=1, backend="online")

    def get_control_points(self, surfaces):
        out = []
        for surf in surfaces:
            out.append(surf.control_points.reshape(-1, 3))
        return torch.cat(out)

    def forward(self, surfaces):
        curr_ctrl_pts = self.get_control_points(surfaces)
        return self.loss_func(curr_ctrl_pts, self.initial_control_points)


class L2ParameterNorm(nn.Module):
    def forward(self, transform_parameters):
        return (transform_parameters ** 2).sum()


class L1ParameterNorm(nn.Module):
    def forward(self, transform_parameters):
        return transform_parameters.abs().sum()


if __name__ == "__main__":
    import numpy as np
    import torch

    print(SimpleDistanceLoss()(torch.zeros(5, 2), torch.zeros(3, 2)).shape)

    def print_indices(grid, indices):
        rows, cols = grid.shape
        grid = np.zeros(rows * cols)
        for index in indices:
            grid[index] = 1
        print(grid.reshape(rows, cols))

    rows, cols = 4, 5
    grid = np.arange(rows * cols).reshape(rows, cols)
    all_indices = np.arange(rows * cols)
    all_indices_grid = all_indices.reshape(rows, cols)
    first_row_indices = all_indices_grid[0, :].reshape(-1)
    last_col_indices = all_indices_grid[:, -1].reshape(-1)
    last_row_indices = all_indices_grid[-1, :].reshape(-1)
    top_left_indices = np.setdiff1d(np.setdiff1d(all_indices, last_col_indices), last_row_indices)
    top_right_indices = top_left_indices + 1
    bot_left_indices = np.setdiff1d(np.setdiff1d(all_indices, last_col_indices), first_row_indices)
    bot_right_indices = bot_left_indices + 1

    left_tri_indices = np.stack((top_left_indices, top_right_indices, bot_left_indices), 1)
    right_tri_indices = np.stack((top_right_indices, bot_right_indices, bot_left_indices), 1)

    data_grid = torch.randint(0, 10, (rows, cols, 3))
    data = data_grid.reshape(-1, 3)
    left_tri_indices = left_tri_indices.reshape(-1)
    left_tris = data[left_tri_indices].reshape(-1, 3, 3)
    right_tri_indices = right_tri_indices.reshape(-1)
    right_tris = data[right_tri_indices].reshape(-1, 3, 3)

    def to_basis_vectors(tris):
        b0 = tris[:, 2] - tris[:, 0]
        b1 = tris[:, 2] - tris[:, 1]
        b2 = b0.cross(b1)
        return b0, b1, b2
    _, _, left_normals = to_basis_vectors(left_tris)
    _, _, right_normals = to_basis_vectors(right_tris)
    normals = torch.cat((left_normals, right_normals))
