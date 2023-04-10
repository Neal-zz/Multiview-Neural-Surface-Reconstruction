import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    def forward(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        '''
        只关注 rays 与 model 的交点，
        论文中的公式 3
        '''
        surface_ray_dirs_0 = surface_ray_dirs.detach()
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        # t - 0? 啥事没做
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # c + t * v
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v
