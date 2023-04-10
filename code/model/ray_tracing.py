import torch
import torch.nn as nn
from utils import rend_util

class RayTracing(nn.Module):
    def __init__(self,
                object_bounding_sphere=1.0,
                sdf_threshold=5.0e-5,
                line_search_step=0.5,
                line_step_iters=1,
                sphere_tracing_iters=10,
                n_steps=100,
                n_secant_steps=8,
                ):
        super().__init__()
        # 1.0
        self.object_bounding_sphere = object_bounding_sphere
        # 5.0e-5
        self.sdf_threshold = sdf_threshold
        # 10
        self.sphere_tracing_iters = sphere_tracing_iters
        # 3
        self.line_step_iters = line_step_iters
        # 0.5
        self.line_search_step = line_search_step
        # 100
        self.n_steps = n_steps
        # 8
        self.n_secant_steps = n_secant_steps

    def forward(self, sdf, cam_loc, object_mask, ray_directions):

        # 1,10000
        batch_size, num_pixels, _ = ray_directions.shape
        # ray 与半径为 r 的球体的交点距离
        # 返回 [n_images,n_rays,2 (close and far)]; [n_images,n_rays]
        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)
        # ray 与 model 的交点；max_dis：ray 与球体较远的交点距离
        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)
        # [10000] ray 与 model 相交
        network_object_mask = (acc_start_dis < acc_end_dis)
        # 处理未找到交点的 ray
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            # [n_images,num_rays,2]
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]
            # 在 sampler_min_max 中采样，寻找真正的交点
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )
            # 交点
            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            # t
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            # network object mask
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
              .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        print('----------------------------------------------------------------')

        if not self.training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis

        # 接下来就是对不属于 sampler_mask 的 ray 进行赋值
        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)
        # 属于 obejct_mask 但 network 认为不属于，同时不属于 sampler_mask
        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        # 不属于 obejct_mask，同时不属于 sampler_mask
        out_mask = ~object_mask & ~sampler_mask
        # 与半径为 r 的球体不相交
        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        # 赋值与圆心最近的点
        if mask_left_out.sum() > 0:
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out
        # 与半径为 r 的球体相交，但尚未处理
        mask = (in_mask | out_mask) & mask_intersect
        # 赋值与
        if mask.sum() > 0:
            # 采样起始点
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]
            # 等距采样，找到 sdf 最小的点
            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)
            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        '''
        球面追踪，寻找 ray 与 model 的交点
        '''
        # [n_images,n_rays,2 (close and far),3]
        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        # [n_images * n_rays]
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()
        # [n_images * num_rays,3]
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        # 与球面较近的交点
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        # [n_images * num_rays]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        # 与球面较近的交点距离
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]
        # [n_images * num_rays,3]
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        # 与球面较远的交点
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        # [n_images * num_rays]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        # 与球面较远的交点距离
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()
        # 迭代寻找 surface
        iters = 0
        # [n_images * num_rays]
        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])
        # [n_images * num_rays]
        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])
        while True:
            # [n_images * num_rays]
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0  # 5.0e-5
            # [n_images * num_rays]
            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0
            # 更新 mask
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)
            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1
            # 更新距离
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end
            # 更新点
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            # 更新 sdf
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])
            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])
            # 判断是否穿过物体表面
            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # 后退一步
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]
                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]
                # 再次更新 sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])
                # 再次判断是否穿过物体表面
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        '''
        处理 sphere tracing 失效的点，
        在 sampler_min_max 中采样，寻找真正的交点（如果无交点，填充一个与模型最近的点）
        '''
        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        # [n_images * num_rays,3]
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        # [n_images * num_rays]
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()
        # [1,1,100]
        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)
        # [n_images,num_rays,100]
        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        # [n_images,num_rays,100,3]
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)
        # 需要处理的 ray
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        # [:,100,3]
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        # [:,100]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]
        # 计算 sdf
        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)
        # [:,100]
        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))
        # [:] 找到第一个为负的点
        sampler_pts_ind = torch.argmin(tmp, -1)
        # [n_images * num_rays,3] 赋值交点
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        # [n_images * num_rays] 赋值 dists
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]
        
        # 判断是否为真正的交点
        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        # 处理非交点
        if n_p_out > 0:
            # [:] 找到最小的正数
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            # 赋值距离最近的点
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            # 赋值 dists
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # 更新 net_obj_mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # 计算交点的割线
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # 交点
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            # 交点与 model 的距离
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            # 交点上一个点
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            # 上一个点与 model 的距离
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            # c
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            # v
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            # 真正交点
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        '''
        在 [z_low, z_high] 内寻找真正的交点
        '''
        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        '''
        等距采样，找到 sdf 最小的点
        '''
        # 需要处理的 ray
        n_mask_points = mask.sum()
        # 100
        n = self.n_steps
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        # ray 与球体较远的交点距离
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        # 采样起始点
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        # 等距采样
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis
        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]
        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * \
            mask_rays.unsqueeze(1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]
        return min_mask_points, min_mask_dist
