import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F

def get_vc(pose, points):
    ''' 返回相机的 v [batch_size,n_rays,3] 和 c [batch_size,3]'''
    # 1,3
    cam_loc = pose[:, :3, 3]
    # 1,4,4
    p = pose

    batch_size, num_rays, _ = points.shape
    x_cam = points[:, :, 0].view(batch_size, -1)
    y_cam = points[:, :, 1].view(batch_size, -1)
    z_cam = points[:, :, 2].view(batch_size, -1)
    ones_cam = torch.ones_like(z).cuda()
    # 1,4,num_points
    pixel_points_cam = torch.stack((x_cam, y_cam, z_cam, ones_cam), dim=-1).permute(0, 2, 1)

    # 1,num_points,3
    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def get_camera_for_plot(pose):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q

def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    '''
    ray 与半径为 r 的球体的交点距离。
    Input: [n_images,3]; [n_images,n_rays,3]
    Output: [n_images,n_rays,2 (close and far)]; [n_images,n_rays]
    '''
    n_imgs, n_pix, _ = ray_directions.shape
    # [n_images,3,1]
    cam_loc = cam_loc.unsqueeze(-1)
    # [n_images,n_rays]
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    # [n_images,n_rays]
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)
    # [n_images * n_rays]
    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    # [n_images * n_rays,2]
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)
    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    # 下限设为 0.0
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)
    return sphere_intersections, mask_intersect

def get_depth(points, pose):
    ''' Retruns depth from 3D points according to camera pose '''
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(pose).bmm(points_hom)
    depth = points_cam[:, 2, :][:, :, None]
    return depth

