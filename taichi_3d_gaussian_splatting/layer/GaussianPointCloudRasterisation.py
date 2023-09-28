from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass

import torch
from dataclass_wizard import YAMLWizard

from ..utils import CameraInfo

class GaussianPointCloudRasterisation(torch.nn.Module):
    @dataclass
    class GaussianPointCloudRasterisationConfig(YAMLWizard):
        near_plane: float = 0.8
        far_plane: float = 1000.
        depth_to_sort_key_scale: float = 100.
        rgb_only: bool = False
        grad_color_factor = 5.
        grad_high_order_color_factor = 1.
        grad_s_factor = 0.5
        grad_q_factor = 1.
        grad_alpha_factor = 20.

    @dataclass
    class GaussianPointCloudRasterisationInput:
        point_cloud: torch.Tensor  # Nx3
        point_cloud_features: torch.Tensor  # NxM
        # (N,), we allow points belong to different objects,
        # different objects may have different camera poses.
        # By moving camera, we can actually handle moving rigid objects.
        # if no moving objects, then everything belongs to the same object with id 0.
        # it shall works better once we also optimize for camera pose.
        point_object_id: torch.Tensor
        point_invalid_mask: torch.Tensor  # N
        camera_info: CameraInfo
        # Kx4, x to the right, y down, z forward, K is the number of objects
        q_pointcloud_camera: torch.Tensor
        # Kx3, x to the right, y down, z forward, K is the number of objects
        t_pointcloud_camera: torch.Tensor
        color_max_sh_band: int = 2

    @dataclass
    class BackwardValidPointHookInput:
        point_id_in_camera_list: torch.Tensor  # M
        grad_point_in_camera: torch.Tensor  # Mx3
        grad_pointfeatures_in_camera: torch.Tensor  # Mx56
        grad_viewspace: torch.Tensor  # Mx2
        magnitude_grad_viewspace: torch.Tensor  # M
        magnitude_grad_viewspace_on_image: torch.Tensor  # HxWx2
        num_overlap_tiles: torch.Tensor  # M
        num_affected_pixels: torch.Tensor  # M
        point_depth: torch.Tensor  # M
        point_uv_in_camera: torch.Tensor  # Mx2

    def __init__(
        self,
        config: GaussianPointCloudRasterisationConfig,
        backward_valid_point_hook: Optional[Callable[[
            BackwardValidPointHookInput], None]] = None,
    ):
        super().__init__()
        self.config = config

        class _module_function(torch.autograd.Function):

            @staticmethod
            def forward(ctx,
                        pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        q_pointcloud_camera,
                        t_pointcloud_camera,
                        camera_info,
                        color_max_sh_band,
                        ):
                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.int8, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                q_camera_pointcloud, t_camera_pointcloud = inverse_SE3_qt_torch(
                    q=q_pointcloud_camera, t=t_pointcloud_camera)
                # Step 1: filter points
                filter_point_in_camera(
                    pointcloud=pointcloud,
                    point_invalid_mask=point_invalid_mask,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_in_camera_mask=point_in_camera_mask,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                )
                point_in_camera_mask = point_in_camera_mask.bool()

                # Get id based on the camera_mask
                point_id_in_camera_list = point_id[point_in_camera_mask].contiguous(
                )
                del point_id
                del point_in_camera_mask

                # Number of points in camera
                num_points_in_camera = point_id_in_camera_list.shape[0]

                # Allocate memory
                point_uv = torch.empty(
                    size=(num_points_in_camera, 2), dtype=torch.float32, device=pointcloud.device)
                point_alpha_after_activation = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)
                point_in_camera = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_uv_conic = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_color = torch.zeros(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_radii = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)

                # Step 2: get 2d features
                generate_point_attributes_in_camera_plane(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    point_id_list=point_id_in_camera_list,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_conic=point_uv_conic,
                    point_alpha_after_activation=point_alpha_after_activation,
                    point_color=point_color,
                    point_radii=point_radii,
                )

                # Step 3: get how many tiles overlapped, in order to allocate memory
                num_overlap_tiles = torch.empty_like(point_id_in_camera_list)
                generate_num_overlap_tiles(
                    num_overlap_tiles=num_overlap_tiles,
                    point_uv=point_uv,
                    point_radii=point_radii,
                    camera_width=camera_info.camera_width,
                    camera_height=camera_info.camera_height,
                )
                # Calculate pre-sum of number_overlap_tiles
                accumulated_num_overlap_tiles = torch.cumsum(
                    num_overlap_tiles, dim=0)
                if len(accumulated_num_overlap_tiles) > 0:
                    total_num_overlap_tiles = accumulated_num_overlap_tiles[-1]
                else:
                    total_num_overlap_tiles = 0
                # The space of each point.
                accumulated_num_overlap_tiles = torch.cat(
                    (torch.zeros(size=(1,), dtype=torch.int32, device=pointcloud.device),
                     accumulated_num_overlap_tiles[:-1]))

                # del num_overlap_tiles

                # 64-bits key
                point_in_camera_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int64, device=pointcloud.device)
                # Corresponding to the original position, the record is the point offset in the frustum (engineering optimization)
                point_offset_with_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int32, device=pointcloud.device)

                # Step 4: calclualte key
                if point_in_camera_sort_key.shape[0] > 0:
                    generate_point_sort_key_by_num_overlap_tiles(
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_radii=point_radii,
                        accumulated_num_overlap_tiles=accumulated_num_overlap_tiles,  # input
                        point_offset_with_sort_key=point_offset_with_sort_key,  # output
                        point_in_camera_sort_key=point_in_camera_sort_key,  # output
                        camera_width=camera_info.camera_width,
                        camera_height=camera_info.camera_height,
                        depth_to_sort_key_scale=self.config.depth_to_sort_key_scale,
                    )

                point_in_camera_sort_key, permutation = point_in_camera_sort_key.sort()
                point_offset_with_sort_key = point_offset_with_sort_key[permutation].contiguous(
                )  # now the point_offset_with_sort_key is sorted by the sort_key
                del permutation

                tiles_per_row = camera_info.camera_width // TILE_WIDTH
                tiles_per_col = camera_info.camera_height // TILE_HEIGHT
                tile_points_start = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                tile_points_end = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                # Find tile's start and end.
                if point_in_camera_sort_key.shape[0] > 0:
                    find_tile_start_and_end(
                        point_in_camera_sort_key=point_in_camera_sort_key,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                    )

                # Allocate space for the image.
                rasterized_image = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, 3, dtype=torch.float32, device=pointcloud.device)
                rasterized_depth = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_accumulated_alpha = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_offset_of_last_effective_point = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                pixel_valid_point_count = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                # print(f"num_points: {pointcloud.shape[0]}, num_points_in_camera: {num_points_in_camera}, num_points_rendered: {point_in_camera_sort_key.shape[0]}")

                # Step 5: render
                if point_in_camera_sort_key.shape[0] > 0:
                    gaussian_point_rasterisation(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_uv_conic=point_uv_conic,
                        point_alpha_after_activation=point_alpha_after_activation,
                        point_color=point_color,
                        rasterized_image=rasterized_image,
                        rgb_only=self.config.rgb_only,
                        rasterized_depth=rasterized_depth,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        pixel_valid_point_count=pixel_valid_point_count)
                ctx.save_for_backward(
                    pointcloud,
                    pointcloud_features,
                    # point_id_with_sort_key is sorted by tile and depth and has duplicated points, e.g. one points is belong to multiple tiles
                    point_offset_with_sort_key,
                    point_id_in_camera_list,  # point_in_camera_id does not have duplicated points
                    tile_points_start,
                    tile_points_end,
                    pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point,
                    num_overlap_tiles,
                    point_object_id,
                    q_pointcloud_camera,
                    q_camera_pointcloud,
                    t_pointcloud_camera,
                    t_camera_pointcloud,
                    point_uv,
                    point_in_camera,
                    point_uv_conic,
                    point_alpha_after_activation,
                    point_color,
                )
                ctx.camera_info = camera_info
                ctx.color_max_sh_band = color_max_sh_band
                # rasterized_image.requires_grad_(True)
                return rasterized_image, rasterized_depth, pixel_valid_point_count

            @staticmethod
            def backward(ctx, grad_rasterized_image, grad_rasterized_depth, grad_pixel_valid_point_count):
                grad_pointcloud = grad_pointcloud_features = grad_q_pointcloud_camera = grad_t_pointcloud_camera = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, \
                        pointcloud_features, \
                        point_offset_with_sort_key, \
                        point_id_in_camera_list, \
                        tile_points_start, \
                        tile_points_end, \
                        pixel_accumulated_alpha, \
                        pixel_offset_of_last_effective_point, \
                        num_overlap_tiles, \
                        point_object_id, \
                        q_pointcloud_camera, \
                        q_camera_pointcloud, \
                        t_pointcloud_camera, \
                        t_camera_pointcloud, \
                        point_uv, \
                        point_in_camera, \
                        point_uv_conic, \
                        point_alpha_after_activation, \
                        point_color = ctx.saved_tensors
                    camera_info = ctx.camera_info
                    color_max_sh_band = ctx.color_max_sh_band
                    grad_rasterized_image = grad_rasterized_image.contiguous()
                    grad_pointcloud = torch.zeros_like(pointcloud)
                    grad_pointcloud_features = torch.zeros_like(
                        pointcloud_features)

                    grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], ), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace_on_image = torch.empty_like(
                        grad_rasterized_image[:, :, :2])

                    in_camera_grad_uv_cov_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_grad_color_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_num_affected_pixels = torch.zeros(
                        size=(point_id_in_camera_list.shape[0],), dtype=torch.int32, device=pointcloud.device)

                    gaussian_point_rasterisation_backward(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics.contiguous(),
                        point_object_id=point_object_id.contiguous(),
                        q_camera_pointcloud=q_camera_pointcloud.contiguous(),
                        t_camera_pointcloud=t_camera_pointcloud.contiguous(),
                        t_pointcloud_camera=t_pointcloud_camera.contiguous(),
                        pointcloud=pointcloud.contiguous(),
                        pointcloud_features=pointcloud_features.contiguous(),
                        tile_points_start=tile_points_start.contiguous(),
                        tile_points_end=tile_points_end.contiguous(),
                        point_offset_with_sort_key=point_offset_with_sort_key.contiguous(),
                        point_id_in_camera_list=point_id_in_camera_list.contiguous(),
                        rasterized_image_grad=grad_rasterized_image.contiguous(),
                        pixel_accumulated_alpha=pixel_accumulated_alpha.contiguous(),
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point.contiguous(),
                        grad_pointcloud=grad_pointcloud.contiguous(),
                        grad_pointcloud_features=grad_pointcloud_features.contiguous(),
                        grad_uv=grad_viewspace.contiguous(),
                        in_camera_grad_uv_cov_buffer=in_camera_grad_uv_cov_buffer.contiguous(),
                        in_camera_grad_color_buffer=in_camera_grad_color_buffer.contiguous(),
                        point_uv=point_uv.contiguous(),
                        point_in_camera=point_in_camera.contiguous(),
                        point_uv_conic=point_uv_conic.contiguous(),
                        point_alpha_after_activation=point_alpha_after_activation.contiguous(),
                        point_color=point_color.contiguous(),
                        need_extra_info=True,
                        magnitude_grad_viewspace=magnitude_grad_viewspace.contiguous(),
                        magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image.contiguous(),
                        in_camera_num_affected_pixels=in_camera_num_affected_pixels.contiguous(),
                    )
                    del tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point
                    grad_pointcloud_features = self._clear_grad_by_color_max_sh_band(
                        grad_pointcloud_features=grad_pointcloud_features,
                        color_max_sh_band=color_max_sh_band)
                    grad_pointcloud_features[:,
                                             :4] *= self.config.grad_q_factor
                    grad_pointcloud_features[:,
                                             4:7] *= self.config.grad_s_factor
                    grad_pointcloud_features[:,
                                             7] *= self.config.grad_alpha_factor

                    # 8, 24, 40 are the zero order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             8] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             24] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             40] *= self.config.grad_color_factor
                    # other coefficients are the higher order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             9:24] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             25:40] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             41:] *= self.config.grad_high_order_color_factor

                    if backward_valid_point_hook is not None:
                        backward_valid_point_hook_input = GaussianPointCloudRasterisation.BackwardValidPointHookInput(
                            point_id_in_camera_list=point_id_in_camera_list,
                            grad_point_in_camera=grad_pointcloud[point_id_in_camera_list],
                            grad_pointfeatures_in_camera=grad_pointcloud_features[
                                point_id_in_camera_list],
                            grad_viewspace=grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace=magnitude_grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                            num_overlap_tiles=num_overlap_tiles,
                            num_affected_pixels=in_camera_num_affected_pixels,
                            point_uv_in_camera=point_uv,
                            point_depth=point_in_camera[:, 2],
                        )
                        backward_valid_point_hook(
                            backward_valid_point_hook_input)
                """_summary_
                pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        q_pointcloud_camera,
                        t_pointcloud_camera,
                        camera_info,
                        color_max_sh_band,

                Returns:
                    _type_: _description_
                """

                return grad_pointcloud, \
                    grad_pointcloud_features, \
                    None, \
                    None, \
                    grad_q_pointcloud_camera, \
                    grad_t_pointcloud_camera, \
                    None, None

        self._module_function = _module_function

    def _clear_grad_by_color_max_sh_band(self, grad_pointcloud_features: torch.Tensor, color_max_sh_band: int):
        if color_max_sh_band == 0:
            grad_pointcloud_features[:, 8 + 1: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 1: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 1: 40 + 16] = 0.
        elif color_max_sh_band == 1:
            grad_pointcloud_features[:, 8 + 4: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 4: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 4: 40 + 16] = 0.
        elif color_max_sh_band == 2:
            grad_pointcloud_features[:, 8 + 9: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 9: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 9: 40 + 16] = 0.
        elif color_max_sh_band >= 3:
            pass
        return grad_pointcloud_features

    def forward(self, input_data: GaussianPointCloudRasterisationInput):
        pointcloud = input_data.point_cloud
        pointcloud_features = input_data.point_cloud_features
        point_invalid_mask = input_data.point_invalid_mask
        point_object_id = input_data.point_object_id
        q_pointcloud_camera = input_data.q_pointcloud_camera
        t_pointcloud_camera = input_data.t_pointcloud_camera
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % TILE_WIDTH == 0
        assert camera_info.camera_height % TILE_HEIGHT == 0
        return self._module_function.apply(
            pointcloud,
            pointcloud_features,
            point_invalid_mask,
            point_object_id,
            q_pointcloud_camera,
            t_pointcloud_camera,
            camera_info,
            color_max_sh_band,
        )
