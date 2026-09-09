'''
Live Record3D mapping. Same detection/mapping logic as rerun_realtime_mapping.py.

ingest=usb: this process owns the USB stream.
ingest=ros: subscribe to record3d_ros_publisher; LatestFrameSlot drops stale frames.
'''

import os
import signal
import time
import pickle
import gzip
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
import supervision as sv

from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun,
    orr_log_annotated_image,
    orr_log_camera,
    orr_log_depth_image,
    orr_log_edges,
    orr_log_objs_pcd_and_bbox,
    orr_log_rgb_image,
    orr_log_vlm_image,
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    get_exp_out_path,
    get_stream_data_out_path,
    get_vlm_annotated_image_path,
    handle_rerun_saving,
    load_saved_detections,
    measure_time,
    save_detection_results,
    save_edge_json,
    save_hydra_config,
    save_obj_json,
    save_objects_for_frame,
    save_pointcloud,
    should_exit_early,
    vis_render_image,
)
from conceptgraph.utils.vis import (
    OnlineObjectRenderer,
    vis_result_fast_on_depth,
    vis_result_fast,
    save_video_detections,
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    process_cfg,
    process_edges,
    processing_needed,
    resize_gobs,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections
from conceptgraph.utils.detector_backends import init_detector, run_detector
from conceptgraph.slam.frame_source import UsbRecord3DFrameSource

torch.set_grad_enabled(False)


def empty_clip_outputs():
    return [], np.empty((0, 0), dtype=np.float32), []


def prompt_yes_no(message: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(message + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="r3d_stream_rerun_realtime_mapping")
def main(cfg: DictConfig):
    ingest = str(getattr(cfg, "ingest", "usb"))
    app = None
    if ingest != "ros":
        from conceptgraph.utils.record3d_utils import DemoApp

        app = DemoApp()

    tracker = MappingTracker()

    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", config=cfg_to_dict(cfg))
    cfg = process_cfg(cfg)

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param=view_param,
            base_objects=None,
            gray_map=False,
        )
        frames = []

    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg["classes_file"],
        bg_classes=detections_exp_cfg["bg_classes"],
        skip_bg=detections_exp_cfg["skip_bg"],
    )

    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    stream_rgb_path, stream_depth_path, stream_poses_path = get_stream_data_out_path(
        cfg.dataset_root, cfg.scene_id
    )

    prev_adjusted_pose = None
    detection_model = None
    sam_predictor = None
    detector_backend = str(cfg.detector)

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)
        detector_backend, detection_model, sam_predictor, detector_class_names = init_detector(
            cfg, obj_classes.get_classes_arr()
        )
        if detector_backend != "yoloe":
            obj_classes.set_class_list(detector_class_names)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.eval().to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        torch.cuda.empty_cache()
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = (
            exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        )
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    loop_start_time = time.time()
    frame_times = []
    total_frames = 15000
    stop_requested = False

    def on_sigint(signum, _frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\nCtrl+C — stopping after this frame. Press again to abort.")

    def log_frame_timing(frame_idx, frame_start_time):
        frame_time_s = time.time() - frame_start_time
        fps = 1.0 / frame_time_s if frame_time_s > 0 else 0.0
        frame_times.append(frame_time_s)
        print(f"Frame {frame_idx}: frame_time_s={frame_time_s:.3f}, fps={fps:.3f}")
        return frame_time_s, fps

    def run_final_cleanup():
        nonlocal objects, map_edges
        print("Performing final denoise / filter / merge...")
        if cfg["run_denoise_final_frame"]:
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                objects=objects,
            )
        if cfg["run_filter_final_frame"]:
            objects = filter_objects(
                obj_min_points=cfg["obj_min_points"],
                obj_min_detections=cfg["obj_min_detections"],
                objects=objects,
                map_edges=map_edges,
            )
        if cfg["run_merge_final_frame"]:
            merged = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges,
            )
            if cfg["make_edges"]:
                objects, map_edges = merged
            else:
                objects = merged

    def persist_scene():
        nonlocal objects
        for obj in objects:
            obj["consolidated_caption"] = ""
        handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)
        if cfg.save_pcd:
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True,
                edges=map_edges,
            )
        if cfg.save_json:
            save_obj_json(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                objects=objects,
            )
            save_edge_json(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                objects=objects,
                edges=map_edges,
            )
        if cfg.save_objects_all_frames:
            save_meta_path = obj_all_frames_out_path / "meta.pkl.gz"
            with gzip.open(save_meta_path, "wb") as f:
                pickle.dump({
                    "cfg": cfg,
                    "class_names": obj_classes.get_classes_arr(),
                    "class_colors": obj_classes.get_class_color_dict_by_index(),
                }, f)
        if run_detections and cfg.save_video:
            save_video_detections(det_exp_path)
        print(f"Saved scene ({len(objects)} objects) to {exp_out_path}")

    prev_sigint = signal.signal(signal.SIGINT, on_sigint)
    if ingest == "ros":
        from conceptgraph.slam.ros_frame_source import RosRgbDPoseSource

        frame_source = RosRgbDPoseSource(
            rgb_topic=str(getattr(cfg, "ros_rgb_topic", "/record3d/color/image_raw")),
            depth_topic=str(getattr(cfg, "ros_depth_topic", "/record3d/depth/image_raw")),
            camera_info_topic=str(
                getattr(cfg, "ros_camera_info_topic", "/record3d/color/camera_info")
            ),
            pose_topic=str(getattr(cfg, "ros_pose_topic", "/record3d/pose")),
        )
    else:
        frame_source = UsbRecord3DFrameSource(app, dev_idx=0)

    print(f"Live mapping: {total_frames} frames max, or Ctrl+C to stop and choose whether to save.")
    print(f"Starting frame source ingest={ingest} (after models loaded)...")
    frame_source.start()

    try:
        for frame_idx in trange(total_frames):
            if stop_requested:
                break
            frame_start_time = time.time()
            tracker.curr_frame_idx = frame_idx
            counter += 1
            orr.set_time("frame", sequence=frame_idx)
            is_final_frame = (frame_idx == total_frames - 1) or stop_requested

            if not exit_early_flag and should_exit_early(cfg.exit_early_file):
                print("Exit early signal detected. Skipping to the final frame...")
                exit_early_flag = True

            if exit_early_flag and frame_idx < total_frames - 1:
                log_frame_timing(frame_idx, frame_start_time)
                continue

            frame = None
            waits = 0
            while frame is None and not stop_requested:
                frame = frame_source.next(timeout=1.0)
                if frame is None:
                    waits += 1
                    if waits == 1 or waits % 5 == 0:
                        print(
                            f"Waiting for frame ({waits}s, ingest={ingest}). "
                            "USB: Record3D streaming, one client. "
                            "ROS: publisher running on /record3d/*."
                        )
            if stop_requested:
                break

            s_rgb = frame.rgb
            s_depth = frame.depth
            s_camera_pose = frame.pose
            s_intrinsic_mat = frame.intrinsics_4x4()
            img_h, img_w = s_rgb.shape[:2]
            cfg.image_height = int(img_h)
            cfg.image_width = int(img_w)
            cam_K = np.asarray(frame.K, dtype=np.float32)
            if frame_idx == 0:
                print(
                    f"Live camera  ingest={ingest}  rgb={s_rgb.shape} depth={s_depth.shape}\n"
                    f"K=\n{cam_K}\n"
                    f"detector={detector_backend}  make_edges={cfg.make_edges}"
                )
            elif frame.dropped_before and frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: dropped {frame.dropped_before} stale frames so far")

            curr_stream_rgb_path = stream_rgb_path / f"{frame_idx}.jpg"
            cv2.imwrite(str(curr_stream_rgb_path), s_rgb)
            color_path = curr_stream_rgb_path

            if cfg.save_detections:
                curr_stream_depth_path = stream_depth_path / f"{frame_idx}.png"
                cv2.imwrite(str(curr_stream_depth_path), s_depth)
                curr_stream_pose_path = stream_poses_path / f"{frame_idx}.npz"
                np.savez(str(curr_stream_pose_path), s_camera_pose)

            image_original_pil = Image.open(color_path)
            color_tensor = torch.from_numpy(s_rgb.astype("float32"))
            depth_tensor = torch.from_numpy(s_depth.astype("float32"))
            intrinsics = s_intrinsic_mat
            depth_array = depth_tensor.cpu().numpy()
            color_np = color_tensor.cpu().numpy()
            image_rgb = color_np.astype(np.uint8)
            assert image_rgb.max() > 1, "Image is not in range [0, 255]"

            vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
            vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(
                det_exp_vis_path, color_path, w_edges=True
            )

            if run_detections:
                image = cv2.imread(str(color_path))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                xyxy_np, confidences, detection_class_ids, masks_np = run_detector(
                    detector_backend,
                    detection_model,
                    sam_predictor,
                    color_path,
                    color_tensor.shape[:2],
                    cfg.detection_conf,
                )
                detection_class_labels = [
                    f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
                    for class_idx, class_id in enumerate(detection_class_ids)
                ]
                curr_det = sv.Detections(
                    xyxy=xyxy_np,
                    confidence=confidences,
                    class_id=detection_class_ids,
                    mask=masks_np,
                )

                # No LLM edges on the live path.
                labels = detection_class_labels
                edges = []
                captions = []
                if len(curr_det.xyxy) == 0:
                    image_crops, image_feats, text_feats = empty_clip_outputs()
                else:
                    image_crops, image_feats, text_feats = compute_clip_features_batched(
                        image_rgb,
                        curr_det,
                        clip_model,
                        clip_preprocess,
                        clip_tokenizer,
                        obj_classes.get_classes_arr(),
                        cfg.device,
                    )

                tracker.increment_total_detections(len(curr_det.xyxy))
                results = {
                    "xyxy": curr_det.xyxy,
                    "confidence": curr_det.confidence,
                    "class_id": curr_det.class_id,
                    "mask": curr_det.mask,
                    "classes": obj_classes.get_classes_arr(),
                    "image_crops": image_crops,
                    "image_feats": image_feats,
                    "text_feats": text_feats,
                    "detection_class_labels": detection_class_labels,
                    "labels": labels,
                    "edges": edges,
                    "captions": captions,
                }
                raw_gobs = results

                if cfg.save_detections:
                    vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                    annotated_image, labels = vis_result_fast(
                        image, curr_det, obj_classes.get_classes_arr()
                    )
                    cv2.imwrite(str(vis_save_path), annotated_image)
                    depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image_rgb = depth_image_rgb.astype(np.uint8)
                    depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                    annotated_depth_image, labels = vis_result_fast_on_depth(
                        depth_image_rgb, curr_det, obj_classes.get_classes_arr()
                    )
                    cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                    cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                    save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
            else:
                if os.path.exists(det_exp_pkl_path / color_path.stem):
                    raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
                elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                    raw_gobs = load_saved_detections(
                        det_exp_pkl_path / f"{int(color_path.stem):06}"
                    )
                else:
                    raise FileNotFoundError(
                        f"No detections found for frame {frame_idx} at paths "
                        f"\n{det_exp_pkl_path / color_path.stem} or "
                        f"\n{det_exp_pkl_path / f'{int(color_path.stem):06}'}."
                    )
                raw_gobs.setdefault("edges", [])
                raw_gobs.setdefault("captions", [])

            adjusted_pose = s_camera_pose
            prev_adjusted_pose = orr_log_camera(
                intrinsics, adjusted_pose, prev_adjusted_pose, img_w, img_h, frame_idx
            )
            orr_log_rgb_image(color_path)
            orr_log_annotated_image(color_path, det_exp_vis_path)
            orr_log_depth_image(depth_tensor)
            if os.path.exists(vis_save_path_for_vlm):
                orr_log_vlm_image(vis_save_path_for_vlm)
            if os.path.exists(vis_save_path_for_vlm_edges):
                orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

            resized_gobs = resize_gobs(raw_gobs, image_rgb)
            filtered_gobs = filter_gobs(
                resized_gobs,
                image_rgb,
                skip_bg=cfg.skip_bg,
                BG_CLASSES=obj_classes.get_bg_classes_arr(),
                mask_area_threshold=cfg.mask_area_threshold,
                max_bbox_area_ratio=cfg.max_bbox_area_ratio,
                mask_conf_threshold=cfg.mask_conf_threshold,
            )
            gobs = filtered_gobs
            gobs.setdefault("edges", [])

            if len(gobs["mask"]) == 0:
                log_frame_timing(frame_idx, frame_start_time)
                continue

            gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

            obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
                depth_array=depth_array,
                masks=gobs["mask"],
                cam_K=cam_K,
                image_rgb=image_rgb,
                trans_pose=adjusted_pose,
                min_points_threshold=cfg.min_points_threshold,
                spatial_sim_type=cfg.spatial_sim_type,
                obj_pcd_max_points=cfg.obj_pcd_max_points,
                device=cfg.device,
            )

            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(
                        pcd=obj["pcd"],
                        downsample_voxel_size=cfg["downsample_voxel_size"],
                        dbscan_remove_noise=cfg["dbscan_remove_noise"],
                        dbscan_eps=cfg["dbscan_eps"],
                        dbscan_min_points=cfg["dbscan_min_points"],
                    )
                    obj["bbox"] = get_bounding_box(
                        spatial_sim_type=cfg["spatial_sim_type"],
                        pcd=obj["pcd"],
                    )

            detection_list = make_detection_list_from_pcd_and_gobs(
                obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
            )

            if len(detection_list) == 0:
                log_frame_timing(frame_idx, frame_start_time)
                continue

            if len(objects) == 0:
                objects.extend(detection_list)
                tracker.increment_total_objects(len(detection_list))
                owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
                log_frame_timing(frame_idx, frame_start_time)
                continue

            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg["spatial_sim_type"],
                detection_list=detection_list,
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
            )
            visual_sim = compute_visual_similarities(detection_list, objects)
            agg_sim = aggregate_similarities(
                match_method=cfg["match_method"],
                phys_bias=cfg["phys_bias"],
                spatial_sim=spatial_sim,
                visual_sim=visual_sim,
            )
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=cfg["sim_threshold"],
            )
            objects = merge_obj_matches(
                detection_list=detection_list,
                objects=objects,
                match_indices=match_indices,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
            )

            for obj in objects:
                curr_obj_class_id_counter = Counter(obj["class_id"])
                most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
                if obj["class_name"] != most_common_class_name:
                    obj["class_name"] = most_common_class_name

            if cfg.make_edges and gobs.get("edges"):
                map_edges = process_edges(
                    match_indices, gobs, len(objects), objects, map_edges, frame_idx
                )
                edges_to_delete = []
                for curr_map_edge in map_edges.edges_by_index.values():
                    if (frame_idx - curr_map_edge.first_detected > 5) and curr_map_edge.num_detections < 2:
                        edges_to_delete.append((curr_map_edge.obj1_idx, curr_map_edge.obj2_idx))
                for edge in edges_to_delete:
                    map_edges.delete_edge(edge[0], edge[1])

            if is_final_frame:
                print("Final frame detected. Performing final post-processing...")

            if processing_needed(
                cfg["denoise_interval"],
                cfg["run_denoise_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                objects = measure_time(denoise_objects)(
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    objects=objects,
                )

            if processing_needed(
                cfg["filter_interval"],
                cfg["run_filter_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                objects = filter_objects(
                    obj_min_points=cfg["obj_min_points"],
                    obj_min_detections=cfg["obj_min_detections"],
                    objects=objects,
                    map_edges=map_edges,
                )

            if processing_needed(
                cfg["merge_interval"],
                cfg["run_merge_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                merged = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=cfg["make_edges"],
                    map_edges=map_edges,
                )
                if cfg["make_edges"]:
                    objects, map_edges = merged
                else:
                    objects = merged

            orr_log_objs_pcd_and_bbox(objects, obj_classes)
            if cfg.make_edges:
                orr_log_edges(objects, map_edges, obj_classes)

            if cfg.save_objects_all_frames:
                save_objects_for_frame(
                    obj_all_frames_out_path,
                    frame_idx,
                    objects,
                    cfg.obj_min_detections,
                    adjusted_pose,
                    color_path,
                )

            if cfg.vis_render:
                vis_render_image(
                    objects,
                    obj_classes,
                    obj_renderer,
                    image_original_pil,
                    adjusted_pose,
                    frames,
                    frame_idx,
                    color_path,
                    cfg.obj_min_detections,
                    cfg.class_agnostic,
                    cfg.debug_render,
                    is_final_frame,
                    cfg.exp_out_path,
                    cfg.exp_suffix,
                )

            if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
                save_pointcloud(
                    exp_suffix=cfg.exp_suffix,
                    exp_out_path=exp_out_path,
                    cfg=cfg,
                    objects=objects,
                    obj_classes=obj_classes,
                    latest_pcd_filepath=cfg.latest_pcd_filepath,
                    create_symlink=True,
                )

            frame_time_s, fps = log_frame_timing(frame_idx, frame_start_time)
            tracker.increment_total_objects(len(objects))
            tracker.increment_total_detections(len(detection_list))
            owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                "frame_time_s": frame_time_s,
                "fps": fps,
            })
    finally:
        frame_source.close()

    signal.signal(signal.SIGINT, prev_sigint)

    total_time_s = time.time() - loop_start_time
    num_frames = len(frame_times)
    avg_fps = num_frames / total_time_s if total_time_s > 0 else 0.0
    print(f"Processed {num_frames} frames in {total_time_s:.2f}s, avg_fps={avg_fps:.3f}")

    should_save = True
    if stop_requested:
        should_save = prompt_yes_no(
            f"Save mapped scene ({len(objects)} objects) to {exp_out_path}?",
            default=True,
        )
        if should_save:
            run_final_cleanup()

    if should_save:
        persist_scene()
    else:
        print("Not saving the map.")

    owandb.finish()


if __name__ == "__main__":
    main()
