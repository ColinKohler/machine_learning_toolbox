import pcl
import numpy as np
from sklean.cluster import MeanShift, estimate_bandwidth
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

class ObjectSegmenter(object):
    def __init__(self, camera_info_filepath):
        cam_info = CameraInfo()
        with open(camera_info_filepath, 'r') as fd:
            cam_data = fc.read()
        cam_info.deserialize(cam_data)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)

    def segmentObjectInFrames(self, depth_frames, rgb_frames, dis_to_object):
        bboxes = list()
        for i, (depth_frame, rgb_frame) in enumerate(zip(depth_frames, rgb_frames)):
            pt_cloud = pcl.PointCloud(cloud.astype(np.float32))

    # Remove all points behind the object to isolate the workspace
    def _removePointsOutsideWorkspace(self, pt_cloud, dis_to_object):
        fil = pt_cloud.make_passthrough_filter()
        fil.set_filter_field_name('z')
        fil.set_filter_limits(0, dis_to_object)
        return fil.filter()

    # Remove the floor points
    def _removeFloorPlane(self, pt_cloud):
        seg = pt_cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.06)
        indices, model = seg.segment()

        return pt_cloud.extract(indices, negative=True)

    # Remove Outliers to reduce noise in cloud
    def _removeOutliers(self, pt_cloud):
        fil = pt_cloud.make_statistical_outlier_filter()
        fil.set_mean_k(100)
        fil.set_std_dev_mul_thresh(0.1)
        return fil.filter()

    # Cluster the cloud points
    def _clusterPoints(self, pt_cloud):
        np_pt_cloud = np.asarray(pt_cloud)
        bandwidth = estimate_bandwidth(np_pt_cloud, quantile=0.65, n_samples=500)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-2)
        ms.fit(np_pt_cloud)
        labels = ms.labels_
        cluster_sizes = np.bincount(labels)
        cluster_label = np.argmax(cluster_sizes)
        cluster = np_pt_cloud[np.where(labels == cluster_label)]

        pt_cloud = pcl.PointCloud(cluster.astype(np.float32))
        return pt_cloud
