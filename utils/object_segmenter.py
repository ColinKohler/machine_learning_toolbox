import pcl
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

class ObjectSegmenter(object):
    def __init__(self, camera_info_filepath):
        cam_info = CameraInfo()
        with open(camera_info_filepath, 'r') as fd:
            cam_data = fd.read()
        cam_info.deserialize(cam_data)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)

    # Segment out the object from the depth frame
    def segmentObjectInFrame(self, pt_cloud, dis_to_object):
        pt_cloud = pcl.PointCloud(pt_cloud.astype(np.float32))
        pt_cloud = self._removePointsOutsideWorkspace(pt_cloud, dis_to_object)
        pt_cloud = self._removeFloorPlane(pt_cloud)
        pt_cloud = self._removeOutliers(pt_cloud)
        pt_cloud = self._clusterPoints(pt_cloud)
        return self._getCloudLimits(pt_cloud)

    # Remove all points behind the object to isolate the workspace
    def _removePointsOutsideWorkspace(self, pt_cloud, dis_to_object):
        # Y Filter
        fil = pt_cloud.make_passthrough_filter()
        fil.set_filter_field_name('y')
        fil.set_filter_limits(0.0, 2.0)
        pt_cloud = fil.filter()

        # Z Filter
        fil = pt_cloud.make_passthrough_filter()
        fil.set_filter_field_name('z')
        fil.set_filter_limits(dis_to_object, dis_to_object+0.75)
        pt_cloud = fil.filter()

        # X Filter
        n_cloud = np.asarray(pt_cloud)
        x_min = np.min(n_cloud, axis=0)[0]
        x_max = np.max(n_cloud, axis=0)[0]

        fil = pt_cloud.make_passthrough_filter()
        fil.set_filter_field_name('x')
        fil.set_filter_limits(x_min+0.75, x_max-0.5)
        pt_cloud = fil.filter()

        return pt_cloud

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

    # Get the bounding box from the point cloud
    def _getCloudLimits(self, cloud):
        n_cloud = np.asarray(cloud)

        min_x = np.min(n_cloud[:,0])
        min_y = np.min(n_cloud[:,1])
        min_z = np.min(n_cloud[:,2])
        min_bbox = [min_x, min_y, min_z]

        max_x = np.max(n_cloud[:,0])
        max_y = np.max(n_cloud[:,1])
        max_z = np.max(n_cloud[:,2])
        max_bbox = [max_x, max_y, max_z]

        min_bbox = self.cam_model.project3dToPixel(min_bbox)
        max_bbox = self.cam_model.project3dToPixel(max_bbox)

        return [min_bbox, max_bbox]
