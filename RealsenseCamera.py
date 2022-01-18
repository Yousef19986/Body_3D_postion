# https://pysource.com
import pyrealsense2 as rs
import numpy as np


class RealsenseCamera:
    """
    Class: RealsenseCamera
    Attributes:
        pipeline:        rs.pipeline
        profile:         rs.pipeline_profile
        align:           rs.align
        intrinsics:      rs.intrinsics
    Methods:
        get_frame_stream(): returns color_image, depth_image, depth_frame
        get_intrinsics():   returns intrinsics
        release():          releases the camera
    """

    def __init__(self, camera_type="lidar"):
        """
        Constructor
        :param camera_type:     string ['lidar','depth_low','depth_high']
        """

        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        config = rs.config()

        if camera_type == "lidar":
            config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        elif camera_type == "depth_low":
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        elif camera_type == "depth_high":
            config.enable_stream(rs.stream.color, 1280,
                                 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # Start streaming
        self.profile = self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame_stream(self):
        """
        Method: get_frame_stream
        description:
            pulls frames from the camera and aligns them
            :returns: color_image, depth_image, depth_frame
        """

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, raise error
            raise Exception(
                "Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected"
            )

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, depth_frame

    def get_intrinsics(self):
        """
        Method: get_intrinsics
        description:
            returns intrinsics of the camera
            :returns: intrinsics
        """
        return (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def release(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
