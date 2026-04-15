#!/usr/bin/env python3

"""
Pose Estimation Demo with NNStreamer + TFLite
---------------------------------------------

This script runs single-person pose estimation using GStreamer pipelines
with NNStreamer tensor_filter (TFLite backend). It supports CPU and NPU backends
on i.MX platforms and can render results with Wayland/X11/cairooverlay.

Author: <Your Name/Org>
License: MIT
"""

import os
import sys
import gi
import logging
import math
import numpy as np
import ctypes
import cairo
import argparse
import signal
import pathlib
import subprocess
import re
import glob
from threading import Lock

gi.require_version('Gst', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GLib

DEBUG = False


class Kalman1D:
    def __init__(self, process_noise=0.18, measurement_noise=0.18):
        self.x = None
        self.p = 1.0
        self.q = process_noise
        self.r = measurement_noise

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict
        self.p += self.q

        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)

        return self.x


class PoseEstimator:
    def __init__(
        self, platform, device, backend, model, labels,  save_video, display,
        width, height, c_format, callback=None, r=1, g=0, b=0):
        """Creates an instance of the demo

        Arguments:
        device -- What camera or video file to use
        backend -- Whether to use NPU or CPU
        model -- the path to the model
        labels -- the path to the labels
        display -- Whether to use X11 or Weston
        callback -- Callback to pass stats to
        width -- Width of output
        height -- Height of output
        r -- Red value for labels
        g -- Green value for labels
        b -- Blue value for labels
        """

        self.kp_lock = Lock()

        self.loop = None
        self.pipeline = None
        self.running = False
        self.video_caps = None
        self.first_frame = True

        self.tflite_model = model
        self.label_path = labels
        self.device = device
        self.backend = backend
        self.display = display
        self.video = save_video
        self.c_format = c_format
        self.callback = callback
        self.interval_time = -1
        self.reload_time = -1
        self.r = r
        self.b = b
        self.g = g
        self.platform = platform

        self.VIDEO_WIDTH = width
        self.VIDEO_HEIGHT = height

        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height

        self.MODEL_INPUT_HEIGHT = 225
        self.MODEL_INPUT_WIDTH = 225

        self.KEYPOINT_SIZE = 17
        self.OUTPUT_STRIDE = 16
        self.GRID_XSIZE = (self.MODEL_INPUT_WIDTH // self.OUTPUT_STRIDE) + 1
        self.GRID_YSIZE = (self.MODEL_INPUT_HEIGHT // self.OUTPUT_STRIDE) + 1
        self.CROP_LEFT = (self.VIDEO_WIDTH - self.IMAGE_WIDTH) // 2
        self.CROP_RIGHT = ((self.VIDEO_WIDTH - self.IMAGE_WIDTH) + 1) // 2
        self.CROP_TOP = (self.VIDEO_HEIGHT - self.IMAGE_HEIGHT) // 2
        self.CROP_BOTTOM = ((self.VIDEO_HEIGHT - self.IMAGE_HEIGHT) + 1) // 2

        self.SCORE_THRESHOLD = 0.7

        self.kalman_filters = [
            {"x": Kalman1D(), "y": Kalman1D()}
            for _ in range(self.KEYPOINT_SIZE)
        ]

        self.tflite_labels = []
        self.kps = []

        if not self.tflite_init():
            raise Exception

        GObject.threads_init()
        Gst.init(None)

    def _select_bitrate(self):
        px = self.VIDEO_WIDTH * self.VIDEO_HEIGHT
        if px >= 1920 * 1080:
            return 8000  # 8 Mbps
        elif px >= 1280 * 720:
            return 5000  # 5 Mbps
        else:
            return 2500  # SD

    def run_code(self):
        """Starts pipeline and runs demo"""

        if self.backend == "CPU":
            backend = "true:cpu custom=NumThreads:4"
        elif self.backend == "GPU":
            os.environ["USE_GPU_INFERENCE"] = "1"
            backend = ("true:gpu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")
        else:
            os.environ["USE_GPU_INFERENCE"] = "0"
            backend = ("true:npu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")
                       
        save_video = ""               
        if self.video == "yes":
            save_video = (
                " tt. ! queue max-size-buffers=16 leaky=no "
                " ! videoconvert "
                " ! videorate "
                " ! video/x-raw,format=NV12,framerate=30/1 "
                " ! queue max-size-buffers=16 leaky=no "
                f" ! vpuenc_h264 bitrate={self._select_bitrate()} gop-size=30 "
                " ! h264parse "
                " ! mp4mux faststart=true "
                " ! filesink location=output_video.mp4 sync=false "
            )
            
        if self.display == "X11":
            display = "ximagesink name=img_tensor"
        elif self.display == "None":
            self.print_time = GLib.get_monotonic_time()
            display = "fakesink  name=img_tensor"
        else:
            display = "waylandsink sync=false max-lateness=-1 name=img_tensor "

        # main loop
        self.loop = GObject.MainLoop()

        self.update_time = GLib.get_monotonic_time()
        self.old_time = GLib.get_monotonic_time()

        if self.platform == "imx8qmmek":
            decoder = "h264parse ! v4l2h264dec ! imxvideoconvert_g2d "
        else:
            decoder = "vpudec "

        if self.device.startswith("rtsp://"):
            gst_launch_cmdline = f"rtspsrc location={self.device} ! "
            gst_launch_cmdline += "rtph264depay ! h264parse ! vpudec ! "
            gst_launch_cmdline += "video/x-raw,format=NV12 ! "
            gst_launch_cmdline += "imxvideoconvert_g2d ! video/x-raw,format=BGRx ! "
            gst_launch_cmdline += "tee name=t "

        elif "/dev/video" in self.device and self.c_format == 'YUY2':
            gst_launch_cmdline = 'v4l2src name=cam_src device=' + self.device
            gst_launch_cmdline += ' ! video/x-raw,width='
            gst_launch_cmdline += str(int(self.VIDEO_WIDTH)) + ',height='
            gst_launch_cmdline += str(int(self.VIDEO_HEIGHT))
            gst_launch_cmdline += ',format='
            gst_launch_cmdline += self.c_format
            gst_launch_cmdline += ' ! tee name=t'
        elif "/dev/video" in self.device:
            gst_launch_cmdline = 'v4l2src name=cam_src device=' + self.device
            gst_launch_cmdline += ' ! imxvideoconvert_g2d ! video/x-raw,width='
            gst_launch_cmdline += str(int(self.VIDEO_WIDTH)) +',height='
            gst_launch_cmdline += str(int(self.VIDEO_HEIGHT))
            gst_launch_cmdline += ',format='
            gst_launch_cmdline += self.c_format
            gst_launch_cmdline += ' ! tee name=t'
        else:
            gst_launch_cmdline = (
                f"filesrc location={self.device} "
                f"! decodebin name=dec "
                f"! videoconvert name=pre_tee_convert "
                f"! tee name=t "
            )

        gst_launch_cmdline += ' t. ! imxvideoconvert_g2d ! video/x-raw,'
        gst_launch_cmdline += 'width={:d},'.format(self.MODEL_INPUT_WIDTH)
        gst_launch_cmdline += 'height={:d},'.format(self.MODEL_INPUT_HEIGHT)
        gst_launch_cmdline += 'format=ARGB ! ' 
        gst_launch_cmdline += 'queue max-size-buffers=2 leaky=2 ! videoconvert'
        gst_launch_cmdline += ' ! video/x-raw,format=RGB ! tensor_converter ! '
        gst_launch_cmdline += ' tensor_filter framework=tensorflow-lite model='
        gst_launch_cmdline += self.tflite_model + ' accelerator=' + backend
        gst_launch_cmdline += ' silent=FALSE name=tensor_filter latency=1 !'
        gst_launch_cmdline += ' tensor_sink name=tensor_sink t.'
        gst_launch_cmdline += ' ! imxvideoconvert_g2d ! '
        gst_launch_cmdline += 'cairooverlay name=tensor_res ! '
        gst_launch_cmdline += 'tee name=tt tt. ! queue max-size-buffers=6 max-size-time=0 max-size-bytes=0 leaky=downstream ! '
        gst_launch_cmdline += display
        gst_launch_cmdline += save_video

        print(gst_launch_cmdline)
        # init pipeline
        self.pipeline = Gst.parse_launch(gst_launch_cmdline)

        # bus and message callback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        self.tensor_filter = self.pipeline.get_by_name('tensor_filter')

        # tensor sink signal : new data callback
        tensor_sink = self.pipeline.get_by_name('tensor_sink')
        tensor_sink.connect('new-data', self.new_data_cb)

        tensor_res = self.pipeline.get_by_name('tensor_res')
        tensor_res.connect('draw', self.draw_overlay_cb)
        tensor_res.connect('caps-changed', self.prepare_overlay_cb)
        if self.callback is not None:
            GObject.timeout_add(500, self.callback, self)

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        self.set_window_title('img_tensor', 'Single Person Pose Estimation')

        def handle_sigint(sig, frame):
            print("\nCtrl+C detected → Sending EOS to finalize MP4...")
            self.pipeline.send_event(Gst.Event.new_eos())

            msg = bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS)
            if msg:
                print("EOS received → pipeline ended cleanly")
            else:
                print("Timeout waiting for EOS → forcing exit")

            self.loop.quit()

        signal.signal(signal.SIGINT, handle_sigint)

        # ---- Run main loop ----
        try:
            self.loop.run()
        finally:
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
            sys.exit(0)

    def tflite_init(self):
        """Check tflite model and load labels.

        :return: True if successfully initialized
        """

        if not os.path.exists(self.tflite_model):
            logging.error('cannot find tflite model [%s]', self.tflite_model)
            return False

        try:
            with open(self.label_path, 'r') as label_file:
                for line in label_file.readlines():
                    self.tflite_labels.append(line)
        except FileNotFoundError:
            logging.error('cannot find tflite label [%s]', self.label_path)
            return False

        logging.info(
            'finished to load labels, total [%d]', len(self.tflite_labels))
        return True

    def should_reset_pose(self, kps):
        """
        Detect if pose should be reset.
        True when almost no keypoints detected or all have near-zero scores.
        """
        if len(kps) == 0:
            return True

        # count valid keypoints with score > threshold
        valid = sum(1 for p in kps if p['score'] > 0.2)

        # If very few points detected → pose lost
        return valid <= 3

    def smooth_keypoints(self, new_kps):
        smoothed = []

        for i, p in enumerate(new_kps):
            if p["score"] < 0.55:
                # Keep previous if detection weak
                if hasattr(self, "prev_kps") and len(self.prev_kps) > 0:
                    smoothed.append(self.prev_kps[i])
                else:
                    smoothed.append(p)
                continue

            kx = self.kalman_filters[i]["x"]
            ky = self.kalman_filters[i]["y"]

            sx = kx.update(p["x"])
            sy = ky.update(p["y"])

            smoothed.append({
                "x": sx,
                "y": sy,
                "score": p["score"],
                "label": p["label"],
            })

        self.prev_kps = smoothed
        return smoothed

    def stabilize_limbs(self, kps):
        """
        Reduces jitter between connected joints by stabilizing limb vectors.
        Works AFTER Kalman filtering.
        """

        LIMBS = [
            (11, 13),  # left hip → left knee
            (13, 15),  # left knee → left ankle
            (12, 14),  # right hip → right knee
            (14, 16),  # right knee → right ankle
            (5, 7),  # left shoulder → left elbow
            (7, 9),  # left elbow → left wrist
            (6, 8),  # right shoulder → right elbow
            (8, 10),  # right elbow → right wrist
        ]

        SMOOTH = 0.12  # lower = stiffer, higher = looser

        for a, b in LIMBS:
            if kps[a]['score'] < 0.4 or kps[b]['score'] < 0.4:
                continue

            ax, ay = kps[a]['x'], kps[a]['y']
            bx, by = kps[b]['x'], kps[b]['y']

            # Vector
            vx = bx - ax
            vy = by - ay

            # Stabilized vector (reduces jitter)
            kps[b]['x'] = ax + (1 - SMOOTH) * vx
            kps[b]['y'] = ay + (1 - SMOOTH) * vy

        return kps

    def new_data_cb(self, sink, buffer):
        """Callback for tensor sink signal."""
        if not self.running:
            return False

        new_time = GLib.get_monotonic_time()
        self.interval_time = new_time - self.old_time
        self.old_time = new_time

        if buffer.n_memory() != 4:
            return False

        # ---- Map Tensor Memory ----
        mem_heatmap = buffer.peek_memory(0)
        ok1, info_heatmap = mem_heatmap.map(Gst.MapFlags.READ)

        mem_offset = buffer.peek_memory(1)
        ok2, info_offset = mem_offset.map(Gst.MapFlags.READ)

        if not ok1 or not ok2:
            return False

        # Decode both tensors
        decoded_heatmap = np.frombuffer(info_heatmap.data, dtype=np.float32)
        decoded_offset = np.frombuffer(info_offset.data, dtype=np.float32)

        new_kps = []
        for keyPointIdx in range(self.KEYPOINT_SIZE):

            yPosIdx = xPosIdx = -1
            highestScore = -1e9

            # Find highest heatmap score in grid
            for yIdx in range(self.GRID_YSIZE):
                for xIdx in range(self.GRID_XSIZE):

                    idx = (yIdx * self.GRID_XSIZE + xIdx) * self.KEYPOINT_SIZE + keyPointIdx
                    current = decoded_heatmap[idx]
                    currentScore = 1.0 / (1.0 + math.exp(-current))

                    if currentScore > highestScore:
                        highestScore = currentScore
                        yPosIdx = yIdx
                        xPosIdx = xIdx

            # Compute final offsets
            base = (yPosIdx * self.GRID_XSIZE + xPosIdx) * self.KEYPOINT_SIZE * 2
            yOffset = decoded_offset[base + keyPointIdx]
            xOffset = decoded_offset[base + self.KEYPOINT_SIZE + keyPointIdx]

            # Scale to input size
            yPosition = (yPosIdx / (self.GRID_YSIZE - 1)) * self.MODEL_INPUT_HEIGHT + yOffset
            xPosition = (xPosIdx / (self.GRID_XSIZE - 1)) * self.MODEL_INPUT_WIDTH + xOffset

            new_kps.append({
                'y': yPosition,
                'x': xPosition,
                'label': keyPointIdx,
                'score': highestScore
            })

        # ---- ALWAYS UNMAP BEFORE ANY PROCESSING ----
        mem_heatmap.unmap(info_heatmap)
        mem_offset.unmap(info_offset)

        # ---- Reset pose if necessary ----
        if self.should_reset_pose(new_kps):
            self.prev_kps = []
            with self.kp_lock:
                self.kps = new_kps
            return False

        # ---- Smooth normally ----
        smoothed_kps = self.smooth_keypoints(new_kps)
        smoothed_kps = self.stabilize_limbs(smoothed_kps)

        with self.kp_lock:
            self.kps = smoothed_kps

        # Print latency in console mode
        if self.display == "None":
            if (GLib.get_monotonic_time() - self.print_time) > 1000000:
                inference = self.tensor_filter.get_property("latency")
                print(
                    "Inference time: " + str(inference / 1000) +
                    " ms (" + "{:5.2f}".format(1 / (inference / 1000000)) + " IPS)"
                )
                self.print_time = GLib.get_monotonic_time()

        return False

    def prepare_overlay_cb(self, overlay, caps):
        self.video_caps = caps

    def draw_overlay_cb(self, overlay, context, timestamp, duration):
        scale_height = self.VIDEO_HEIGHT/1080
        scale_width = self.VIDEO_WIDTH/1920
        scale_text = max(scale_height, scale_width)
        if self.first_frame:
            context.select_font_face(
                'Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            context.set_source_rgb(1, 1,1)
            context.move_to(int(50 * scale_width), int(100 * scale_height))
            context.set_font_size(int(50.0 * min(scale_width,scale_height)))
            context.show_text("Loading...")
            self.first_frame = False
            return

        if self.video_caps == None or not self.running:
            return

        # mutex_lock alternative required
        with self.kp_lock:
            kpts = list(self.kps)

        # mutex_unlock alternative needed

        context.select_font_face(
            'Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(35.0)

        # draw body lines
        context.set_source_rgb(self.r, self.g, self.b)
        context.set_line_width(5)

        self.draw_line(overlay, context, kpts, 5, 6)
        self.draw_line(overlay, context, kpts, 5, 7)
        self.draw_line(overlay, context, kpts, 7, 9)
        self.draw_line(overlay, context, kpts, 6, 8)
        self.draw_line(overlay, context, kpts, 8, 10)
        self.draw_line(overlay, context, kpts, 6, 12)
        self.draw_line(overlay, context, kpts, 5, 11)
        self.draw_line(overlay, context, kpts, 11, 13)
        self.draw_line(overlay, context, kpts, 13, 15)
        self.draw_line(overlay, context, kpts, 12, 11)
        self.draw_line(overlay, context, kpts, 12, 14)
        self.draw_line(overlay, context, kpts, 14, 16)

        context.stroke()

    def draw_line(self, overlay, context, kpts, from_key, to_key):
        kpts_len = len(kpts)
        if from_key > kpts_len-1 or to_key > kpts_len-1:
            return
        if kpts[from_key]['score'] < self.SCORE_THRESHOLD or (
            kpts[to_key]['score'] < self.SCORE_THRESHOLD):
            return

        # Get actual overlay size from caps
        structure = self.video_caps.get_structure(0)
        w = structure.get_int("width")[1]
        h = structure.get_int("height")[1]

        scale_x = w / self.MODEL_INPUT_WIDTH
        scale_y = h / self.MODEL_INPUT_HEIGHT
        # --- FIX ENDS HERE ---

        context.move_to(
            kpts[from_key]['x'] * scale_x,
            kpts[from_key]['y'] * scale_y
        )
        context.line_to(
            kpts[to_key]['x'] * scale_x,
            kpts[to_key]['y'] * scale_y
        )

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            logging.info('received eos message')
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.warning('[error] %s : %s', error.message, debug)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            logging.warning('[warning] %s : %s', error.message, debug)
        elif message.type == Gst.MessageType.STREAM_START:
            logging.info('received start message')
        elif message.type == Gst.MessageType.QOS:
            data_format, processed, dropped = message.parse_qos_stats()
            format_str = Gst.Format.get_name(data_format)
            logging.debug(
                '[qos] format[%s] processed[%d] dropped[%d]',
                format_str, processed, dropped)

    def set_window_title(self, name, title):
        element = self.pipeline.get_by_name(name)
        if element is not None:
            pad = element.get_static_pad('sink')
            if pad is not None:
                tags = Gst.TagList.new_empty()
                tags.add_value(Gst.TagMergeMode.APPEND, 'title', title)
                pad.send_event(Gst.Event.new_tag(tags))

ALLOWED_RESOLUTIONS = [(640, 360), (640, 480), (1280, 720), (1920, 1080)]

def get_video_nodes():
    return glob.glob("/dev/video*")

def is_streaming(dev, num_buffers=3):
    """Check if the video node can stream."""
    try:
        subprocess.run(
            ["timeout", "5", "gst-launch-1.0", "-q",
             "v4l2src", f"device={dev}", f"num-buffers={num_buffers}", "!", "fakesink"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_format_info(dev):
    """Return only supported (w,h,pixfmt) among 640x480, 1280x720, 1920x1080."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        formats = []
        current_pixfmt = None

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("["):
                pixfmt_match = re.search(r"'(\w+)'", line)
                if pixfmt_match:
                    current_pixfmt = pixfmt_match.group(1)
            elif "Size:" in line and "Discrete" in line:
                size_match = re.search(r"(\d+)x(\d+)", line)
                if size_match and current_pixfmt:
                    w, h = map(int, size_match.groups())
                    if (w, h) in ALLOWED_RESOLUTIONS and current_pixfmt in ("YUYV", "BGRx", "MJPG"):
                        formats.append((w, h, current_pixfmt))
        return formats
    except subprocess.CalledProcessError:
        return []

def select_webcam():
    print("Available supported video nodes for streaming:")
    print("-------------------------------------")
    nodes = get_video_nodes()
    working_nodes = []

    # Check each node
    for dev in nodes:
        if is_streaming(dev):
            formats = get_format_info(dev)
            if not formats:
                continue
            print(f"{len(working_nodes)+1}. {dev}")
            for idx, (w, h, fmt) in enumerate(formats, start=1):
                print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")
            working_nodes.append((dev, formats))

    if not working_nodes:
        print("No working video nodes found with standard resolutions.")
        return

    # Ask user to select node
    while True:
        choice = input(f"\nSelect a video node [1-{len(working_nodes)}]: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(working_nodes):
                selected_node, formats = working_nodes[choice-1]
                break
        print("Invalid choice, try again.")

    # Show available standard resolutions for selected node
    print(f"\nAvailable supported resolutions for {selected_node}:")
    for idx, (w, h, fmt) in enumerate(formats, start=1):
        print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")

    # Ask user to select resolution
    while True:
        res_choice = input(f"Select a resolution [1-{len(formats)}]: ")
        if res_choice.isdigit():
            res_choice = int(res_choice)
            if 1 <= res_choice <= len(formats):
                width, height, pixfmt = formats[res_choice-1]
                break
        print("Invalid choice, try again.")

    # Display final selection
    print("\nSelected video node parameters:")
    print(f"Device: {selected_node}")
    print(f"Resolution: {width}x{height}")
    print(f"Pixel Format: {pixfmt}")
    print(f"Framerate: 30/1")

    return {
        "device": selected_node,
        "width": width,
        "height": height,
        "pixfmt": pixfmt,
        "framerate": 30
    }

            
def get_resolution(filepath: str):
    """
    Get resolution (width, height) of a video file or RTSP stream
    using gst-discoverer-1.0.
    Returns (width, height) or (None, None) on failure.
    """
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        width, height = None, None
        for line in result.stdout.splitlines():
            line = line.strip()
            # Match "Width: 1920", "Height: 1088", etc.
            m_w = re.search(r"Width\s*:\s*(\d+)", line, re.IGNORECASE)
            m_h = re.search(r"Height\s*:\s*(\d+)", line, re.IGNORECASE)
            if m_w:
                width = int(m_w.group(1))
            if m_h:
                height = int(m_h.group(1))

        if width is None or height is None:
            print(f"[WARNING] Could not detect resolution for {filepath}")
            return None, None

        print(f"[INFO] Detected resolution: {width}x{height}")
        return width, height

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gst-discoverer-1.0 failed: {e}")
        return None, None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pose Estimation (NNStreamer + TFLite)")
    parser.add_argument("--video", default="/workspace/videos/pose_example.mov", help="Video source: 'discover', RTSP URL (rtsp://<ip>:<port>/<path>), or video file path")
    parser.add_argument("--platform", default="imx8mp", help="Target platform")
    parser.add_argument("--backend", choices=["CPU", "NPU"], default="CPU", help="Inference backend")
    parser.add_argument("--model", default="/workspace/models/posenet_resnet50_uint8_float32_quant.tflite", help="TFLite model path")
    parser.add_argument("--labels", default="/workspace/models/key_point_labels.txt", help="Labels file")
    parser.add_argument("--save-video", choices=["yes", "no"], default="no", help="Save output video (default: no)")
    parser.add_argument("--display", choices=["X11", "weston"], default="weston", help="Display backend (default: weston)")
    parser.add_argument("--cam-width", type=int, default=1920,help="Camera input width (default: 1920)")
    parser.add_argument("--cam-height", type=int, default=1080,help="Camera input height (default: 1080)")
    parser.add_argument("--cam-format", default="BGRx",help="Camera pixel format (default: BGRx)")
    
    args = parser.parse_args()
    
    # Video source handling
    if args.video.lower() == "discover":
        cam_params = select_webcam()
        args.video = cam_params["device"]
        w, h = cam_params["width"], cam_params["height"]
        args.cam_format = cam_params["pixfmt"]
        if args.cam_format == "YUYV":
            args.cam_format = "YUY2"
        else:
            args.cam_format = "BGRx"
    elif args.video.startswith("rtsp://") or not args.video.startswith("/dev/video"):
        # RTSP stream or video file: probe resolution using gst-discoverer-1.0
        w, h = get_resolution(args.video)
        if w is None or h is None:
            # fallback if probe fails
            w, h = args.cam_width, args.cam_height
            print(f"[WARNING] Could not detect resolution, using defaults {w}x{h}")
        print(f"Stream/Video selected -> {args.video}, resolution: {w}x{h}")
    
    elif args.video.startswith("/dev/video"):
        # Direct camera device: use provided width/height
        w, h = args.cam_width, args.cam_height
        print(f"Direct camera device detected -> {args.video}, resolution {w}x{h}")

    if not w or not h or w <= 0 or h <= 0:
        print(f"[ERROR] Invalid resolution detected: width={w}, height={h}")
        sys.exit(1)

    # Print configuration
    print("========== Configuration ==========")
    print(f"Video Source : {args.video}")
    print(f"Platform     : {args.platform}")
    print(f"Backend      : {args.backend}")
    print(f"Model File   : {args.model}")
    print(f"Label File   : {args.labels}")
    print(f"Save Video   : {args.save_video}")
    print(f"Display      : {args.display}")
    print(f"Width        : {w}")
    print(f"Height       : {h}")
    print(f"Format       : {args.cam_format}")
    print("===================================")

    est = PoseEstimator(
        platform=args.platform,
        device=args.video,
        backend=args.backend,
        model=args.model,
        labels=args.labels,
        save_video=args.save_video,
        display=args.display,
        width=w,
        height=h,
        c_format=args.cam_format
    )
    est.run_code()
