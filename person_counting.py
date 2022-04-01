#!/usr/bin/env python3

"""
 Copyright (c) 2021-2022 Edoardo Maione
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import time
import queue
from logging.handlers import RotatingFileHandler
from socket import gethostname
from threading import Thread
import os
import logging as log

import numpy

from configs.api import server_addr, server_port, auth_server
from configs.buffer import BUFF_SZ, TEMPL_RATIO, FPS_FILL, MIN_SZ, PXL_ABS_TOLLERANCE, PXL_DIFF, THR_REFR_SZ
from net_io.server_pusher import ServerPusherThreadBody
from utils.timer import Timer

log_file = f'logs/log-{gethostname()}.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
rot_handler = RotatingFileHandler(log_file, mode='a', maxBytes=10 * 1024 * 1024,
                                  backupCount=20, encoding=None, delay=0)

log.basicConfig(  # filename=log_file,
    level=log.INFO,
    format='%(levelname)s %(asctime)s %(threadName)-10s %(message)s',
    handlers=[rot_handler])
# log.getLogger().handlers[0].flush()

import os
import random
import sys

import cv2 as cv

from counts.crowd_detection import CrowdWatcher, CrowdGuardian
from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, DetectionsFromFileReader
from mc_tracker.mct import MultiCameraTracker
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from counts.person_counter import GateWatcher, PersonCounter
from utils.video import MulticamCapture, NormalizerCLAHE
from utils.visualization import visualize_multicam_detections
from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

from configs.mu_conf import t_conf

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'common/python'))
from utils import monitors

set_log_config()


def check_detectors(args):
    detectors = {
        '--m_detector': args.m_detector,
        '--m_segmentation': args.m_segmentation,
        '--detections': args.detections
    }
    non_empty_detectors = [(det, value) for det, value in detectors.items() if value]
    det_number = len(non_empty_detectors)
    if det_number == 0:
        log.error('No detector specified, please specify one of the following parameters: '
                  '\'--m_detector\', \'--m_segmentation\' or \'--detections\'')
    elif det_number > 1:
        det_string = ''.join('\n\t{}={}'.format(det[0], det[1]) for det in non_empty_detectors)
        log.error('Only one detector expected but got {}, please specify one of them:{}'
                  .format(len(non_empty_detectors), det_string))
    return det_number


def update_detections(output, detections, frame_number):
    for i, detection in enumerate(detections):
        entry = {'frame_id': frame_number, 'scores': [], 'boxes': []}
        for det in detection:
            entry['boxes'].append(det[0])
            entry['scores'].append(float(det[1]))
        output[i].append(entry)


class FramesThreadBody:
    """
    Thread that retrieves video-frame input, and provide them into a queue.
    Fill the queue with variable framerate, depending on activity around the gate-entrance.
    If no/few pixels changes, fill the queue at process-data rate.
    Otherwise grow the framerate, and fill the queue until its max capacity.
    """
    def __init__(self, capture, t_boxes, box_grow_rat=1.1, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

        self.template = None
        self.img_bounds = None
        self.template_boxes = t_boxes
        self.t_box_ratio = box_grow_rat

    def __call__(self):
        tim = Timer('FramesThreadBody')
        t = time.time_ns()
        try:
            self.init_template_boxes()
        except Exception as e:
            log.error(f'[FramesThread] Setup Template Boxes:\n {str(e)}')
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(1 / 20)
                # _, _ = self.capture.get_frames()
                continue
            # tim.check('ONE STEP')
            t_sleep = time.time_ns() - t
            t_sleep = t_sleep / 1000000000
            t_sleep = (1 / FPS_FILL) - t_sleep
            if t_sleep > 0:
                time.sleep(t_sleep)
            tim.start()
            has_frames, frames = self.capture.get_frames()
            tim.check('self.capture.get_frames()')
            t = time.time_ns()

            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                for i in range(0, len(frames)):
                    frames[i] = cv.flip(frames[i], -1)
                tim.start()
                is_same = self.is_same_templ(frames)
                tim.check('is_same_templ(frames)')
                if self.frames_queue.qsize() < MIN_SZ or not is_same:
                    self.frames_queue.put(frames)
        tim.avg_calc(True)

    def init_template_boxes(self):
        if self.img_bounds is not None:
            return

        has_frames, frames = self.capture.get_frames()
        bounds = []
        for i in range(len(frames)):
            h, w, _ = frames[i].shape
            bounds.append((h, w))
        self.img_bounds = bounds

        self.set_template_boxes(self.template_boxes, self.t_box_ratio)

    def set_template_boxes(self, tboxes, ratio):
        if self.img_bounds is None:
            return
        t_boxs = []
        for i in range(len(tboxes)):
            box = tboxes[i]
            left, top, right, bottom = box
            h, w = bottom - top, right - left
            dw = int((w * (ratio - 1)) / 2)
            dh = int((h * (ratio - 1)) / 2)
            # h, w = h * ratio, w * ratio
            left = max([0, left - dw])
            left = min([self.img_bounds[i][1], left])
            right = min([self.img_bounds[i][1], right + dw])
            right = max([0, right])
            top = max([0, top - dh])
            top = min([self.img_bounds[i][0], top])
            bottom = min([self.img_bounds[i][0], bottom + dh])
            bottom = max([0, bottom])
            t_boxs.append((left, top, right, bottom))
        self.template_boxes = t_boxs

    def edit_templ(self, rgb_imgs):
        templs = []
        for i in range(len(rgb_imgs)):
            l, t, r, b = self.template_boxes[i]
            templ = rgb_imgs[i][t:b, l:r]
            th, tw, _ = templ.shape
            h, w = b - t, r - l
            h, w = int(h/(self.t_box_ratio)), int(w/(self.t_box_ratio))
            ratio = self.t_box_ratio
            dw = int((w * (1 / ratio)) / 2)
            dh = int((h * (1 / ratio)) / 2)
            # h, w = h * ratio, w * ratio
            lb = max([0, dw])
            lb = min([tw, lb])
            rb = min([tw, tw - dw])
            rb = max([0, rb])
            tb = max([0, dh])
            tb = min([th, tb])
            bb = min([th, th - dh])
            bb = max([0, bb])
            # cv.rectangle(rgb_imgs[i], (l+lb, t+tb), (l+rb, t+bb), [0, 0, 0], thickness=-1)
            cv.rectangle(rgb_imgs[i], (l, t), (r, b), [0, 0, 0], thickness=1)
            templ = cv.cvtColor(templ, cv.COLOR_BGR2GRAY)
            cv.rectangle(templ, (lb, tb), (rb, bb), [0, 0, 0], thickness=-1)
            templs.append(templ)
        return templs

    def set_template(self, rgb_imgs):
        templs = self.edit_templ(rgb_imgs)
        self.template = templs

    def is_same_templ(self, rgb_imgs):
        if self.template is None:
            self.set_template(rgb_imgs)
        grayImages = self.edit_templ(rgb_imgs)
        for templ, gray_img in zip(self.template, grayImages):
            diff = numpy.isclose(templ, gray_img, atol=PXL_ABS_TOLLERANCE)
            pixl_diff = (diff == False).sum()
            is_same = pixl_diff < PXL_DIFF
            # print(f'\rBuff_sz={self.frames_queue.qsize()}, pxl={pixl_diff}', end='')

            if not is_same:
                return False
        return True


def small_detections(all_detects, resz_p=0.9):
    """
    Resize the detection box of silhouette person, to improve computation speed and
    use only body-central-area to the ID's assign task.

    :param all_detects:
    :param resz_p: resize param: [0., 1.]
    :return:
    """
    assert resz_p <= 1.0
    for detects in all_detects:
        for i in range(len(detects)):
            lft, top, rgh, btm = detects[i][0]
            h = (btm - top) * (resz_p - 0.1)
            w = (rgh - lft) * resz_p
            px = (lft + rgh) / 2
            py = (top + btm) / 2 - (h / 5)
            # resz_p = resz_p/2
            lft = int(px - (w / 2))
            top = int(py - (h / 2))
            rgh = int(px + (w / 2))
            btm = int(py + (h / 2))

            if lft < rgh and top < btm:
                detects[i] = ((lft, top, rgh, btm), detects[i][1])


def run(params, config, capture, detector, reid, counter, guardian):
    """
    Given all initialized datastructures, start the main loop of Person Counting Service

    :param params: configs parameters, defined in configs.mu_conf.py.
    :param config: cofigs parameters for inference engine functionalities, exploited for ID-Assign
    :param capture: MulticamCapture object instance, that provides input frames-stream
    :param detector: MaskRCNN object, used for silhouettes detections
    :param reid: VectorCNN object, used to assign ID for each silhouette detected
    :param counter: PersonCounter Object, that perform counts estimation
    :param guardian: CrowdGuardian object, that signal crowd formation
    :return:
    """
    tim1 = Timer('run')

    win_name = 'Multi camera tracking'
    frame_number = 0
    avg_latency = AverageEstimator()
    output_detections = [[] for _ in range(capture.get_num_sources())]
    key = -1

    if config.normalizer_config.enabled:
        capture.add_transform(
            NormalizerCLAHE(
                config.normalizer_config.clip_limit,
                config.normalizer_config.tile_size,
            )
        )

    # Create a new instance of MultiCameraTracker, that assign IDs for each silhouette detected
    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config.sct_config,
                                 **vars(config.mct_config), visual_analyze=config.analyzer)

    resert_tracker = 0

    # Create a rectangles-list, to pass to FrameThreadBody.
    # It exploit those rectangles to identify the area to check if there are pixels-changes, and grow-up framerate
    # to fill the buffer
    t_boxes = []
    for gw in counter.watchers:
        t_boxes.append(gw.gate_rect)

    thread_body = FramesThreadBody(capture, t_boxes, box_grow_rat=TEMPL_RATIO, max_queue_length=BUFF_SZ)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    s_pusher = ServerPusherThreadBody(server_addr, server_port, auth_server)
    th_s_pusher = Thread(target=s_pusher)
    th_s_pusher.start()

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)  # forward the first frame to the NCS2 for silhouettes detection
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    start = time.perf_counter()

    log.info('Server start Loop ...')
    while thread_body.process:
        if not params.no_show:
            key = check_pressed_keys(key)
            if key == 27:
                break
            presenter.handleKey(key)
        try:
            frames = thread_body.frames_queue.get_nowait()  # get a new frame
            frames_read = True
        except queue.Empty:
            frames = None
            # log.warning('Frame Miss')

        if frames is None:
            continue

        try:
            tim1.start()
            all_detections = detector.wait_and_grab()  # Retrieves the result of previous silhouette detection task
            tim1.check('detector.wait_and_grab()')
        except Exception as e:
            log.error('Grab detections from NCS2: ' + str(e))

        small_detections(all_detections, 0.5)

        try:
            # tim1.start()
            crowd_alert = guardian.check(all_detections)  # check if there is a crowd in front of entrance
            # tim1.check('guardian.check()')
        except Exception as e:
            log.error('Crowd alert detection: ' + str(e))

        frame_number += 1
        try:
            tim1.start()
            detector.run_async(frames, frame_number)  # forward frame to the NCS2 for silhouettes detection
            tim1.check('detector.run_async()')
        except Exception as e:
            log.error('Feed NCS2 for next frames: ' + str(e))

        # Apply pixel's mask to the current frame: rotate/transpose/stretch/... the image
        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        # Update the current template-area to be considered for frame-rate to use to fill frames-queue.
        # This is due to possible lightning variation in the template area.
        for detections in all_detections:
            if len(detections) < 1 and thread_body.frames_queue.qsize() > THR_REFR_SZ:
                tim1.start()
                thread_body.set_template(prev_frames)
                tim1.check('set_template(prev_frames)')

        try:
            tim1.start()
            tracker.process(prev_frames, all_detections, all_masks)  # Elaborate silhouette detections to assign an ID
            tim1.check(f'tracker.process({len(all_detections[0])})')
        except Exception as e:
            log.error('tracker.process(): ' + str(e))

        tracked_objects = tracker.get_tracked_objects()  # Obtain the list of silhouettes with IDs assigned

        latency = max(time.perf_counter() - start, sys.float_info.epsilon)
        avg_latency.update(latency)
        fps = round(1. / latency, 1)
        start = time.perf_counter()

        if fps < 2 and len(all_detections[0]) < 1:
            # MultiCameraTracker object contain too much Silhouette IDs values, should be restarted
            # to keep frame-rate of computations reasonable
            log.warning(f'[FPS:{fps} (avg:{round(1. / avg_latency.get(), 1)})] [#frame:{frame_number}], ' +
                        f'detections:{len(all_detections[0])}, ' +
                        f'last_track_id:{tracker.last_global_id}, ' +
                        f'#tracks:{len(tracker.scts[0].tracks)}, ' +
                        f'#tracks_history:{len(tracker.scts[0].history_tracks)}')
            resert_tracker += 1
            if resert_tracker >= 10:
                tim1.start()
                resert_tracker = 0
                del tracker
                tracker = MultiCameraTracker(capture.get_num_sources(), reid, config.sct_config,
                                             **vars(config.mct_config), visual_analyze=config.analyzer)
                log.info('Tracker resets')
                tim1.check('RESET_TRACKER')
                tim1.avg_calc(True)

                tim1.times_repo = {}

        try:
            # tim1.start()
            counter.update(prev_frames, tracked_objects)  # perform counts estimation task
            # tim1.check('counter.update()')
        except Exception as e:
            log.error('Update Counters: ' + str(e))

        counts = counter.get_counters()
        s_pusher.add_update(counts[0], counts[1])  # it add current estimated counts to send to the collector

        guardian.render_alerts(prev_frames)

        if frame_number % 10000 == 0:
            # It log periodically the computation stats
            log.info(
                f'fps_avg:{round(1. / avg_latency.get(), 1)}\tin:{counter.global_in}\tout:{counter.global_out}\talert:{crowd_alert}')
            tim1.avg_calc(False)
            tim1.avg_log("logs/stats.txt")

        tim1.start()
        # Enrich last elaborated frame with debug informations
        vis = visualize_multicam_detections(prev_frames, tracked_objects, fps, thread_body.frames_queue.qsize(),
                                            **vars(config.visualization_config))
        tim1.check('visualize_multicam_detections()')

        if not params.no_show:
            # show debugging frame on localhost GUI
            cv.imshow(win_name, vis)
        elif bool(os.getenv("DEBUG")):
            tim1.start()
            # add a new frame to send to collector
            s_pusher.enqueue_img(vis)
            tim1.check('s_pusher.enqueue_img(vis)')

        # print('\rProcessing frame: {}, #Person = {}, fps = {} (avg_fps = {:.3})'.format(
        #     frame_number, counter.global_count, fps, 1. / avg_latency.get()), end="")

        prev_frames, frames = frames, prev_frames

    print(presenter.reportMeans())
    print('')

    thread_body.process = False
    frames_thread.join()

    s_pusher.process = False
    th_s_pusher.join()

    tim1.avg_calc(True)


def tracing(person_counter, crowd_guardian, setup):
    """
    It perform the Loadings of Networks to the Intel Neural Compute Stick 2,
    and start the main function-loop

    :param person_counter: PersonCounter object
    :param crowd_guardian: CrowdGuardian object
    :param setup: True to print the pixels grid into the frame
    :return:
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    args = t_conf
    setattr(args, 'config', os.path.join(current_dir, t_conf.config))

    if check_detectors(args) != 1:
        sys.exit(1)

    if len(args.config):
        log.info('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error('No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config.random_seed)
    capture = MulticamCapture(args.input, args.loop)

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.detections:
        object_detector = DetectionsFromFileReader(args.detections, args.t_detector)
    elif args.m_segmentation:
        object_detector = MaskRCNN(ie, args.m_segmentation,
                                   config.obj_segm.trg_classes,
                                   args.t_segmentation,
                                   args.device, args.cpu_extension,
                                   capture.get_num_sources())
    else:
        object_detector = Detector(ie, args.m_detector,
                                   config.obj_det.trg_classes,
                                   args.t_detector,
                                   args.device, args.cpu_extension,
                                   capture.get_num_sources())

    if args.m_reid:
        object_recognizer = VectorCNN(ie, args.m_reid, args.device, args.cpu_extension)
    else:
        object_recognizer = None

    run(args, config, capture, object_detector, object_recognizer, person_counter, crowd_guardian)
    log.info('The watch is end')


def one_run():
    """
    Service EntryPoint, that retrieve configuration and setup main Objects to perform People Counts Estimation

    :return:
    """
    watchers = []
    crowd_detectors = []
    assert len(t_conf.input) == len(t_conf.gate_rects) == len(t_conf.out_thresholds)
    setup = t_conf.setup_grid
    for gate, out_thr in zip(t_conf.gate_rects, t_conf.out_thresholds):
        watchers.append(GateWatcher(gate, out_thr, setup))
        crowd_detectors.append(CrowdWatcher("guardian", t_conf.crowd_thresh, 10, 10))

    counter = PersonCounter(watchers)
    guardian = CrowdGuardian(crowd_detectors)
    tracing(counter, guardian, setup)


if __name__ == '__main__':
    watchers = []
    crowd_detectors = []
    assert len(t_conf.input) == len(t_conf.gate_rects) == len(t_conf.out_thresholds)
    setup = t_conf.setup_grid
    for gate, out_thr in zip(t_conf.gate_rects, t_conf.out_thresholds):
        watchers.append(GateWatcher(gate, out_thr, setup))
        crowd_detectors.append(CrowdWatcher("guardian", t_conf.crowd_thresh, 10, 10))

    counter = PersonCounter(watchers)
    guardian = CrowdGuardian(crowd_detectors)
    tracing(counter, guardian, setup)
