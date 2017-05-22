import os
import time
import argparse

import sys
sys.path.append('../SSD-Tensorflow/')

#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from notebooks import visualization

from ssd import process_image

import cv2
import numpy as np

#deep_sort
sys.path.append('../deep_sort/')

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from generate_detections import create_box_encoder


def gather_sequence_info(cam_id, frame):

    seq_info = {
        "sequence_name": cam_id,
        "image_filenames": cam_id,
        "detections": None,
        "groundtruth": None,
        "image_size": frame.shape,
        "min_frame_idx": 1,
        "max_frame_idx": float('Inf'),
        "feature_dim": None,
        "update_ms": 5
    }
    return seq_info

def feature_extraction(encoder, image, boxes):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    image : ndarrary
        Image read from file or frame
    boxes :
        Matrix of boudning boxes and corresponding confidence in format '(x, y, w, h, confidence)'

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.
    """
    if image is None or image == []:
        print ('Image is none.')
        return []

    if boxes == []:
        print ('Boxes are empty.')
        return []

    if not np.shape(boxes)[1] == 5:
        print ('Boxes dimension is incorrect. (x, y, w, h, confidence)')
        return []

    bbox = boxes[:, :4]
    confidence = boxes[:, 4]
    features = encoder(image, bbox.copy())
    print('Size of the extracted features: {}'.format(np.asarray(features).shape))
    detection_list = []
    #detection_out = [np.r_[(box, confidence, feature)] for box, confidence, feature
    #                   in zip(bbox, confidences, features)]
    for box, confid, feature in zip(bbox, confidence, features):
        detection_list.append(Detection(box, confid, feature))

    return detection_list

def cam_detection(encoder, cam_id, min_confidence, max_cosine_distance, nms_max_overlap, nn_budget):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera {} open failed.".format(cam_id))
        return

    ret, frame = cap.read()
    seq_info = gather_sequence_info(cam_id, frame)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    jump_step = 3
    print ('Start frame_callback')
    def frame_callback(vis, frame_idx):
        print ('Processing frame: {:d}'.format(frame_idx))
        if not cap.isOpened(): return
        st_time = time.time()
        # Load image and generate detections.
        ret, frame = cap.read()

        # Object detection -- SSD
        rclasses, rscores, rbboxes = process_image(frame, select_threshold=min_confidence,
                                                   nms_threshold=nms_max_overlap)     #boxes in (x1, y1, x2, y2) format

        # Select only person
        cls_ind = 15

        inds = np.where(rclasses == cls_ind)[0]
        if len(inds) == 0:
            dets = []
        else:
            cls_boxes = rbboxes[inds, :]
            cls_boxes_v = cls_boxes.copy()
            #convert normalized (ny1, nx1, ny2, nx2) to pixel coordinate (x1, y1, x2, y2)
            cls_boxes_v[:, 1] = cls_boxes[:, 0] * frame.shape[0]
            cls_boxes_v[:, 0] = cls_boxes[:, 1] * frame.shape[1]
            cls_boxes_v[:, 3] = cls_boxes[:, 2] * frame.shape[0]
            cls_boxes_v[:, 2] = cls_boxes[:, 3] * frame.shape[1]

            cls_boxes_v[:, 2:] -= cls_boxes_v[:, :2]            #transfer to (x,y,w,h) format
            cls_scores = rscores[inds]

            dets = np.hstack((cls_boxes_v,
                              cls_scores[:, np.newaxis])).astype(np.float32)

            print ('Filtered detection size: {}'.format(np.asarray(dets).shape))
        # Feature extraction
        detections = feature_extraction(encoder, frame, dets)

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        vis.set_image(frame.copy())
        vis.draw_detections(detections)
        vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        end_time = time.time()
        print('Processing time: {:.3f}s'.format(end_time-st_time))

    # Run tracker.
    visualizer = visualization.Visualization(seq_info, update_ms=5)

    visualizer.run(frame_callback)

    cap.release()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SSD_DeepSort_Demo')
    parser.add_argument('--cameraID', dest='cam_id', help='Camera ID to use [0]', default=0, type=int)

    # Re-ID feature extractor
    parser.add_argument(
        "--reID_model",
        default="../deep_sort/resources/networks/mars-small128.ckpt-68577",
        help="Path to checkpoint file")
    parser.add_argument(
        "--loss_mode", default="cosine", help="Network loss training mode")

    # Tracking
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
                              " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
                                 "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
                                       "box height. Detections with height smaller than this value are "
                                       "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap", help="Non-maxima suppression threshold: Maximum "
                                  "detection overlap.", default=0.45, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
                                      "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
                            "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    '''
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        st_time = time.time()
        ret, frame = cap.read()

        rclasses, rscores, rbboxes = process_image(frame)

        print(rclasses.shape, rscores.shape, rbboxes.shape)

        visualization.bboxes_draw_on_img(frame, rclasses, rscores, rbboxes, visualization.colors_plasma)
        cv2.imshow('Camera 0', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        print('Detection time:', end_time - st_time)
        print('Cls no.:', rbboxes)
    '''
    f = create_box_encoder(args.reID_model, batch_size=32, loss_mode=args.loss_mode)
    cam_detection(f, args.cam_id, args.min_confidence, args.max_cosine_distance, args.nms_max_overlap, args.nn_budget)
