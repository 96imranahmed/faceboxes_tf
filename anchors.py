import numpy as np
import itertools
import cv2
import tensorflow as tf

# Anchor configuration
SCALE_FACTOR = [10, 5]
EPSILON = 1e-8

def densify(cx, cy, scale_x_norm, scale_y_norm, factor):
    box_width_x = scale_x_norm + (factor - 1)*scale_x_norm/factor 
    box_width_y = scale_y_norm + (factor - 1)*scale_y_norm/factor
    x_top = cx - box_width_x/2
    y_top = cy - box_width_y/2
    slide_x = scale_x_norm/factor
    slide_y = scale_y_norm/factor
    anchor_dense = np.zeros((factor, factor , 4))
    for w, h in itertools.product(range(factor), repeat = 2):
        x = x_top + slide_x*w
        y = y_top + slide_y*h
        anchor_dense[w, h] = np.array([x, y, x + scale_x_norm, y + scale_y_norm])
    return np.reshape(anchor_dense, (factor**2, 4))

def get_boxes(input_config, normalised = False):
    b_out = []
    b_std = []
    shape_stub = []
    for lst in input_config:
        _, b_cur = get_anchor_boxes(lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], normalised=normalised)
        b_std.append(b_cur)
        shape_stub.append(b_cur.shape[:-1])
        b_out.append(np.reshape(b_cur, (-1, 4)).copy())
    return np.vstack(tuple(b_out)), b_std, shape_stub
    
def get_anchor_boxes(shape_dim_x, shape_dim_y, space_x, space_y, scale_x, scale_y, densify_rate, normalised = False):
    # Returns boxes in (left, top, bottom right) form (same as wider) - (0, 0) top left
    # Coordinates stay consistent with OpenCV
    step_x_norm = space_x/shape_dim_x
    step_y_norm = space_y/shape_dim_y
    scale_x_norm = scale_x/shape_dim_x
    scale_y_norm = scale_y/shape_dim_y
    dim_x = int(1/step_x_norm)
    dim_y = int(1/step_y_norm)
    boxes = []
    anchor_centers = np.zeros((dim_y, dim_x, 2))
    anchor_boxes = np.zeros((dim_y, dim_x, densify_rate**2, 4))
    for w, h in itertools.product(range(dim_x), range(dim_y)):
        cx = (w + 0.5)*step_x_norm
        cy = (h + 0.5)*step_y_norm
        cur_center = (cx, cy)
        anchor_centers[h, w] = cur_center
        anchor_boxes[h, w] = densify(cx, cy, scale_x_norm, scale_y_norm, densify_rate)
    if normalised:
        return anchor_centers, anchor_boxes
    else:
        a_c_rest = anchor_centers
        a_c_rest[:, :, 0]*=shape_dim_x
        a_c_rest[:, :, 1]*=shape_dim_y
        bx_rest = anchor_boxes
        bx_rest[:, :, :, 0]*=shape_dim_x
        bx_rest[:, :, :, 1]*=shape_dim_y
        bx_rest[:, :, :, 2]*=shape_dim_x
        bx_rest[:, :, :, 3]*=shape_dim_y
        return np.round(a_c_rest), np.round(anchor_boxes)

def get_shape_stub(shape_dim_x, shape_dim_y, space_x, space_y, densify_rate):
    step_x_norm = space_x/shape_dim_x
    step_y_norm = space_y/shape_dim_y
    dim_x = int(1/step_x_norm)
    dim_y = int(1/step_y_norm)
    return [dim_y, dim_x, densify_rate**2]

def transform_ltbr_to_lbwh(box):
    # Transforms from (left, top, bottom, right) - (0, 0) top left, to (left, bottom, width, height) - (0, 0) top left
    # Used to plot using default rectangle plot in Matplotlib
    # Coordinates stay consistent with OpenCV
    width = np.abs(box[2] - box[0])
    height = np.abs(box[3] - box[1])
    c_p = [box[0], box[1] + height, width, height]
    return c_p

def compute_mAP(imgs, true, preds, normalised = False):
    DEBUG = False
    mAP = []
    for i in range(len(imgs)):
        if len(true[i]) == 0:
            if len(preds[i]) > 0:
                mAP.append(0.0)
            else:
                continue
        i_c = np.squeeze(imgs[i]).shape
        h,w, _ = i_c
        img_t = np.zeros((i_c[0], i_c[1], 1))
        img_p = img_t.copy()
        im_out = img_t.copy()
        # True
        for box in true[i]:
            if DEBUG: print('Bt', box, np.tile((h, w), 2))
            if normalised: 
                box = np.multiply(np.array(box),np.tile((h, w), 2)[::-1])
            if DEBUG: print('At', box)
            cv2.rectangle(img_t, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), color = 1, thickness = -1)
        for box in preds[i]:
            if DEBUG: print('Bp', box, np.tile((h, w), 2))
            if normalised: 
                box = np.multiply(np.array(box),np.tile((h, w), 2)[::-1])
            if DEBUG: print('Ap',box)
            if not np.sum(np.array(box) < 0) > 0:
                cv2.rectangle(img_p, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), color = 1, thickness = -1)
        im_out += img_t
        im_out += img_p
        if np.sum(im_out > 0) > 0:
            mAP.append(np.sum(im_out == 2)/np.sum(im_out > 0))
    if len(mAP) > 0:
        return np.mean(mAP)
    else:
        return 1.0

def compute_iou_np(bboxes1, bboxes2):
    # Extracted from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum(0, yB - yA)*np.maximum(0, xB-xA)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    # Fix divide by 0 errors
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 0.00001)
    return np.clip(iou, 0, 1)

def non_max_suppression(boxes, overlapThresh):
    # Extracted from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	pick = []
 
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1) * (y2 - y1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1)
		h = np.maximum(0, yy2 - yy1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return pick

def encode(anchors_all, boxes, threshold):
    global SCALE_FACTOR, EPSILON

    boxes = np.array(boxes).copy()
    locs = np.zeros((anchors_all.shape[0], 4))
    confs = np.zeros((anchors_all.shape[0], 1))
    if not len(boxes) == 0:
        iou_mat = compute_iou_np(anchors_all,np.array(boxes))
        # n = len(boxes)
        # Compute anchor that gives max IOU for every box [N]
        max_iou, max_iou_ids = np.max(iou_mat, axis = 0), np.argmax(iou_mat, axis = 0) 
        # Compute box that gives max IOU for every anchor [num_anchors]
        max_obj_iou, max_obj_iou_ids = np.max(iou_mat, axis = 1), np.argmax(iou_mat, axis = 1) 
        
        # Set to -1 (no match) if anchor IOU < 0.35
        max_obj_iou_ids = np.squeeze(max_obj_iou_ids)
        max_iou_ids = np.squeeze(max_iou_ids)

        # if (len(max_iou_ids) != n) or (len(max_obj_iou_ids) != anchors_all.shape[0]):
        #     raise ValueError('Invalid shapes')

        threshold_ids = max_obj_iou < threshold
        ids_out = max_obj_iou_ids.copy()
        ids_out[threshold_ids] = -1
        # Zero out forced match
        ids_out[max_iou_ids] = -1

        # Get corresponding anchor locs
        anchor_boxes = anchors_all[ids_out >= 0, :]

        centers_a = np.array(anchor_boxes[:, 2:] + anchor_boxes[:, :2]).astype(np.float32)/2
        w_h_a = np.array(anchor_boxes[:, 2:] - anchor_boxes[:, :2]).astype(np.float32)
        w_h_a += EPSILON

        select_ids = ids_out[ids_out >= 0]
        select_boxes = np.take(boxes, select_ids, axis = 0)

        centers = np.array(select_boxes[:, :2] + select_boxes[:, 2:]).astype(np.float32)/2
        w_h = np.array(select_boxes[:, 2:] - select_boxes[:, :2]).astype(np.float32)
        w_h += EPSILON
        
        cxcy_out = (centers - centers_a)/w_h_a
        cxcy_out*=SCALE_FACTOR[0]
        wh_out = np.log(w_h/w_h_a)
        wh_out*=SCALE_FACTOR[1]

        cat_items = np.concatenate((cxcy_out, wh_out), axis = -1)
        locs[ids_out >= 0] = cat_items
        confs[ids_out >= 0] = 1
        # NOTE: confs is a N x 1 matrix (not one-hot)
    return locs, np.squeeze(confs)

def encode_batch(anchors, boxes, threshold):
    out_locs = []
    out_confs = []
    for i in range(len(boxes)):
        l, c = encode(anchors, boxes[i], threshold)
        out_locs.append(l)
        out_confs.append(c)
    return np.array(out_locs), np.array(out_confs)[:, :, np.newaxis]

def decode_batch(anchors, locs, confs, min_conf = 0.5):
    out_boxes = []
    for i in range(len(locs)):
        b, _, _ = decode(anchors, np.squeeze(locs[i]), np.squeeze(confs[i]), min_conf = min_conf, do_nms = True)
        out_boxes.append(b)
    return out_boxes

def decode(anchor_boxes, locs, confs, min_conf = 0.05, keep_top = 400, nms_thresh = 0.3, do_nms = True):
    # NOTE: confs is a N x 2 matrix
    global SCALE_FACTOR

    centers_a = np.array(anchor_boxes[:, 2:] + anchor_boxes[:, :2])/2
    w_h_a = np.array(anchor_boxes[:, 2:] - anchor_boxes[:, :2])

    cxcy_in = locs[:, :2]
    wh_in = locs[:, 2:]

    cxcy_in/=SCALE_FACTOR[0]
    wh_in/= SCALE_FACTOR[1]
    
    wh = np.exp(wh_in)*w_h_a
    cxcy = cxcy_in*w_h_a + centers_a

    boxes_out = np.concatenate([cxcy-wh/2, cxcy+wh/2], axis = -1)

    # Get only if confidence > 0.05 & keep top 400 boxes
    conf_ids = np.squeeze(np.argwhere(confs[:, 1] > min_conf))
    conf_merge = np.reshape(np.stack((conf_ids, confs[conf_ids, 1]), axis = -1), (-1, 2))
    conf_merge = conf_merge[conf_merge[:, 1].argsort()[::-1]]
    conf_merge = conf_merge[:keep_top, :]
    conf_ids, conf_vals = conf_merge[:, 0].astype(int), conf_merge[:, 1]
    # Run NMS on extracted boxes
    boxes_out = boxes_out[np.array(conf_merge[:, 0], dtype = int)]
    if do_nms:
        keep = non_max_suppression(boxes_out, nms_thresh)
        return boxes_out[keep], conf_ids[keep], conf_vals[keep]
    else:
        return boxes_out, conf_ids, conf_vals