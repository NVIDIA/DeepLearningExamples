# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""WBF for test-time augmentation."""
import tensorflow as tf


def vectorized_iou(clusters, detection):
  """Calculates the ious for box with each element of clusters."""
  x11, y11, x12, y12 = tf.split(clusters[:, 1:5], 4, axis=1)
  x21, y21, x22, y22 = tf.split(detection[1:5], 4)

  xa = tf.maximum(x11, x21)
  ya = tf.maximum(y11, y21)
  xb = tf.minimum(x12, x22)
  yb = tf.minimum(y12, y22)

  inter_area = tf.maximum((xb - xa), 0) * tf.maximum((yb - ya), 0)

  boxa_area = (x12 - x11) * (y12 - y11)
  boxb_area = (x22 - x21) * (y22 - y21)

  iou = inter_area / (boxa_area + boxb_area - inter_area)

  return iou


def find_matching_cluster(clusters, detection):
  """Returns the index of the highest iou matching cluster for detection."""
  if not clusters:
    return -1
  ious = vectorized_iou(tf.stack(clusters), detection)
  ious = tf.reshape(ious, [len(clusters)])
  if tf.math.reduce_max(ious) < 0.55:
    # returns -1 if no iou is higher than 0.55.
    return -1
  return tf.argmax(ious)


def weighted_average(samples, weights):
  return tf.math.reduce_sum(samples * weights) / tf.math.reduce_sum(weights)


def average_detections(detections, num_models):
  """Takes a list of detections and returns the average, both in box co-ordinates and confidence."""
  num_detections = len(detections)
  detections = tf.stack(detections)
  return [
      detections[0][0],
      weighted_average(detections[:, 1], detections[:, 5]),
      weighted_average(detections[:, 2], detections[:, 5]),
      weighted_average(detections[:, 3], detections[:, 5]),
      weighted_average(detections[:, 4], detections[:, 5]),
      tf.math.reduce_mean(detections[:, 5]) * min(1, num_detections/num_models),
      detections[0][6],
  ]


def ensemble_detections(params, detections, num_models):
  """Ensembles a group of detections by clustering the detections and returning the average of the clusters."""
  all_clusters = []

  for cid in range(params['num_classes']):
    indices = tf.where(tf.equal(detections[:, 6], cid))
    if indices.shape[0] == 0:
      continue
    class_detections = tf.gather_nd(detections, indices)

    clusters = []
    cluster_averages = []
    for d in class_detections:
      cluster_index = find_matching_cluster(cluster_averages, d)
      if cluster_index == -1:
        clusters.append([d])
        cluster_averages.append(average_detections([d], num_models))
      else:
        clusters[cluster_index].append(d)
        cluster_averages[cluster_index] = average_detections(
            clusters[cluster_index], num_models)

    all_clusters.extend(cluster_averages)

  all_clusters.sort(reverse=True, key=lambda d: d[5])
  return tf.stack(all_clusters)
