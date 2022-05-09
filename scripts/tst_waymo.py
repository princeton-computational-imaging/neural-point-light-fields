import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from NeuralSceneGraph.data_loader.load_waymo_od import load_waymo_od_data


tf.compat.v1.enable_eager_execution()

# Plot every i_plt image
i_plt = 1

start = 50
end = 60

frames_path = '/home/julian/Desktop/waymo_open/segment-9985243312780923024_3049_720_3069_720_with_camera_labels.tfrecord'
basedir = '/home/julian/Desktop/waymo_open'
img_dir = '/home/julian/Desktop/waymo_open/tst_01'

cam_ls = ['front', 'front_left', 'front_right']

records = []
dir_list = os.listdir(basedir)
dir_list.sort()
for f in dir_list:
    if 'record' in f:
        records.append(os.path.join(basedir, f))



images = load_waymo_od_data(frames_path, selected_frames=[start, end])[0]
for i_record, tf_record in enumerate(records):
    dataset = tf.data.TFRecordDataset(tf_record, compression_type='')
    print(tf_record)

    for i, data in enumerate(dataset):
        if not i % i_plt:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            for index, camera_image in enumerate(frame.images):
                if camera_image.name in [1, 2, 3]:
                    img_arr = np.array(tf.image.decode_jpeg(camera_image.image))
                    # plt.imshow(img_arr, cmap=None)

                    cam_dir = os.path.join(img_dir, cam_ls[camera_image.name-1])
                    im_name = 'img_' + str(i_record) + '_' + str(i) + '.jpg'
                    im = Image.fromarray(img_arr)
                    im.save(os.path.join(cam_dir, im_name))

# frames = []
# max_frames=10
# i_plt = 100
#
# for i, data in enumerate(dataset):
#     if not i % i_plt:
#         frame = open_dataset.Frame()
#         frame.ParseFromString(bytearray(data.numpy()))
#
#         for index, camera_image in enumerate(frame.images):
#             if camera_image.name in [1, 2, 3]:
#                 plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=None)

            # layout = [3, 3, index+1]
            # ax = plt.subplot(*layout)
            #
            # plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=None)
            # plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
            # plt.grid(False)
            # plt.axis('off')

    # frames.append(frame)
    # if i >= max_frames-1:
    #     break

# frame = frames[0]

(range_images, camera_projections, range_image_top_pose) = (
    frame_utils.parse_range_image_and_camera_projection(frame))

print(frame.context)

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

plt.figure(figsize=(25, 20))

for index, image in enumerate(frame.images):
  show_camera_image(image, frame.camera_labels, [3, 3, index+1])


a = 0