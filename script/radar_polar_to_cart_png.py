import argparse
import os
import numpy as np
import cv2

from datetime import datetime
import time

###############################################
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .125 # m
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 2001 # pixels
# data path
path = "/mnt/Disk2/Oxford_radar/"
# output folder name
directory = "radar_cart_0.125_2001_timestamped"
###############################################

radar_init_t = 0
radar_end_t = 0

def UnixTimeToSec(unix_timestamp):
    time = datetime.fromtimestamp(unix_timestamp / 1000000)
    s = unix_timestamp % 1000000
    sec_timestamp = time.hour*3600 + time.minute*60 + time.second + (float(s)/1000000)
    return sec_timestamp


def main():
    dir_path = os.path.join(path, directory)
    os.mkdir(dir_path)
    print("Directory '%s' created" %directory)

    for foldername in os.listdir(path):
        print(foldername)

        if foldername != '2019-01-10-12-32-52-radar-oxford-10k':
          continue

        dir_path = os.path.join(path+'/'+directory, foldername)
        os.mkdir(dir_path)
        print("Directory '%s' created" %foldername)

        timestamps_path = path+'/'+foldername+'/radar.timestamps'

        if not os.path.isfile(timestamps_path):
            raise IOError("Could not find timestamps file")

        interpolate_crossover = True

        title = "Radar Visualisation Example"

        radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

        radar_init_t = UnixTimeToSec(radar_timestamps[0])
        radar_end_t = UnixTimeToSec(radar_timestamps[np.size(radar_timestamps)-1])
        radar_t_list = np.array([], dtype='string')

        for radar_timestamp in radar_timestamps:
            curr_radar_t = UnixTimeToSec(radar_timestamp)-radar_init_t
            curr_radar_t_str = str("%010.5f"%curr_radar_t)
            radar_t_list = np.append(radar_t_list, [curr_radar_t_str], axis=0)

            filename = path + '/' + foldername + "/radar/" + str(radar_timestamp) + '.png'

            if not os.path.isfile(filename):
                print("Could not find radar example: {}".format(filename))

            ##########################################################################################
            radar_resolution = np.array([0.0432], np.float32)
            encoder_size = 5600

            raw_example_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            timestamps = raw_example_data[:, :8].copy().view(np.int64)
            azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
            valid = raw_example_data[:, 10:11] == 255
            fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
            ##########################################################################################

            ##########################################################################################
            if (cart_pixel_width % 2) == 0:
                cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
            else:
                cart_min_range = cart_pixel_width // 2 * cart_resolution
            coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
            Y, X = np.meshgrid(coords, -coords)
            sample_range = np.sqrt(Y * Y + X * X)
            sample_angle = np.arctan2(Y, X)
            sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

            # Interpolate Radar Data Coordinates
            azimuth_step = azimuths[1] - azimuths[0]
            sample_u = (sample_range - radar_resolution / 2) / radar_resolution
            sample_v = (sample_angle - azimuths[0]) / azimuth_step

            # We clip the sample points to the minimum sensor reading range so that we
            # do not have undefined results in the centre of the image. In practice
            # this region is simply undefined.
            sample_u[sample_u < 0] = 0

            if interpolate_crossover:
                fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
                sample_v = sample_v + 1

            polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
            cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)

            ##########################################################################################

            # Combine polar and cartesian for visualisation
            # The raw polar data is resized to the height of the cartesian representation
            downsample_rate = 4
            fft_data_vis = fft_data[:, ::downsample_rate]
            resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
            fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
            vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation

            cart_img = cart_img * 255.0

            cv2.imwrite(path + "/" + directory + "/" + foldername + '/' + curr_radar_t_str + '.png', cart_img) # write
            np.savetxt(path + "/" + directory + "/" + foldername + '/radar_t_list', radar_t_list, delimiter=' ', fmt='%s')
            cv2.waitKey(1)

if __name__ == '__main__':
  main()
