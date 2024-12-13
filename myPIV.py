import numpy as np
import cv2
from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
import pathlib

class myParticleImageVelocimetry:
    def __init__(self, video_name, winsize, searchsize, overlap, fps):
        self.video_name = video_name
        self.winsize = winsize
        self.searchsize = searchsize
        self.overlap = overlap
        self.fps = fps
        self.dt = 1 / fps
        self.Nframes = 0

        self.xmin = 0
        self.xmax = 1280
        self.ymin = 0
        self.ymax = 720
        self.lengthScale = 1

    def analyze_frame(self, i):
        frame_a_name = f'{self.video_name}/frame_{i:04d}.png'
        frame_b_name = f'{self.video_name}/frame_{i+1:04d}.png'
        frame_a = tools.imread(frame_a_name)
        frame_b = tools.imread(frame_b_name)

        u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), frame_b.astype(np.int32),
            window_size=self.winsize, overlap=self.overlap, dt=self.dt, search_area_size=self.searchsize, sig2noise_method='peak2mean')

        x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=self.searchsize, overlap=self.overlap)

        # Validation and filtering
        u0_std = np.nanstd(u0)
        u0_mean = np.nanmean(u0)
        v0_std = np.nanstd(v0)
        v0_mean = np.nanmean(v0)
        
        invalid_mask = validation.global_val(u0, v0, [0, int(u0_mean + u0_std * 1)], [0, int(v0_mean + v0_std * 1)])
        u2, v2 = filters.replace_outliers(u0, v0, flags=invalid_mask, method='localmean', max_iter=3, kernel_size=3)

        # Convert and transform
        x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=self.lengthScale)

        # Transform coordinates if necessary
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

        # Save results
        tools.save(f'{self.video_name}/{i:04d}.txt', x, y, u3, v3, invalid_mask)

    def run_analysis(self):
        for i in range(1, self.Nframes):
            self.analyze_frame(i)

    def summarize(self):
        file_path = self.video_name + '/0001.txt'
        data = np.loadtxt(file_path)
        x, y, u, v, invalid_mask = data[:,0], data[:,1], data[:,2], data[:,3], (np.round((data[:,4]-1)*-1)).astype(int)


        for i in np.arange(2, self.Nframes):
            data = np.loadtxt(self.video_name + '/{:04d}.txt'.format(i))
            x1, y1, u1, v1, invalid_mask1 = data[:,0], data[:,1], data[:,2], data[:,3], (np.round((data[:,4]-1)*-1)).astype(int)
            u += u1 * invalid_mask1
            v += v1 * invalid_mask1
            invalid_mask += invalid_mask1
        u /= invalid_mask.astype(float)+1e-10
        v /= invalid_mask.astype(float)+1e-10

        invalid_mask[invalid_mask==0] = -1
        invalid_mask[invalid_mask>0] = 0
        invalid_mask[invalid_mask==-1] = 0

        invalid_mask[(x * self.lengthScale < self.xmin) | (x * self.lengthScale > self.xmax)] = 1
        invalid_mask[(y * self.lengthScale > self.frame_a_size[0] - self.ymin) | (y * self.lengthScale < self.frame_a_size[0] - self.ymax)] = 1

        tools.save(self.video_name + 'sum.txt', x, y, u, v, invalid_mask)

    def movie_to_images(self, videoName):
        cap = cv2.VideoCapture(videoName)

        if not cap.isOpened():
            print("Can not read a movie")

        self.Nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"動画の総フレーム数: {self.Nframes}")

        for frame_to_save in np.arange(self.Nframes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_save)
            ret, frame = cap.read() # read a frame
            output_image_path = self.video_name + f"/frame_{frame_to_save + 1:04d}.png"
            cv2.imwrite(output_image_path, frame)
        cap.release()

    def setAnalyzeRegion(self, xmin, xmax, ymin, ymax):
        # 解析領域の設定
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax 

        frame_a_name = self.video_name + '/frame_0001.png' 
        frame_b_name = self.video_name + '/frame_0002.png' 
        frame_a  = tools.imread(frame_a_name)
        frame_b  = tools.imread(frame_b_name)

        self.frame_a_size = frame_a.shape
        self.frame_b_size = frame_b.shape

        fig,ax = plt.subplots(1,2,figsize=(12,10))
        ax[0].imshow(frame_a,cmap=plt.cm.gray)
        ax[1].imshow(frame_b,cmap=plt.cm.gray)
        ax[0].plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')
        ax[1].plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')
        plt.show()

        print("Frame A のピクセルサイズ: 高さ {0} ピクセル, 幅 {1} ピクセル".format(self.frame_a_size[0], self.frame_a_size[1]))
        print("Frame B のピクセルサイズ: 高さ {0} ピクセル, 幅 {1} ピクセル".format(self.frame_b_size[0], self.frame_b_size[1]))

    def showField(self, i):
        if(i == 0):
            fieldPath = self.video_name + 'sum.txt'
            imagePath = self.video_name + '/frame_{:04d}.png'.format(1)
        else:
            fieldPath = self.video_name + '/{:04d}.txt'.format(i)
            imagePath = self.video_name + '/frame_{:04d}.png'.format(i)
        fig, ax = plt.subplots(figsize=(8,8))
        tools.display_vector_field(
            pathlib.Path(fieldPath),
            ax=ax, scaling_factor=1,
            scale=1e3, # scale defines here the arrow length
            width=0.0035, # width is the thickness of the arrow
            show_invalid=False,
            window_size = 64 / self.lengthScale,
            on_img=True, # overlay on the image
            image_name= imagePath,
        )
        plt.show()
