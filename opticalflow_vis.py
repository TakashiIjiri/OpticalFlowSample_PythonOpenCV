import tkinter
import cv2
import numpy as np
import webcam_selector
from PIL import Image, ImageTk



class VideoViewer(tkinter.Frame):

    def __init__(self, root=None,  webcam_idx=0):
        super().__init__(root)
        root.geometry('640x520')
        self.root = root
        self.pack()

        # MainPanel を 全体に配置
        self.mainpanel = tkinter.Label(root)
        self.mainpanel.pack(expand=1)

        self.MAX_CORNER = 20 * 20
        #ボタンを作る（現状特に機能なし）
        self.btngray  = tkinter.Button(root, text='grid points'   , command=self.func_switch_grid)
        self.btncolor = tkinter.Button(root, text='feature points', command=self.func_switch_feature)
        self.btngray.pack(side="left")
        self.btncolor.pack(side="left")
        self.track_mode = "grid"

        #open web cam stream (複数webcamがある場合は，引数を変更する)
        self.cap = cv2.VideoCapture( webcam_idx )
        ret, frame = self.cap.read()
        if ret == 0 :
            print("failed to webcam")
            exit()
        self.do_gray = False

        self.width = 640
        self.height = 480
        self.opt_img = np.zeros( (self.height, self.width, 3), np.uint8)
        self.update_counter = 0
        self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        feature_params = dict( maxCorners = self.MAX_CORNER, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
        self.track_points = cv2.goodFeaturesToTrack(self.prev_frame, mask = None, **feature_params)
        self.color = np.random.randint(0,255,(self.MAX_CORNER,3))

        self.grid_points = np.zeros((self.MAX_CORNER, 1, 2), dtype=np.float32)
        for y in range(20):
            for x in range(20):
                i = y*20+x
                self.grid_points[i,0,0] = (x+1)*32.0
                self.grid_points[i,0,1] = (y+1)*24.0

    def func_switch_grid(self):
        self.track_mode = "grid"

    def func_switch_feature(self):
        self.track_mode = "feature"


    def update_video(self):
        self.update_counter += 1
        self.root.geometry('700x520')

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_LANCZOS4)
        frame_vis  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.update_counter % 10 == 0 :
            print("update points -----------------------------------------------------------")
            self.opt_img = np.zeros( (self.height, self.width, 3), np.uint8)

            if self.track_mode == "feature" :
                feature_params = dict( maxCorners = self.MAX_CORNER, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
                self.track_points = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            elif self.track_mode == "grid" :
                # gen grid points
                self.track_points = self.grid_points.copy()

        print(self.track_points.shape, self.track_points.dtype)
        lk_params = dict(
                winSize  = (51,51), maxLevel = 0,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        new_track_points, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, frame_gray,
                self.track_points, None, **lk_params)
        self.prev_frame = frame_gray.copy()

        for i in range(self.track_points.shape[0]) :
            if st[i] != 1 : continue
            a, b = self.track_points[i,0]
            c, d = new_track_points[i,0]
            self.opt_img = cv2.line(self.opt_img, (a,b),(c,d), self.color[i].tolist(), 2)
        self.track_points = new_track_points.copy()

        # self.opt_img = self.opt_img // 10 * 9
        frame_vis[self.opt_img[:,:,0]>0] = 0
        frame_vis = frame_vis + self.opt_img

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_vis))
        self.mainpanel.imgtk = imgtk
        self.mainpanel.configure(image=imgtk)

        #33ms後に自分自身を呼ぶ
        self.mainpanel.after(33, self.update_video)


if __name__ == "__main__":

    # webcamを選択する場合
    #idx = webcam_selector.select_webcam_idx()
    #dlg = VideoViewer(root=tkinter.Tk(), webcam_idx = idx)

    dlg = VideoViewer(root=tkinter.Tk(), webcam_idx = 0)
    dlg.update_video()
    dlg.mainloop()
