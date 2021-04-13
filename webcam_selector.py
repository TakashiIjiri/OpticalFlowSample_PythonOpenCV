import tkinter as tk
from PIL import Image, ImageTk
import cv2

# NOTE : install cv2, Pillow, (and tkinter if necessary)


class WebcomSelector(tk.Frame):
    def __init__(self, _root):
        super().__init__(_root)
        self.root = _root
        self.pack()
        self.root.title("Web com selector")

        # 利用できるwebcamをすべて検索 [id, img]
        self.webcam_ids = []

        for i in range(10):
            c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ret, frame = c.read()
            if ret:
                print(frame.shape, c.getBackendName())
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ratio = frame.shape[0] / frame.shape[1]
                frame = cv2.resize(frame, (300, int(300*ratio)), interpolation=cv2.INTER_LANCZOS4)
                self.webcam_ids.append([i, frame])
            c.release()
            cv2.destroyAllWindows()

        # dialog 作成
        # add radio button
        sub_frame = tk.Frame(self.root)
        sub_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        btn_ok = tk.Button(sub_frame, text="OK", width=8, command=self.ok_button_pressed)
        btn_ok.pack(side=tk.LEFT)

        self.var_idx = tk.IntVar()
        self.var_idx.set(0)
        for i in range(len(self.webcam_ids)):
            s = str(self.webcam_ids[i][0])
            rb = tk.Radiobutton(sub_frame, text=s, value=i, variable=self.var_idx, command=self.update)
            rb.pack(side=tk.LEFT)

        # add panel
        self.main_panel = tk.Label(self.root)
        self.main_panel.pack()
        self.update()

    def update(self):
        idx = self.var_idx.get()
        print(idx)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(self.webcam_ids[idx][1]))
        self.main_panel.imgtk = imgtk
        self.main_panel.configure(image=imgtk)

    def ok_button_pressed(self):
        print(self.var_idx.get())
        self.root.destroy()
        self.root.quit()

    def get_selected_idx(self):
        return self.var_idx.get()


def select_webcam_idx():
    root = tk.Tk()
    dlg = WebcomSelector(root)
    dlg.mainloop()
    return dlg.get_selected_idx()


if __name__ == "__main__":

    idx = select_webcam_idx()

    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    while 1:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break

    print("finish")
