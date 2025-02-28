import cv2
import numpy as np
import wandb


class Visualizer:
    def __init__(
        self,
        freq=1,
        num_videos=1,
        imsize=(200, 200),
        fps=80,
        format="mp4",
        hide_info=False,
    ):
        self._freq = freq
        self._num_videos = num_videos
        self._imsize = imsize
        self._fps = fps
        self._format = format
        self._hide_info = hide_info
        self._frames = []

        self._info_keys = [
            "step",
            "reward",
            "success",
            "task_id",
            "stages_completed",
            "x_velocity",
            "policy_id",
            "real_policy_id",
            "obj_to_target",
        ]
        self._vis_info_prefix = "VIS:"
        self._text_x = 10
        self._text_y = 20
        self._text_y_gap = 20

    @staticmethod
    def put_text(img, text, pos, size=0.5):
        return cv2.putText(
            img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    @property
    def num_videos(self):
        return self._num_videos

    def do_visualization(self, epoch):
        if self._freq <= 0:
            return False
        elif epoch == 0:
            return True
        else:
            return epoch % self._freq == 0

    def reset(self):
        self._frames = []

    def add(self, frames, info=None):
        if not isinstance(frames, list):
            frames = [frames]
        for frame in frames:
            frame = cv2.resize(frame, self._imsize)
            if info is not None and not self._hide_info:
                vis_data = {k: info.get(k, "") for k in self._info_keys}
                vis_data.update(
                    {
                        k[len(self._vis_info_prefix) :]: info[k]
                        for k in info
                        if k.startswith(self._vis_info_prefix)
                    }
                )
                x, y = self._text_x, self._text_y
                for k, v in vis_data.items():
                    if isinstance(v, float):
                        text = "{}: {:.4f}".format(k, v)
                    else:
                        text = "{}: {}".format(k, v)
                    frame = self.put_text(frame, text, (x, y))
                    y += self._text_y_gap
            self._frames.append(frame)

    def get_video(self, fps=None, format=None):
        fps = fps or self._fps
        format = format or self._format
        frames = np.array(self._frames).transpose(0, 3, 1, 2)
        return {"Video": wandb.Video(frames, fps=fps, format=format)}
