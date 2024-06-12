from record_video import record_camera
from fit_3d import do_line_fitting_on_image, do_hand_landmarker

if __name__ == '__main__':
    name = record_camera()
    do_hand_landmarker(name)
    do_line_fitting_on_image(name)