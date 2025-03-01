import contextlib
import gc
import vmbpy

class VXCam(contextlib.ExitStack):

    def __init__(self, cam_num = 0):
        super().__init__()
        self.streaming = False
        self._binning = 1
        self._pad_left = 0
        self._pad_right = 0
        self._pad_top = 0
        self._pad_bottom = 0
        self._height = 0
        self._width = 0
        self._latest = None
        self.cam_num = cam_num # for multi-camera support

    # Reversing
    @property
    def reverse_x(self) -> bool: return self.cam.get_feature_by_name('ReverseX').get()
    @reverse_x.setter
    def reverse_x(self, value): self.cam.get_feature_by_name('ReverseX').set(value)

    @property
    def reverse_y(self) -> bool: return self.cam.get_feature_by_name('ReverseY').get()
    @reverse_y.setter
    def reverse_y(self, value): self.cam.get_feature_by_name('ReverseY').set(value)

    # Pixel format
    @property
    def pixel_format(self) -> vmbpy.PixelFormat: return self.cam.get_pixel_format()
    @pixel_format.setter
    def pixel_format(self, value): self.cam.set_pixel_format(value)

    # Gain
    @property
    def gain_db(self) -> float: return self.cam.get_feature_by_name('Gain').get()
    @gain_db.setter
    def gain_db(self, value): self.cam.get_feature_by_name('Gain').set(value)
    
    # Exposure
    @property
    def exposure_us(self) -> float: return self.cam.get_feature_by_name('ExposureTime').get()
    @exposure_us.setter
    def exposure_us(self, value): self.cam.get_feature_by_name('ExposureTime').set(value)

    # Auto exposure
    @property
    def auto_exposure(self) -> str: return self.cam.get_feature_by_name('ExposureAuto').get()
    @auto_exposure.setter
    def auto_exposure(self, value): self.cam.get_feature_by_name('ExposureAuto').set(value)
    @property
    def auto_exposure_min_us(self) -> float: return self.cam.get_feature_by_name('ExposureAutoMin').get()
    @auto_exposure_min_us.setter
    def auto_exposure_min_us(self, value): self.cam.get_feature_by_name('ExposureAutoMin').set(value)
    @property
    def auto_exposure_max_us(self) -> float: return self.cam.get_feature_by_name('ExposureAutoMax').get()
    @auto_exposure_max_us.setter
    def auto_exposure_max_us(self, value): self.cam.get_feature_by_name('ExposureAutoMax').set(value)

    # White balance
    @property
    def white_balance(self) -> str: return self.cam.get_feature_by_name('BalanceWhiteAuto').get()
    @white_balance.setter
    def white_balance(self, value): self.cam.get_feature_by_name('BalanceWhiteAuto').set(value)

    # Height and width
    @property
    def width(self) -> int: return self.cam.get_feature_by_name('Width').get()
    @property
    def height(self) -> int: return self.cam.get_feature_by_name('Height').get()

    # Binning and padding
    @property
    def binning(self) -> int: return self._binning
    @binning.setter
    def binning(self, value): self._set_roi(binning=value)
    @property
    def pad_left(self) -> int: return self._pad_left
    @pad_left.setter
    def pad_left(self, value): self._set_roi(pad_left=value)
    @property
    def pad_right(self) -> int: return self._pad_right
    @pad_right.setter
    def pad_right(self, value): self._set_roi(pad_right=value)
    @property
    def pad_top(self) -> int: return self._pad_top
    @pad_top.setter
    def pad_top(self, value): self._set_roi(pad_top=value)
    @property
    def pad_bottom(self) -> int: return self._pad_bottom
    @pad_bottom.setter
    def pad_bottom(self, value): self._set_roi(pad_bottom=value)

    # Set ROI and binning
    def _set_roi(self, 
        binning:int = None, 
        pad_left:int = None, 
        pad_right:int = None, 
        pad_top:int = None, 
        pad_bottom:int = None):

        # Only enable setting if the stream isn't running
        if self.streaming: return
        
        # Update globals
        if binning is not None: self._binning = binning
        if pad_left is not None: self._pad_left = pad_left
        if pad_right is not None: self._pad_right = pad_right
        if pad_top is not None: self._pad_top = pad_top
        if pad_bottom is not None: self._pad_bottom = pad_bottom
     
        # Set binning
        self.cam.get_feature_by_name('BinningVertical').set(self._binning)
        self.cam.get_feature_by_name('BinningHorizontal').set(self._binning)
        
        # Reset offsets to capture currently available width/height
        self.cam.get_feature_by_name('OffsetX').set(0)
        self.cam.get_feature_by_name('OffsetY').set(0)
        w = self.cam.get_feature_by_name('Width').get_range()[1]
        h = self.cam.get_feature_by_name('Height').get_range()[1]
        self.cam.get_feature_by_name('Width').set(w)
        self.cam.get_feature_by_name('Height').set(h)
        
        # Finally set ROI
        x0 = w * self._pad_left // 2 * 2
        x1 = w * (1 - self._pad_right - self._pad_left) // 8 * 8
        y0 = h * self._pad_top // 2 * 2
        y1 = h * (1 - self._pad_bottom - self._pad_top) // 8 * 8
        self.cam.get_feature_by_name('Width').set(x1)
        self.cam.get_feature_by_name('OffsetX').set(x0)
        self.cam.get_feature_by_name('Height').set(y1)
        self.cam.get_feature_by_name('OffsetY').set(y0)

    # Frame ready callback
    def __call__(self, cam:vmbpy.Camera, stream:vmbpy.Stream, frame:vmbpy.Frame):
        if frame.get_status() == vmbpy.FrameStatus.Complete: self._latest = frame
        self.cam.queue_frame(frame)

    # Enter additional contexts on __enter__ and reset ROI
    def __enter__(self):
        super().__enter__()
        try:
            self.vmb = self.enter_context(vmbpy.VmbSystem.get_instance())
            self.cam = self.enter_context(self.vmb.get_all_cameras()[self.cam_num])
            print(len(self.vmb.get_all_cameras()))
        except:
            raise
        self._set_roi()
        return self
    
    # Start streaming
    def start(self):
        self.streaming = True
        self.cam.start_streaming(handler=self, buffer_count=10)
    
    # Pop latest frame
    def pop(self, opencv_format:vmbpy.PixelFormat = None):
        cpy = self._latest
        self._latest = None
        if cpy is None: 
            return
        elif opencv_format is not None: 
            return cpy.convert_pixel_format(opencv_format).as_opencv_image()
        return cpy

    # Exit out of other contexts and clear memory
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.streaming = False
        self.cam.stop_streaming()
        self.cam.__exit__(exc_type, exc_val, exc_tb)
        self.vmb.__exit__(exc_type, exc_val, exc_tb)
        gc.collect()