from skimage import io
from skimage.viewer import ImageViewer

hulk = io.imread('hulk.jpg')

viewer = ImageViewer(hulk)

viewer.show()
