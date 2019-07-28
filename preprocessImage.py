import keras
from keras import backend as K
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline #notebook command

mobile = keras.applications.mobilenet.MobileNet()
#call to applications.mobilenet.Mobilenet() in keras gives you a copy of a previously trained mobilenet model that has weights from when it was trained using imageNet images 

#function takes in an image file and preprocesses it to a file that the model accepts
def prepare_image(file):    
    img_path = 'MobileNet-images/' #path to images
    img = image.load_img(img_path + file, target_size=(224, 224)) #keras function: image.load accepts image file + target size; 224 is standard 
    img_array = image.img_to_array(img) #keras function to convert image to array
    img_array_expanded_dims = np.expand_dims(img_array, axis=0) 
    #img_array_expanded_dims results in a numpy array
    return keras.application.mobilenet.preprocess_input(img_array_expanded_dims) #preprocesses given image data to the same format of the image data mobilenet was originally trained on; scales pixel values of image [-1,1]

   
    
    
