from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img


class Predictor(object):
    """initializer for Predictor"""
    def __init__(self, model_path):
        super(Predictor, self).__init__()
        self.model = load_model(model_path)

    def predict(self, image):
        """
        Process image to be evaluated by the network and predict class of image
        Args:
            image: the image we want to predict the class of

        Returns:
            Predicted class
        """
        # dimensions of our images.
        img = load_img(image, target_size=self.model.input_shape[1:3])
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
       
        eval_generator = ImageDataGenerator(rescale=1./255,
                                            samplewise_std_normalization=True,
                                            featurewise_std_normalization=True)
        img = eval_generator.standardize(img) 

        return self.model(img, training=False)[0][0]
