from .computer_vision import ColorMatcher
from .analytical import BasisTransformer
from .models import MiniBatchRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
import numpy as np

def preprocess(x_test):
    cm = ColorMatcher()
    style_image = np.load('../data/style_image.npz')["style_image"]
    x_test_preprocessed = cm.fit_transform(style_image, x_test)
    transformer = BasisTransformer()
    x_test_transformed = transformer.fit_transform(x_test_preprocessed)
    x_test_flat = np.nan_to_num(x_test_transformed).reshape(
            (-1, x_test_transformed.shape[-1]))
    weights = np.load('../weights/regressor_weights.npz')
    sgd_regressor = MiniBatchRegressor(
        regressor=SGDRegressor(penalty='elasticnet', l1_ratio=0.11, max_iter = 5, tol = None),
        batch_size=1000,
        num_iters=50000
    )
    #this is so we can call predict
    sgd_regressor.fit(np.zeros((1, 8)), np.zeros((1, 1)).ravel())
    sgd_regressor.regressor.coef_ = weights["sgd"]
    pa_regressor = MiniBatchRegressor(
        regressor=PassiveAggressiveRegressor(C = .2, max_iter = 5, tol = None),
        batch_size=1000,
        num_iters=1000
    )
    #this is so we can call predict
    pa_regressor.fit(np.zeros((1, 8)), np.zeros((1, 1)).ravel())
    pa_regressor.regressor.coef_ = weights["pa"]
    x_test_extended = np.zeros((len(x_test), 256, 256, 2))
    x_test_extended[:, :, :, 0] = sgd_regressor.predict_images(x_test_transformed)
    x_test_extended[:, :, :, 1] = pa_regressor.predict_images(x_test_transformed)
    return x_test_extended
