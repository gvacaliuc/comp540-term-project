def predict(x_test):
    if is_sparse(x_test):
        return (model.predict(otsu(x_test).reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
    else:
        if is_black_white(X_test):
            return (model.predict(otsu(x_test).reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
        else:
            if has_white_background(x_test):
                if (has_purple_foreground(x_test)):
                    return (model.predict(otsu(1 - x_test, lower_bound = .7).reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
                else:
                    return (model.predict(otsu(1 - x_test).reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
            else:
                return (model.predict(otsu(1 - x_test).reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
