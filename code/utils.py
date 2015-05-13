import numpy
import theano.tensor as T

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def mean_subtraction_normalization(data, data_shape):
    """ Function used for normalization by subtraction of
        mean value in the same spatial position.
        We use a 3D mean filter for implementation.
        data: input 4D theano.tensor
    """
    filter_shape = (data_shape[1], data_shape[1], 1, 1) 
    mean_filter = theano.shared(
        numpy.asarray(
            1./ data_shape[1] * numpy.ones(filter_shape),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    mean_tensor =  theano.tensor.nnet.conv.conv2d(
        input=data,
        filters=mean_filter,
        filter_shape=filter_shape,
        image_shape=data_shape
    )
    return (data - mean_tensor)

def local_responce_normalization(data, eps=0.001):
    """ Function used for local responce normalization. 
        data: input 4D theano.tensor
        eps: small constant in case the normalizer gets 0
    """
    normalizer = T.sqrt(eps + (data**2).sum(axis=1))
    return data / normalizer.dimshuffle(0,'x',1,2)

def max_tensor_scalar(a, b):
    """ Function to compare values of two tensor scalars.
        a, b: tensor scalars
    """
    return T.switch(a<b, b, a)

def local_responce_normalization_(data, k=2, n=5, alpha=0.0001, beta=0.75):
    """ Function for local responce normalization.
        data  : input 4D theano.tensor
        k     : constant number in denominator
        n     : receptive field channels
        alpha : coefficient
        beta  : exponential term
    """
    half = n // 2
    sq = T.sqr(data)
    b, ch, r, c = data.shape
    extra_channels = T.alloc(0., b, ch + 2*half, r, c)
    sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)
    scale = k
    for i in xrange(n):
        scale += alpha * sq[:,i:i+ch,:,:]
    scale = scale ** beta

    return data / scale

def relu(x):
    """ Function for rectified linear unit.
        Returns the maximum value of input and 0.
    """
    return T.switch(x<0, 0, x)

def unit_scaling(data, scale=255.0):
    """ Function used to normalize data, i.e.
        to divide the data by a certain factor. 
        data: numpy.ndarray.
        scale: scaling factor to be divided. """

    return data / scale 

def mean_subtraction_preprocessing(data):
    """ Function used to subtract the mean image
        of the dataset.
        data: numpy array of size n*m, representing
              n m-dim row vectors of image.
    """
    return (data - data.mean(axis=0))

def crop_images(data, image_shape, border_width=8, mode=0):
    """ Function used to crop the images by a certain border width.
        data         : input data, theano 4D tensor
        image_shape  : 4-tuple, (batch_size, num_channels, image_rows, image_cols)
        border_width : border width to be cropped, default value 8
        mode         : binary, 0 for random, 1 for centered crop.
    """
    row_step = image_shape[2] - border_width
    col_step = image_shape[3] - border_width
    output = T.alloc(0., image_shape[0], image_shape[1], row_step, col_step) 
    for i in range(image_shape[0]):           
        begin_idx = numpy.random.randint(border_width)
        print "beginning index: ", begin_idx
        output = T.set_subtensor(output[i,:,:,:], data[i,:,begin_idx:(begin_idx+row_step),begin_idx:(begin_idx+col_step)])
    return output 
