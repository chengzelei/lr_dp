# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for building and training NN models.
"""
from __future__ import division

import math
import random

import numpy
import tensorflow as tf

BATCH_COLLECTION_KEY = "batch"
PREV_EPOCH_DATA_COLLECTION_KEY = "prev_epoch"
VAR_BATCH_SUFFIX = "_batch_var_"
NOISE_BATCH_SUFFIX = "_batch_noise_"
VAR_PREV_SUFFIX = "_prev_var"
NOISE_PREV_SUFFIX = "_prev_noise"
NOISE_CUR_SUFFIX = "_cur_noise"
REUSE_NOISE_COEFF = "_reuse_noise_coeff"
NEW_NOISE_COEFF = "_new_noise_coeff"

# NUM_EPOCH_RESET=2

class LayerParameters(object):
  """class that defines a non-conv layer."""
  def __init__(self):
    self.name = ""
    self.num_units = 0
    self._with_bias = False
    self.relu = False
    self.gradient_l2norm_bound = 0.0
    self.bias_gradient_l2norm_bound = 0.0
    self.trainable = True
    self.weight_decay = 0.0


class ConvParameters(object):
  """class that defines a conv layer."""
  def __init__(self):
    self.patch_size = 5
    self.stride = 1
    self.in_channels = 1
    self.out_channels = 0
    self.with_bias = True
    self.relu = True
    self.max_pool = True
    self.max_pool_size = 2
    self.max_pool_stride = 2
    self.trainable = False
    self.in_size = 28
    self.name = ""
    self.num_outputs = 0
    self.bias_stddev = 0.1


# Parameters for a layered neural network.
class NetworkParameters(object):
  """class that define the overall model structure."""
  def __init__(self):
    self.input_size = 0
    self.projection_type = 'NONE'  # NONE, RANDOM, PCA
    self.projection_dimensions = 0
    self.default_gradient_l2norm_bound = 0.0
    self.layer_parameters = []  # List of LayerParameters
    self.conv_parameters = []  # List of ConvParameters


def GetTensorOpName(x):
  """Get the name of the op that created a tensor.

  Useful for naming related tensors, as ':' in name field of op is not permitted

  Args:
    x: the input tensor.
  Returns:
    the name of the op.
  """

  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]


def BuildNetwork(inputs, network_parameters):
  """Build a network using the given parameters.

  Args:
    inputs: a Tensor of floats containing the input data.
    network_parameters: NetworkParameters object
      that describes the parameters for the network.
  Returns:
    output, training_parameters: where the outputs (a tensor) is the output
      of the network, and training_parameters (a dictionary that maps the
      name of each variable to a dictionary of parameters) is the parameters
      used during training.
  """

  training_parameters = {}
  num_inputs = network_parameters.input_size
  outputs = inputs
  projection = None

  # First apply convolutions, if needed
  for conv_param in network_parameters.conv_parameters:
    outputs = tf.reshape(
        outputs,
        [-1, conv_param.in_size, conv_param.in_size,
         conv_param.in_channels])
    conv_weights_name = "%s_conv_weight" % (conv_param.name)
    conv_bias_name = "%s_conv_bias" % (conv_param.name)
    conv_std_dev = 1.0 / (conv_param.patch_size
                          * math.sqrt(conv_param.in_channels))
    conv_weights = tf.Variable(
        tf.truncated_normal([conv_param.patch_size,
                             conv_param.patch_size,
                             conv_param.in_channels,
                             conv_param.out_channels],
                            stddev=conv_std_dev),
        trainable=conv_param.trainable,
        name=conv_weights_name)
    conv_bias = tf.Variable(
        tf.truncated_normal([conv_param.out_channels],
                            stddev=conv_param.bias_stddev),
        trainable=conv_param.trainable,
        name=conv_bias_name)
    training_parameters[conv_weights_name] = {}
    training_parameters[conv_bias_name] = {}
    conv = tf.nn.conv2d(outputs, conv_weights,
                        strides=[1, conv_param.stride,
                                 conv_param.stride, 1],
                        padding="SAME")
    relud = tf.nn.relu(conv + conv_bias)
    mpd = tf.nn.max_pool(relud, ksize=[1,
                                       conv_param.max_pool_size,
                                       conv_param.max_pool_size, 1],
                         strides=[1, conv_param.max_pool_stride,
                                  conv_param.max_pool_stride, 1],
                         padding="SAME")
    outputs = mpd
    num_inputs = conv_param.num_outputs
    # this should equal
    # in_size * in_size * out_channels / (stride * max_pool_stride)

  # once all the convs are done, reshape to make it flat
  outputs = tf.reshape(outputs, [-1, num_inputs])

  # Now project, if needed
  if network_parameters.projection_type is not "NONE":
    projection = tf.Variable(tf.truncated_normal(
        [num_inputs, network_parameters.projection_dimensions],
        stddev=1.0 / math.sqrt(num_inputs)), trainable=False, name="projection")
    num_inputs = network_parameters.projection_dimensions
    outputs = tf.matmul(outputs, projection)

  # Now apply any other layers

  for layer_parameters in network_parameters.layer_parameters:
    num_units = layer_parameters.num_units
    hidden_weights_name = "%s_weight" % (layer_parameters.name)
    hidden_weights = tf.Variable(
        tf.truncated_normal([num_inputs, num_units],
                            stddev=1.0 / math.sqrt(num_inputs)),
        name=hidden_weights_name, trainable=layer_parameters.trainable)
    training_parameters[hidden_weights_name] = {}
    if layer_parameters.gradient_l2norm_bound:
      training_parameters[hidden_weights_name]["gradient_l2norm_bound"] = (
          layer_parameters.gradient_l2norm_bound)
    if layer_parameters.weight_decay:
      training_parameters[hidden_weights_name]["weight_decay"] = (
          layer_parameters.weight_decay)

    outputs = tf.matmul(outputs, hidden_weights)
    if layer_parameters.with_bias:
      hidden_biases_name = "%s_bias" % (layer_parameters.name)
      hidden_biases = tf.Variable(tf.zeros([num_units]),
                                  name=hidden_biases_name)
      training_parameters[hidden_biases_name] = {}
      if layer_parameters.bias_gradient_l2norm_bound:
        training_parameters[hidden_biases_name][
            "bias_gradient_l2norm_bound"] = (
                layer_parameters.bias_gradient_l2norm_bound)

      outputs += hidden_biases
    if layer_parameters.relu:
      outputs = tf.nn.relu(outputs)
    # num_inputs for the next layer is num_units in the current layer.
    num_inputs = num_units

  return outputs, projection, training_parameters


def BuildBatchVariables(num_batchs):
    for var in tf.trainable_variables():
        hidden_weights_name = GetTensorOpName(var)
        num_inputs, num_units = var.shape[0].value, var.shape[1].value

        prev_hidden_weights = tf.Variable(numpy.zeros((num_inputs, num_units), dtype=numpy.float32), 
                        name=hidden_weights_name+VAR_PREV_SUFFIX, trainable=False)
        tf.add_to_collection(PREV_EPOCH_DATA_COLLECTION_KEY, prev_hidden_weights)
        
        prev_hidden_noise = tf.Variable(numpy.zeros((num_inputs, num_units), dtype=numpy.float32),
                        name=hidden_weights_name+NOISE_PREV_SUFFIX, trainable=False)
        tf.add_to_collection(PREV_EPOCH_DATA_COLLECTION_KEY, prev_hidden_noise)

        cur_hidden_noise = tf.Variable(numpy.zeros((num_inputs, num_units), dtype=numpy.float32),
                        name=hidden_weights_name+NOISE_CUR_SUFFIX, trainable=False)
        tf.add_to_collection(PREV_EPOCH_DATA_COLLECTION_KEY, cur_hidden_noise)

        reuse_noise_coeff = tf.Variable(0., dtype=tf.float32, name=hidden_weights_name+REUSE_NOISE_COEFF, trainable=False)
        new_noise_coeff = tf.Variable(1., dtype=tf.float32, name=hidden_weights_name+NEW_NOISE_COEFF, trainable=False)

        for batch_id in xrange(num_batchs):
            suffix=VAR_BATCH_SUFFIX+str(batch_id)
            hidden_weights_on_batch = tf.Variable(numpy.ones((num_inputs, num_units), dtype=numpy.float32),
                                name=hidden_weights_name+suffix, trainable=False)
            tf.add_to_collection(BATCH_COLLECTION_KEY, hidden_weights_on_batch)

            suffix=NOISE_BATCH_SUFFIX+str(batch_id)
            hidden_noise_on_batch = tf.Variable(numpy.zeros((num_inputs, num_units), dtype=numpy.float32), 
                                name=hidden_weights_name+suffix, trainable=False)
            tf.add_to_collection(BATCH_COLLECTION_KEY, hidden_noise_on_batch)

def VaryRate(start, end, saturate_epochs, epoch):
  """Compute a linearly varying number.

  Decrease linearly from start to end until epoch saturate_epochs.

  Args:
    start: the initial number.
    end: the end number.
    saturate_epochs: after this we do not reduce the number; if less than
      or equal to zero, just return start.
    epoch: the current learning epoch.
  Returns:
    the caculated number.
  """
  if saturate_epochs <= 0:
    return start

  step = (start - end) / (saturate_epochs - 1)
  if epoch < saturate_epochs:
    return start - step * epoch
  else:
    return end


def BatchClipByL2norm(t, upper_bound, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t

def MyBatchClipByL2norm(t, upper_bound, lower_bound=0, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

#   assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    new_upper_bound = tf.maximum(lower_bound, upper_bound)
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.divide(1.0, new_upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.multiply(tf.minimum(l2norm_inv, upper_bound_inv), new_upper_bound)
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t, new_upper_bound

def SoftThreshold(t, threshold_ratio, name=None):
  """Soft-threshold a tensor by the mean value.

  Softthreshold each dimension-0 vector (for matrix it is each column) by
  the mean of absolute value multiplied by the threshold_ratio factor. Here
  we soft threshold each column as it corresponds to each unit in a layer.

  Args:
    t: the input tensor.
    threshold_ratio: the threshold ratio.
    name: the optional name for the returned tensor.
  Returns:
    the thresholded tensor, where each entry is soft-thresholded by
    threshold_ratio times the mean of the aboslute value of each column.
  """

  assert threshold_ratio >= 0
  with tf.name_scope(values=[t, threshold_ratio], name=name,
                     default_name="soft_thresholding") as name:
    saved_shape = tf.shape(t)
    t2 = tf.reshape(t, tf.concat(axis=0, values=[tf.slice(saved_shape, [0], [1]), -1]))
    t_abs = tf.abs(t2)
    t_x = tf.sign(t2) * tf.nn.relu(t_abs -
                                   (tf.reduce_mean(t_abs, [0],
                                                   keep_dims=True) *
                                    threshold_ratio))
    return tf.reshape(t_x, saved_shape, name=name)


def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """
  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t

def AddMyGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """
  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    diff = tf.random_normal(tf.shape(t), stddev=sigma)
    noisy_t = t + diff
  return noisy_t, diff

# def MyAddGaussianNoise(t, sigma, d, name=None):
#     """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

#     Args:
#         t: the input tensor.
#         sigma: the stddev of the Gaussian noise.
#         name: optional name.
#     Returns:
#         the noisy tensor.
#     """
#     with tf.name_scope(values=[t, sigma], name=name, default_name="add_gaussian_noise") as name:
#         # print(tf.random_normal(tf.shape(t), stddev=sigma))
#         noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
#     return noisy_t

def GenerateBinomialTable(m):
  """Generate binomial table.

  Args:
    m: the size of the table.
  Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
  """

  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)


def average_shuffle_batch(batch_id_list, batch_size, train_images, train_labels):
    size_per_pick = int(batch_size/len(batch_id_list))
    for bat_id in batch_id_list:
        max_idx=batch_size
        for bid in batch_id_list:
            idx = numpy.arange(0, max_idx)
            random.shuffle(idx)
            idx = idx[0:size_per_pick]
            train_images[bid] = numpy.vstack([train_images[bid], train_images[bat_id][idx]])
            train_labels[bid] = numpy.vstack([train_labels[bid], train_labels[bat_id][idx]])
            
            train_images[bat_id] = numpy.delete(train_images[bat_id], idx, axis=0)
            train_labels[bat_id] = numpy.delete(train_labels[bat_id], idx, axis=0)
            max_idx-=size_per_pick

    return train_images, train_labels


def average_vars_grads(sess, total_batch):
    update_avg_noise = []
    for var in tf.trainable_variables():
        var_name = GetTensorOpName(var)
        num_inputs, num_units = var.shape[0].value, var.shape[1].value
        avg_noise=None
        for num_batch in xrange(total_batch):
            batch_noise_name = var_name + NOISE_BATCH_SUFFIX + str(num_batch)
            batch_noise = tf.get_collection(BATCH_COLLECTION_KEY, scope=batch_noise_name)[0]

            if num_batch==0:
                avg_noise=batch_noise
            else:
                avg_noise=tf.add(avg_noise, batch_noise)

        pre_noise = tf.get_collection(PREV_EPOCH_DATA_COLLECTION_KEY, scope=var_name+NOISE_PREV_SUFFIX)[0]
        update_avg_noise.append(tf.assign(pre_noise, tf.divide(avg_noise, total_batch)))

    _ = sess.run([update_avg_noise])


def assign_prev_vars_grads_for_cur_batch(sess, num_batch):
    # obtain last-round grads and weights for this batch
    assign_prev_vars=[]
    assign_prev_noises=[]
    for var in tf.trainable_variables():
        var_name = GetTensorOpName(var)

        str_batch_var = var_name + VAR_BATCH_SUFFIX + str(num_batch)
        batch_var = tf.get_collection(BATCH_COLLECTION_KEY, scope=str_batch_var)[0]

        pre_var = tf.get_collection(PREV_EPOCH_DATA_COLLECTION_KEY, scope=var_name+VAR_PREV_SUFFIX)[0]
        assign_prev_vars.append(tf.assign(pre_var, batch_var))

        str_batch_noise = var_name + NOISE_BATCH_SUFFIX + str(num_batch)
        batch_noise = tf.get_collection(BATCH_COLLECTION_KEY, scope=str_batch_noise)[0]

        pre_noise = tf.get_collection(PREV_EPOCH_DATA_COLLECTION_KEY, scope=var_name+NOISE_PREV_SUFFIX)[0]
        assign_prev_noises.append(tf.assign(pre_noise, batch_noise))
    _ = sess.run([assign_prev_vars, assign_prev_noises])

def update_prev_vars_grads_for_cur_batch(sess, epoch, num_batch):
    # update last-round grads and weights for this batch on each layer
    update_pre_vars = []
    update_pre_noise = []
    for var in tf.trainable_variables():
        var_name = GetTensorOpName(var)
        num_inputs, num_units = var.shape[0].value, var.shape[1].value

        # pre_tensor_name = var_name + VAR_BATCH_SUFFIX + str(num_batch)
        # pre_v = tf.get_collection(BATCH_COLLECTION_KEY, scope=pre_tensor_name)[0]

        # update_pre_vars.append(tf.assign(pre_v, var))

        batch_pre_noise_name = var_name + NOISE_BATCH_SUFFIX + str(num_batch)
        batch_pre_noise = tf.get_collection(BATCH_COLLECTION_KEY, scope=batch_pre_noise_name)[0]

        cur_noise = tf.get_collection(PREV_EPOCH_DATA_COLLECTION_KEY, scope=var_name+NOISE_CUR_SUFFIX)[0]
        update_pre_noise.append(tf.assign(batch_pre_noise, cur_noise))
    _ = sess.run([update_pre_vars, update_pre_noise])

def update_d_coefficient(sess, epoch, epochs, c):
    if epoch==0:
        return [0, 0], [1, 1]

    var_list = tf.trainable_variables()
    reuse_noise_list = []
    new_noise_list = []
    for var in var_list:
        tensor_name = GetTensorOpName(var)

        if epoch==0:
            d=1
            dd = 2.0/math.sqrt(epochs*1.0)
        else:
            d = 0.1
            # aa = 1.0/math.pow(2.0/math.sqrt(epochs*1.0), 2)
            # a = math.sqrt((epochs-1)/(epochs-aa))
            a=1
            dd = a*(1-c*math.pow((1-d),2))

        reuse_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+REUSE_NOISE_COEFF)[0]
        new_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+NEW_NOISE_COEFF)[0]
        
        reuse_noise_list.append(tf.assign(reuse_coeff, c*(1-d)))
        new_noise_list.append(tf.assign(new_coeff, dd))

    reuse_list = sess.run(reuse_noise_list)
    new_list = sess.run(new_noise_list)

    return reuse_list, new_list

    # if epoch==0 or epoch>switch_epoch:
    #     return [0, 0], [1, 1]

    # var_list = tf.trainable_variables()
    # reuse_noise_list = []
    # new_noise_list = []
    # if epoch==switch_epoch:
    #     i=0
    #     for var in var_list:
    #         tensor_name = GetTensorOpName(var)
    #         reuse_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+REUSE_NOISE_COEFF)[0]
    #         new_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+NEW_NOISE_COEFF)[0]
    #         reuse_noise_list.append(tf.assign(reuse_coeff, 0))
    #         dd = math.sqrt((epochs-epoch)*1.0/(epochs-1-accum_dd[i]/100.0))
    #         new_noise_list.append(tf.assign(new_coeff, dd))
    #         i+=1
    # else:
    #     for var in var_list:
    #         tensor_name = GetTensorOpName(var)
    #         pre_tensor_name = tensor_name + VAR_PREV_SUFFIX
    #         pre_var = tf.get_collection(PREV_EPOCH_DATA_COLLECTION_KEY, scope=pre_tensor_name)[0]

    #         d_tensor = tf.abs(var-pre_var)
    #         d = tf.minimum(tf.reduce_max(d_tensor) , 1.0)
    #         dd = tf.sqrt(tf.subtract(1.0, tf.pow(tf.multiply(c, 1.0-d), 2.0))) #tf.multiply(d, c*2.0)-c*tf.pow(d, 2) + (1-c)

    #         reuse_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+REUSE_NOISE_COEFF)[0]
    #         new_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tensor_name+NEW_NOISE_COEFF)[0]
            
    #         reuse_noise_list.append(tf.assign(reuse_coeff, c*(1-d)))
    #         new_noise_list.append(tf.assign(new_coeff, dd))
    
    # reuse_list = sess.run(reuse_noise_list)
    # new_list = sess.run(new_noise_list)
    # if epoch<switch_epoch and epoch>=1:
    #     for i in range(len(new_list)):
    #         accum_dd[i]+=math.pow((1-c*math.pow(reuse_list[i]/c, 2))/new_list[i], 2)

    # return reuse_list, new_list
        

    # num_layer=0
    # dc_list = []
    # var_list = tf.trainable_variables()
    # for var in var_list:
    #     var_name = GetTensorOpName(var)
    #     d_coeff = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_name+D_COEFF)[0]

    #     # if (epoch+1)%NUM_EPOCH_RESET==1:
    #     #     sess.run(tf.assign(d_coeff, 1.0))
    #     #     continue

    #     # coefficient for 1st layer is max value of next layer's weights 
    #     if num_layer==0:
    #         var = var_list[num_layer+1]
    #         dc = sess.run(tf.assign(d_coeff, tf.reduce_max(var)))
    #         dc_list.append(dc)
    #     # coefficients for following layers are related with 
    #     # max row vector l2 norms of previous layers' weights
    #     else:
    #         var = var_list[num_layer-1]
    #         c=1.0
    #         if num_layer>1:
    #             c=dc_list[num_layer-1]
    #         if num_layer==len(var_list)-1:
    #             c=c*1.0/8
    #         max_row_l2_norm = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.pow(var, 2), 1)))
    #         dc = sess.run(tf.assign(d_coeff, tf.multiply(max_row_l2_norm, c)))
    #         dc_list.append(dc)
    #     num_layer = num_layer+1
    # print(dc_list)