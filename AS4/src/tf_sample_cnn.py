'''
    Convolutional Neural Network for MNIST Dataset

    Reference Web Links:
    1. TensorFlow CNN Estimator tutorial: https://www.tensorflow.org/tutorials/estimators/cnn
'''

import tensorflow as tf
import numpy as np
import time


def compute_measures(labels, predictions):
    '''
    computes prediction measures
    :param labels:      actual class labels
    :param predictions: predicted class labels
    :return:            accuracy, recall, precision of classification
    '''
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    recall = tf.metrics.recall(labels=labels, predictions=predictions)
    precision = tf.metrics.precision(labels=labels, predictions=predictions)
    return accuracy, recall, precision


def convolution_layer(input, filters, winSize, initializer, name):
    '''
    Adds a single convolution layer to graph
    :param input:       input features (tf.float32)
    :param filters:     number of filters to learn
    :param winSize:     size of kernel (int)
    :param initializer: weight initialization method
    :param name:        name to assign tensor
    :return:            features convoluted with each filter
    '''
    return tf.layers.conv2d(
        inputs=input,
        filters=filters,
        kernel_size=[winSize, winSize],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=initializer['fast_conv'],
        name=name)


def convolute_and_pool(features, depth, initializer):
    '''
    Implements convolution and pooling for a deep network
    :param features:    Input features (images)
    :param depth:       Depth of the CNN
    :param initializer: weight initialization method
    :return:            features from all filters in last layer
    '''
    f_features = features
    for i in range(depth):
        layer_ID = i + 1
        filter_Num = 2**(5 + i)
        t_name = 'Conv_{}'.format(layer_ID)

        # Convolutional Layer
        # Computes (X = filter_Num of features using a 5x5 kernel size for each filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, W', H', previous_num_of_filters]
        # Output Tensor Shape: [batch_size, W', H', filter_Num]
        with tf.name_scope(t_name):
            f_features = convolution_layer(f_features, filter_Num, winSize=5,
                                           initializer=initializer, name=t_name.lower())

        # Pooling Layer
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, W', H', filter_Num]
        # Output Tensor Shape: [batch_size, W'/2, H'/2, filter_Num]
        with tf.name_scope('Pool_{}'.format(layer_ID)):
            f_features = tf.layers.max_pooling2d(inputs=f_features, pool_size=[2, 2], strides=2)

    return f_features


def feed_to_cnn(batch_input, depth, initializer, drop_rate, mode):
    '''
    Feeds batch inputs to logits
    :param batch_input: features (images) of a batch
    :param depth:       Number of convolutional layers
    :param initializer: weight initialization method/variable
    :param drop_rate:   percentage of nodes to mask in each sample
    :return:            logits tensor (vector of size 9)
    '''
    batch_output = convolute_and_pool(batch_input, depth, initializer)

    # Input Tensor Shape: [batch_size, W/16, H/16, 64]
    # Output Tensor Shape: [batch_size, 1024]
    with tf.name_scope('Dense_Layer'):
        # Flatten tensor into a batch of vectors
        flat = tf.reshape(batch_output, [-1, 7 * 7 * 64])
        # Dense mapping of vector to a fully connected layer with 1024 neurons
        dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

    # Dropout operation; randomly set a fraction(drop_rate) of the input's nodes to 0
    # Input Tensor Shape == Output Tensor Shape: [batch_size, 1024]
    with tf.name_scope('Dropout_Layer'):
        dropout = tf.layers.dropout(
            inputs=dense, rate=drop_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    with tf.name_scope('Logits_Layer'):
        batch_logits = tf.layers.dense(inputs=dropout, units=10)

    return batch_logits


def compute_loss(batch_logits, batch_labels):
    '''
    Computes softmax cross-entropy loss for batch for (for both TRAIN and EVAL modes)
    :param batch_logits:    logits tensot for batch
    :param batch_labels:    corresponding class labels
    :return:                cross-entropy loss op
    '''
    with tf.name_scope('Compute_Loss'):
        batch_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=batch_labels,
            logits=batch_logits
        )

    tf.summary.scalar('Cross_Entropy_Loss', batch_loss)
    tf.summary.histogram('Loss-Histogram', batch_loss)
    return batch_loss


def log_training_summaries(actual_labels, predicted_labels, params):
    '''
    operation for computing and saving training accuracy metric
    :param actual_labels:       known class of samples
    :param predicted_labels:    predicted class of samples
    :param params:              model parameters
    :return:                    save summaries hook
    '''
    with tf.name_scope('Training_Summaries'):
        accuracy = tf.contrib.metrics.accuracy(labels=actual_labels, predictions=predicted_labels)
        tf.summary.scalar('training_accuracy', accuracy)

        # Create a SummarySaverHook
        training_hooks = []
        train_summary_hook = tf.train.SummarySaverHook(
            save_steps=params['summary_steps'],
            output_dir=params['model_dir'],
            summary_op=tf.summary.merge_all()
        )

        # Add SummarySaverHook to the evaluation_hook list
        training_hooks.append(train_summary_hook)

    return training_hooks


def log_evaluation_summaries(actual_labels, predicted_labels, loss, mode, params):
    '''
    operation for computing and save some evaluation metrics
    :param actual_labels:       known class of samples
    :param predicted_labels:    predicted class of samples
    :param loss:                computed loss of predictions
    :param mode:                tf.mode
    :param params:              model parameters
    :return:                    evaluation operation estimator spec
    '''
    with tf.name_scope('Evaluation_Summaries'):
        accuracy, recall, precision = compute_measures(actual_labels, predicted_labels)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision
        }

        # Create a SummarySaverHook
        evaluation_hooks = []
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=params['summary_steps'],
            output_dir=params['model_dir'] + "/eval",
            summary_op=tf.summary.merge_all()
        )

        # Add SummarySaverHook to the evaluation_hook list
        evaluation_hooks.append(eval_summary_hook)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=evaluation_hooks
    )


def train_cnn(loss, mode, learn_rate, training_hooks):
    '''
    Configure the Training Op (for TRAIN mode)
    :param loss:            computed classification class
    :param mode:            tf.mode
    :param learn_rate:      optimizer learning rate
    :param training_hooks:  training hooks for summaries
    :return:                train operation estimator spec
    '''

    with tf.name_scope('Optimization'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)  # Important for distributed training
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=training_hooks)



def cnn_model_fn(features, labels, mode, params):
    '''
    Convolutional Neural Network model function
    :param features:    features(images) of a given batch
    :param labels:      corresponding labels of given batch
    :param mode:        tensorflow estimator mode
    :param params:      parameters passed to estimator
    :return:            operation for training, evaluation, or prediction
    '''

    # Input Layer: Reshape X to 4-D tensor: [batch_size, width, height, channels]
    with tf.name_scope('Input'):
        imgW, imgH, channels = params['img_size']
        input = tf.reshape(features["x"], [-1, imgW, imgH, channels])
        tf.summary.image('Input-Images', input, 10)

    logits = feed_to_cnn(input, params['depth'], params['w_inits'], params['drop_rate'], mode)
    tf.summary.histogram('Logits-Histogram', logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
        # Add `softmax_tensor` to the graph.
        # It is used for PREDICT and by the `logging_hook`.
        # softmax changes scores from logits to probabilities
        "probabilities": tf.nn.softmax(logits, name="logits_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = compute_loss(logits, labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_hooks = log_training_summaries(labels, predictions["classes"], params)
        return train_cnn(loss, mode, params['learn_rate'], train_hooks)

    return log_evaluation_summaries(labels, predictions["classes"], loss, mode, params)



def main():
    # Initializing model training variables
    imgW, imgH, channels = 28, 28, 1
    train_epochs = 20
    batch_size = 200
    drop_rate = 0.4
    learn_rate = 0.001
    cnn_layers = 2
    model_dir = './cnn_model_20epochs/'

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Approximations
    num_of_steps_per_epoch = len(train_data) / batch_size
    checkpointing_frequency = int(round(2 * num_of_steps_per_epoch, 0))
    save_summary_frequency = int(round(num_of_steps_per_epoch / 4, 0))
    logging_frequency = 100 # steps

    # Different types of popular weight initialization methods
    initializers = {
        'fast_conv': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True),
        'he_rec': tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
        'xavier': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
    }
    parameters = {'img_size': [imgH, imgW, channels], 'summary_steps': save_summary_frequency,
                  'model_dir': model_dir, 'learn_rate': learn_rate, 'drop_rate': drop_rate,
                  'w_inits': initializers, 'depth': cnn_layers}

    # setup directory structure
    print("\n---------------------------------------------" + \
          "\nTrain Set Image Shape:             " + str(train_data.shape) + \
          "\nTrain Set Label Shape:             " + str(train_labels.shape) + \
          "\nTest Set Image Shape:              " + str(eval_data.shape) + \
          "\nTest Set Label Shape:              " + str(eval_labels.shape) + \
          "\nApprox. number of Steps per Epoch: " + str(round(num_of_steps_per_epoch, 3)) + \
          "\nCheckpoint(save model) done every: " + str(checkpointing_frequency) + " steps" + \
          "\nSave summary done every:           " + str(save_summary_frequency) + " steps" + \
          "\n\nHyper Parameters" + \
          "\nNumber of CNN Layers:              " + str(cnn_layers) + \
          "\nBatch size used:                   " + str(batch_size) + \
          "\nTotal number of epochs trained:    " + str(train_epochs) + \
          "\nDropout Rate:                      " + str(drop_rate) + \
          "\nLearning Rate:                     " + str(learn_rate) + \
          "\n---------------------------------------------\n")

    # Setup Estimator Run Configuration
    configuration = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=None,
        save_summary_steps=save_summary_frequency,
        save_checkpoints_steps=checkpointing_frequency,
        save_checkpoints_secs=None,
        session_config=None,
        keep_checkpoint_max=4,
        keep_checkpoint_every_n_hours=10000,
        log_step_count_steps=logging_frequency
    )

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(cnn_model_fn), # Distributed Training on All GPUs:
        config=configuration,
        params=parameters
    )

    # Setup logging
    tf.logging.set_verbosity(tf.logging.INFO)
    # Setup logging for predictions
    # Prints to console the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"\n\nClass Probabilities:\n": "logits_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=logging_frequency)

    start_time = time.time()
    for epoch in range(train_epochs):
        # todo: reset random seed

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=2,
            shuffle=True)

        classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        eval_results = classifier.evaluate(input_fn=eval_input_fn, steps=None)

        print("\n\nEpoch: " + str(epoch + 1) + "\n" + str(eval_results))
    print("\n---------------------------------------------")
    print("\nRuntime: " + str(int((time.time() - start_time) / 60)) + " minutes")
    print("\n---------------------------------------------")



# Only train model if this script is run as main script (ie called directly)
if __name__ == "__main__":
    main()