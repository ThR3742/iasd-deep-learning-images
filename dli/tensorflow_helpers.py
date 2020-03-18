import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


def normalize(train_images_, test_images_):
    # convert from integers to floats
    train_norm = train_images_.astype("float32")
    test_norm = test_images_.astype("float32")
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def conv_block(
    filter_size_,
    kernel_size_,
    max_pooling_size_,
    activation_type_=None,
    input_shape_=None,
    seed=42,
):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model_.seed = seed
    model_.add(
        tf.keras.layers.Conv2D(
            filters=filter_size_,
            kernel_size=kernel_size_,
            activation=activation_type_,
            input_shape=input_shape_,
            padding="same",
        )
    )
    model_.add(
        tf.keras.layers.Conv2D(
            filters=filter_size_,
            kernel_size=kernel_size_,
            activation=activation_type_,
            padding="same",
        )
    )
    model_.add(tf.keras.layers.MaxPooling2D(max_pooling_size_))
    return model_


def accuracy_lost_curves(history):
    # plot loss
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(history.history["loss"], color="blue", label="train")
    ax1.plot(history.history["val_loss"], color="orange", label="test")
    ax1.set_title("Cross Entropy Loss")
    ax1.legend(["Training Loss", "Validation Loss"])
    # plot accuracy
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(history.history["accuracy"], color="blue", label="train")
    ax2.plot(history.history["val_accuracy"], color="orange", label="test")
    ax2.set_title("Classification Accuracy")
    ax2.legend(["Training Acc", "Validation Acc"])


def display_images(
    images_, labels_, rows_, cols_, additional_input_=""
):  # sample here is an image from a dataset tfds
    fig = plt.figure(figsize=(8, 8))
    for img in range(rows_ * cols_):
        ax = fig.add_subplot(rows_, cols_, 1 + img)
        ax.imshow(images_[img])
        ax.set_axis_off()
        if additional_input_ != "":
            ax.set_title(str(labels_[img]) + " vs: " + str(additional_input_[img]))
        else:
            ax.set_title(str(labels_[img]))


def resize_image(image_, sizes=(32, 32)):
    image = tf.image.resize_with_crop_or_pad(image_, sizes[0], sizes[1])
    return image


def get_images_and_labels(datasetv1Adapter):
    import time

    start_time = time.time()
    dataset = tfds.as_numpy(dataset)
    dataset = np.array(list(dataset))

    images = np.array([example["image"] for example in dataset])
    labels = np.array([example["label"] for example in dataset])
    print("--- %s seconds ---" % (time.time() - start_time))

    # If you're using datasetv1Adapter as input
    # images = np.array([example['image'] for example in datasetv1Adapter])
    # labels =  np.array([np.array(example['label']) for example in datasetv1Adapter])
    return images, labels
