
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from keras.preprocessing import image
import cv2

class DataGenerator:

    def __init__(self, method_read_data, config):
        self.method_read_data = method_read_data
        self.config = config
        self.batch_size = int(self.config['batch_size'])
        self.image_dimension_3D = tuple(self.config['image_dimension_3D'])
        self.data_folder_path = self.config['data_folder_path']  # folder of people
        self.train_file = self.config['train_file']
        self.test_file = self.config['test_file']
        self.validation_file = self.config['validation_file']
        self.txt_file_path = self.config['txt_file_path']  # original data file
        self.all_data_file = self.config['all_data_file']
        self.min_num_img = self.config['minumum_number_image']


    def load_data_train(self):
        """
        like python library which loads mnist
        this method loads data in format of generators
        """
        if self.method_read_data == 'create_data_file':
            self.create_imageFile_with_label()
            self.make_test_train_pairs()
            self.split_data()
            return self.read_data()
        if self.method_read_data == 'data_file_exists':
            return self.read_data()

    def read_data(self):
        """
        reads data from their files and create generators
        passes them to load data method
        :return: train data generator
        :return: validation data generator
        :return: length of train pairs
        :return: length of validation pairs
        """
        train_df = pd.read_csv(self.train_file)
        validation_df = pd.read_csv(self.validation_file)
        train_gen = self.generate_batches(train_df)
        validation_gen = self.generate_batches(validation_df)
        train_len = len(train_df)
        validation_len = len(validation_df)

        return train_gen, validation_gen, train_len, validation_len

    def load_data_test(self):
        """
        loads test data in format of generator
        :return: test data generator
        :return: test data in format of dataframe
        """
        test_df = pd.read_csv(self.test_file)
        test_gen = self.generate_batches_test(test_df)

        return test_gen, test_df


    def create_imageFile_with_label(self):
        """
        There are lots of folders in names of individuals
        a file should be created with person name and it's label
        to create pairs from that file later
        """
        counter = 0
        with open(self.txt_file_path, 'a') as the_file:
            for folder_name in glob.glob(self.data_folder_path + '/*'):
                for image in glob.glob(folder_name + '/*'):
                    the_file.write(str(image) + ',' + str(counter) + '\n')
                counter = counter + 1

    def make_test_train_pairs(self):
        """
        makes positive and negative pairs
        and stores them in csv file
        sample line of input file(.txt) : 000001.jpg 2880
        sample line of output file(.csv):
        195015,099221.jpg,086512.jpg,0
        395220,201001.jpg,197034.jpg,1
        """
        with open(self.txt_file_path, 'r', encoding="utf8") as f:
            lines = f.readlines()
        image_name = []
        image_class = []
        for i in range(0, len(lines)):
            image_name.append(lines[i].split(",")[0])
            image_class.append(int(lines[i].split(",")[1]))

        data = np.array(image_name)
        label = np.array(image_class)
        num_classes = len(np.unique(label))
        print(num_classes)

        data, label = self.delete_low_num_images(data, label, self.min_num_img)
        (pairData, labelData) = self.make_pairs(data, label, num_classes)
        datapair_df = pd.DataFrame(np.concatenate((pairData, labelData), axis=1))
        datapair_df.columns = ["image1", "image2", "label"]
        datapair_df.to_csv(self.all_data_file)

    def split_data(self):
        """
        splits pairs into test and train
        80% train, 20% test, and 10% of train for validation
        """
        datapair_df = pd.read_csv(self.all_data_file)
        X_train, X_test = train_test_split(datapair_df, test_size=0.2, shuffle=True)
        X_train, X_val = train_test_split(X_train, test_size=0.1, shuffle=True)

        X_train.to_csv(self.train_file)
        X_test.to_csv(self.test_file)
        X_val.to_csv(self.validation_file)

    def make_pairs(self, images, labels, num_classes):
        """
        initializes two empty lists to hold the (image, image) pairs and labels
        to indicate if a pair is positive or negative
        and then builds a list of indexes for each class label that
        provides the indexes for all examples with a given label
        :param images: numpy array of images name list
        :param labels: numpy array of labels list
        :param num_classes: length of labels
        :return: image pair and label
        """
        pairImages = []
        pairLabels = []
        numClasses = int(num_classes) + 1
        idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

        for idxA in range(len(images)):
            """
            loops over all images
            grabs the current image and label belonging to the current iteration
            randomly picks an image that belongs to the *same* class label
            prepares a positive pair and updates the images and labels lists respectively
            grabs the indices for each of the class labels *not* equal to the current label 
            and randomly picks an image corresponding to a label *not* equal to the current label
            returns a 2-tuple of our image pairs and labels
            """
            currentImage = images[idxA]
            label = labels[idxA]
            idxB = np.random.choice(idx[label])
            posImage = images[idxB]
            counter_pos = 0
            print([currentImage, posImage])
            while currentImage == posImage or [posImage, currentImage] in pairImages:
                print("counter pos", counter_pos)
                counter_pos = counter_pos + 1
                idxB = np.random.choice(idx[label])
                posImage = images[idxB]
                print("second ", [currentImage, posImage])
                if counter_pos > 10:
                    break
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])

            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]
            counter_neg = 0
            while [negImage, currentImage] in pairImages:
                counter_neg = counter_neg + 1
                negImage = images[np.random.choice(negIdx)]
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])

            """
            if activation function is sigmoid label should be 0 or 1
            if activation function is softmax label should be [0,1] or [1,0]
            """
        return (np.array(pairImages), np.array(pairLabels))

    def delete_low_num_images(self, data, label, num):
        """
        deletes some individuals with images less than num
        cause labels with few images are not enough for making distinct pairs
        for positive and negative images to not be the same we have to omit these
        :param data: numpy array of images name list
        :param label: numpy array of labels list
        :param num: minimum number of images in folder
        :return:  numpy array of images name list
        :return: numpy array of labels list
        """
        diction = collections.Counter(label)
        for i in diction:
            if diction[i] < num:
                index_image = np.where(label == i)
                label = np.delete(label, index_image[0])
                data = np.delete(data, index_image[0])

        return data, label

    def generate_batches(self, data_file):
        """
        train is so much that we have to make a generator to prevent memory loss
        so data is read every time in amount of batch size
        and yielded to model
        :param data_file: numpy array of image name list
        """
        counter = 0
        while True:
            counter = (counter) % len(data_file)
            X_train, y_train = self.get_batch_data(data_file, counter)
            counter = counter + self.batch_size
            yield (X_train, y_train)

    def generate_batches_test(self,data_file):
        """
        test generator is a little different with test
        cause in test model has to see data once
        """
        for cbatch in range(0, len(data_file), self.batch_size):
            X_train, y_train = self.get_batch_data(data_file, cbatch)
            yield (X_train, y_train)

    def get_batch_data(self, data_file, cbatch):
        """
        creates a batch of data and returns to generator
        a batch of data means pairs of images in form of numpy array like a list
        with their labels
        :param data_file: dataframe of image pairs
        :param cbatch: batch size
        :return: image pair numpy
        :return: label of pair images
        """
        pairImages = []
        pairLabels = []
        for i in range(cbatch, cbatch + self.batch_size):
            if i < len(data_file):
                image1_name = str(data_file.loc[i]['image1'])
                image1 = self.get_image_with_cv2(image1_name)
                image2_name = str(data_file.loc[i]['image2'])
                image2 = self.get_image_with_cv2(image2_name)
                pairImages.append([image1.reshape(self.image_dimension_3D), image2.reshape(self.image_dimension_3D)])
                label = int(data_file.loc[i]['label'])
                if label == 0:
                    pairLabels.append(0.0)
                    # pairLabels.append([1, 0]) # if activation func = softmax
                if label == 1:
                    pairLabels.append(1.0)
                    # pairLabels.append([0, 1])

        pairTrain = np.array(pairImages)
        pairLabel = np.array(pairLabels)

        return [pairTrain[:, 0], pairTrain[:, 1]], pairLabel[:]

    def get_image_data(self, image_path):
        """
        reads an image
        changes the format to numpy array
        and normalizes it
        :param image_path: image path
        :return: image numpy
        """

        img = image.load_img(image_path, target_size=self.image_dimension, interpolation='bilinear')
        img_data = image.img_to_array(img) / 255.
        img_data.reshape(self.image_dimension_3D)

        return img_data

    def get_image_with_cv2(self, image_path):
        """
        reads an image
        changes the format to numpy array
        and normalizes it
        :param image_path: image path
        :return: image numpy
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.config['image_dimension_2D']))
        img = img.reshape(tuple(self.config['image_dimension_4D']))
        img_data = img / 255.

        return img_data
