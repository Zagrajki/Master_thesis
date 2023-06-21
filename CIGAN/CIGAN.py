import tensorflow as tf
from tensorflow import keras
import numpy as np
from joblib import Parallel, delayed
import copy

class CIGAN():
    def __init__(self,
                 X_train,
                 y_train,
                 n_samples,
                 minor_classes='all',
                 coding_size='auto',
                 batch_size=32,
                 max_iter=10,
                 generator_hidden_layer_sizes=[100, 200, 300, 400, 500],
                 discriminator_hidden_layer_sizes=[500, 400, 300, 200, 100],
                 generator_hidden_layer_activation='selu',
                 discriminator_hidden_layer_activation='selu',
                 generator_optimizer=keras.optimizers.Adam,
                 discriminator_optimizer=keras.optimizers.Adam,
                 generator_learning_rate=10 ** -4,
                 discriminator_learning_rate=10 ** -4,
                 random_seed=42,
                 n_jobs=1):
        """
        Initialize the GAN class
        
        Parameters
        ----------
        X_train : The training feature matrix
        y_train : The training target vector
        minor_classes : the list of minority classes need to be oversampled, list of integers or 'all' (all the minority classes)
        coding_size : the dimension of the latent gaussian noise, an integer or 'auto' (half of the number of features)
        batch_size : the batch size for minibatch gradident descent
        max_iter: the maximum epoch for minibatch gradident descent
        generator_hidden_layer_sizes : the hidden layer sizes of the generator
        discriminator_hidden_layer_sizes : the hidden layer sizes of the discriminator
        generator_hidden_layer_activation: the hidden layer activation of the generator
        discriminator_hidden_layer_activation: the hidden layer activation of the discriminator
        generator_optimizer : the optimizer for the generator, Adam by default
        discriminator_optimizer : the optimizer for the discriminator, Adam by default
        generator_learning_rate : the learning rate for the generator, 10 ** -3 by default
        discriminator_learning_rate : the learning rate for the discriminator, 10 ** -3 by default
        random_seed : the random seed
        n_jobs : the number of CPU cores used when parallelizing over classes
        """
        self.n_samples = n_samples
        
        # Get the number of rows and columns in X_train
        self.m, self.n = X_train.shape
        
        # Get the classes and their number of samples
        self.classes, self.unique_counts = np.unique(y_train, return_counts=True)

        # Get the list of minority classes need to be oversampled, list of integers or 'all' (all the minority classes)
        self.minor_classes = [self.classes[i] for i in range(len(self.classes)) if self.unique_counts[i] < np.max(self.unique_counts)] if minor_classes == 'all' else minor_classes
        
        # Get the dimension of the latent gaussian noise, an integer or 'auto' (half of the number of features)
        self.coding_size = self.n // 2 if coding_size == 'auto' else coding_size
        
        # Get the batch size for minibatch gradident descent
        self.batch_size = batch_size
        
        # Get the maximum epoch for minibatch gradident descent
        self.max_iter = max_iter

        # Get the hidden layer sizes of the generator
        self.generator_hidden_layer_sizes = generator_hidden_layer_sizes

        # Get the hidden layer sizes of the discriminator
        self.discriminator_hidden_layer_sizes = discriminator_hidden_layer_sizes

        # Get the hidden layer activation of the generator
        self.generator_hidden_layer_activation = generator_hidden_layer_activation

        # Get the hidden layer activation of the discriminator
        self.discriminator_hidden_layer_activation = discriminator_hidden_layer_activation
        
        # Get the optimizer for the generator, Adam by default
        self.generator_optimizer = generator_optimizer
        
        # Get the optimizer for the discriminator, Adam by default
        self.discriminator_optimizer = discriminator_optimizer
        
        # Get the learning rate for the generator, 10 ** -3 by default
        self.generator_learning_rate = generator_learning_rate
        
        # Get the learning rate for the discriminator, 10 ** -3 by default
        self.discriminator_learning_rate = discriminator_learning_rate
        
        # Get the random seed
        self.random_seed = random_seed

        # The number of CPU cores used when parallelizing over classes 
        self.n_jobs = n_jobs
        
    def fit_resample(self, X_train, y_train):
        """
        Oversample the minority classses
        
        Parameters
        ----------
        X_train : The training feature matrix
        y_train : The training target vector
        
        Returns
        ----------
        The augmented training feature matrix and target vector
        """
        
        # Initialize the augmented training feature matrix
        self.X_gan_train = copy.deepcopy(X_train)
        
        # Initialize the augmented training target vector
        self.y_gan_train = copy.deepcopy(y_train)
        
        # Set backend="multiprocessing" (default) to prevent sharing memory between parent and threads
        Parallel(n_jobs=self.n_jobs)(delayed(self.oversample)(X_train, y_train, minor_class)
        for minor_class in self.minor_classes)   
        
        return [self.X_gan_train, self.y_gan_train]
    
    def oversample(self, X_train, y_train, minor_class):
        """
        Oversample a minority classs
        
        Parameters
        ----------
        X_train : The training feature matrix
        y_train : The training target vector
        minor_class : A minority class
        """
        
        # Build GAN
        self.build_gan()
        
        # Compile GAN
        self.compile_gan()
        
        # Train GAN
        self.train_gan(X_train, y_train, minor_class)
        
        # Augment the training data by adding samples generated for the minority class
        self.augment(minor_class)
        
    def build_gan(self):
        """
        Build GAN
        """
        
        # Build the generator
        self.build_generator()
        
        # Build the discriminator
        self.build_discriminator()
        
        # Build GAN
        self.gan = keras.models.Sequential([self.generator, self.discriminator])
        
    def build_generator(self):
        """
        Build the generator
        """
        
        # Initialize the generator
        self.generator = keras.models.Sequential()
        
        # For each hidden layer
        for i in range(len(self.generator_hidden_layer_sizes)):
            # Get the layer_size
            layer_size = self.generator_hidden_layer_sizes[i]
            
            # If it is the first hidden layer
            if i == 0:
                # Get the layer
                layer = keras.layers.Dense(layer_size,
                                           activation=self.generator_hidden_layer_activation,
                                           input_shape=[self.coding_size])
            # If it is not the first hidden layer            
            else:
                # Get the layer
                layer = keras.layers.Dense(layer_size,
                                           activation=self.generator_hidden_layer_activation)                
            # Add the layer to the generator
            self.generator.add(layer)
        
        # Add the output layer to the generator
        self.generator.add(keras.layers.Dense(self.n, activation='sigmoid'))
        
    def build_discriminator(self):
        """
        Build the discriminator
        """
        
        # Initialize the discriminator
        self.discriminator = keras.models.Sequential()

        # Add the first hidden layer to the discriminator
        self.discriminator.add(keras.layers.Dense(self.n))
        
        # For each hidden layer
        for i in range(len(self.discriminator_hidden_layer_sizes)):
            # Get the layer_size
            layer_size = self.discriminator_hidden_layer_sizes[i]
            
            # Get the layer
            layer = keras.layers.Dense(layer_size,
                                       activation=self.discriminator_hidden_layer_activation)                
            
            # Add the layer to the discriminator
            self.discriminator.add(layer)    
            
        # Add the output layer to the discriminator
        self.discriminator.add(keras.layers.Dense(1, activation='sigmoid'))
        
    def compile_gan(self):
        """
        Compile GAN
        """
        
        # Compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.discriminator_optimizer(learning_rate=self.discriminator_learning_rate))
        # Freeze the discriminator
        self.discriminator.trainable = False

        # Compile the generator
        self.gan.compile(loss='binary_crossentropy',
                         optimizer=self.generator_optimizer(learning_rate=self.generator_learning_rate))
        
    def train_gan(self, X_train, y_train, minor_class):
        """
        Train GAN
        
        Parameters
        ----------
        X_train : The training feature matrix
        y_train : The training target vector
        minor_class : A minority class
        """
        
        # Get the training feature matrix of the minority class
        X_minor_train = X_train[np.where(y_train == minor_class)]

        # Get the training target vector of the minority class
        y_minor_train = y_train[np.where(y_train == minor_class)]

        # Get the indices of the training data of the minority class
        idxs_minor_train = np.array(range(X_minor_train.shape[0]))

        # Get the number of minibatches
        n_batch = len(idxs_minor_train) // self.batch_size

        # For each epoch
        for _ in range(self.max_iter):
            # Shuffle the data
            np.random.RandomState(seed=self.random_seed).shuffle(idxs_minor_train)

            # For each minibatch
            for i in range(n_batch):
                # Get the first and last index (exclusive) of the minibatch
                first_idx = i * self.batch_size
                last_idx = min((i + 1) * self.batch_size, len(idxs_minor_train))

                # Get the minibatch
                mb = idxs_minor_train[first_idx : last_idx]

                # Get the real feature matrix
                real_features = X_minor_train[mb, :]

                # Get the noise
                noise = tf.random.normal(shape=[len(mb), self.coding_size], seed=self.random_seed)

                # Get the generated feature matrix
                gen_features = self.generator(noise)

                # Cominbe the generated and real feature matrix
                gen_real_features = tf.concat([gen_features, real_features], axis=0)

                # Get the target vector
                y = tf.constant([[0.]] * len(mb) + [[1.]] * len(mb))

                # Unfreeze the discriminator
                self.discriminator.trainable = True

                # Train the discriminator
                self.discriminator.train_on_batch(gen_real_features, y)

                # Get the noise
                noise = tf.random.normal(shape=[len(mb), self.coding_size], seed=self.random_seed)

                # Get the target
                y = tf.constant([[1.]] * len(mb))

                # Freeze the discriminator
                self.discriminator.trainable = False

                # Train the generator
                self.gan.train_on_batch(noise, y)

            # Save GAN
            self.gan.save('/content/CIGAN' + str(minor_class) + '/model.h5')
    
    def augment(self, minor_class):
        """
        Augment the training data by adding samples generated for the minority class
        
        Parameters
        ----------
        minor_class : A minority class
        """
        
        # Get the number of majority class
        n_major_class = np.max(self.unique_counts)
        
        # Get the number of minority class
        n_minor_class = self.unique_counts[self.classes[np.where(self.classes == minor_class)][0]]

        # Get the difference between the number of majority class and minority class
        #n_class_diff = n_major_class - n_minor_class
        n_class_diff = self.n_samples

        # Initialize the generated data
        gen_data = np.zeros((n_class_diff, self.n + 1))

        # For each sample
        for i in range(n_class_diff):
            # Get the noise
            noise = tf.random.normal(shape=[1, self.coding_size], seed=self.random_seed)

            # Get the generated features
            gen_features = self.generator(noise)

            # Update the generated data
            gen_data[i, :-1], gen_data[i, -1] = gen_features, minor_class
            
        # Augment the training feature matrix
        self.X_gan_train = np.vstack((self.X_gan_train, gen_data[:, :-1]))

        # Augment the training target vector
        self.y_gan_train = np.vstack((self.y_gan_train.reshape(-1, 1), gen_data[:, -1].reshape(-1, 1))).reshape(-1)      

