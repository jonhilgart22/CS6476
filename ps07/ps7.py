"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods.
    Refer to the method run_particle_filter( ) in experiment.py to understand how this
    class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.
        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles. This should be a
                                        N x 2 array where N = self.num_particles. This component
                                        is used by the autograder so make sure you define it
                                        appropriately. Each particle should be added using the
                                        convention [row, col]
        - self.weights (numpy.array): Array of N weights, one for each particle.
                                      Hint: initialize them with a uniform normalized distribution
                                      (equal weight for each one). Required by the autograder
        - self.template (numpy.array): Cropped section of the first video frame that will be used
                                       as the template to track.
        - self.frame (numpy.array): Current video frame from cv2.VideoCapture().

        Parameters
        ----------
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255]
            template (numpy.array): color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles (int): number of particles.
            - sigma_mse (float): sigma value used in the similarity measure.
            - sigma_dyn (float): sigma value that can be used when adding gaussian noise to u and v.
            - template_rect (dict): Template coordinates with x, y, width, and height values.
        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_mse = kwargs.get('sigma_mse') / 2.  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - extract any additional keyword arguments you need.

        self.template = template
        self.frame = frame
        height, width = self.frame.shape[:2]
        self.particles = np.array([self._generate_random_particle(width, height) for _ in xrange(self.num_particles)])  # Todo: Initialize your particles array. Read docstring.
        self.weights = np.ones(self.num_particles) / self.num_particles  # Todo: Initialize your weights array. Read docstring.
        # Initialize any other components you may need when designing your filter.
        self.count = 0

    @staticmethod
    def _generate_random_particle(width, height):
        """
        Method to get a random particle pair
        :param row:
        :param col:
        :return:
        """
        return [np.random.choice(height), np.random.choice(width)]

    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.weights

    def _sample(self):
        """
        Method to sample from the weighted distribution
        :return:
        """
        new_particles = np.zeros((0, 2))
        new_weights = np.array([])
        # get the number of sample needed for each index
        distribution = np.random.multinomial(self.num_particles, self.weights, size=1)

        for i, n in enumerate(distribution.reshape(self.num_particles)):
            new_particles = np.append(new_particles, np.full((n, 2), self.particles[i]), axis=0)
            new_weights = np.append(new_weights, np.full((n,), self.weights[i]), axis=0)

        self.particles = new_particles
        self.weights = new_weights / np.sum(new_weights)

    @staticmethod
    def _get_box(center, template):
        cy, cx = center
        h, w = template.shape[:2]
        x = int(cx - (w / 2))
        y = int(cy - (h / 2))
        return x, y, w, h

    @staticmethod
    def _calculate_mse(template, image):
        mse = np.sum(np.abs(template.astype(np.float) - image.astype(np.float)))
        mse /= np.float(template.shape[0] * template.shape[1])
        return mse

    def _calculate_similarity(self, mse):
        similarity = np.exp(-mse / (2. * np.square(self.sigma_mse)))
        return similarity

    def _get_noise(self):
        du = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)
        dv = np.random.randint(-self.sigma_dyn, self.sigma_dyn + 1)
        return np.array([du, dv])

    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        # sample from weighted distribution
        self._sample()

        normalization = 0.
        # for each particle
        for i, point in enumerate(self.particles):
            x, y, w, h = self._get_box(point, self.template)
            image = frame[y:y+h, x:x+w]

            # TODO: should we pad the box?
            if self.template.shape != image.shape:
                continue

            # calculate the similarity
            mse = self._calculate_mse(self.template, image)
            p_zx = self._calculate_similarity(mse)

            # update the weights
            self.weights[i] += p_zx
            # keep track of normalization
            normalization += self.weights[i]

            # add noise
            # TODO: should we mod by shape?
            self.particles[i] = self.particles[i] + self._get_noise()

        if normalization > 0:
            self.weights /= normalization
            self.weights /= np.sum(self.weights)

    def render(self, frame_in):
        """Visualize current particle filter state.
        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        Parameters
        ----------
            frame_in: copy of frame to overlay visualization on
        """

        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]

            # draw colored dot point on the image
            v, u = self.particles[i]
            # frame_in[v, u] = [0, 255, 0]
            cv2.circle(frame_in, (int(u), int(v)), 1, (0, 255, 0))

        # Complete the rest of the code as instructed.

        # based on the docstring, the center is simply the weighted mean of the (u, v)
        cy, cx = (u_weighted_mean, v_weighted_mean)
        # get the box size
        x, y, w, h = self._get_box((cy, cx), self.template)
        # draw the rectangle
        cv2.rectangle(frame_in, (x, y), (x + w, y + h), (255, 0, 0))

        # get euclidean distance of every particle to the weighted mean
        dist = self.particles - [u_weighted_mean, v_weighted_mean]
        dist = np.sqrt(np.sum(dist**2, axis=1))
        # calculate the average radius
        radius = np.sum(dist) / self.num_particles
        # draw the circle
        cv2.circle(frame_in, (int(cx), int(cy)), int(radius), (0, 0, 255))

        pass  # TODO: Your code here - draw particles, tracking window and a circle to indicate spread


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above. There is one
        element that is added called alpha which is explained in the problem set documentation.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - extract any additional keyword arguments you need.

    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        # sample from weighted distribution
        self._sample()

        best = np.array([])
        high_similarity = 0.
        normalization = 0.
        # for each particle
        for i, point in enumerate(self.particles):
            x, y, w, h = self._get_box(point, self.template)
            image = frame[y:y+h, x:x+w]

            # TODO: should we pad the box?
            if self.template.shape != image.shape:
                continue

            # calculate the similarity
            mse = self._calculate_mse(self.template, image)
            p_zx = self._calculate_similarity(mse)

            # get the best image with higher similarity
            if p_zx > high_similarity:
                best = image
                high_similarity = p_zx

            # update the weights
            self.weights[i] += p_zx
            # keep track of normalization
            normalization += self.weights[i]

            # add noise
            # TODO: should we mod by shape?
            self.particles[i] = self.particles[i] + self._get_noise()

        # update the template
        self.template = self.alpha * best + (1 - self.alpha) * self.template

        if normalization > 0:
            self.weights /= normalization
            self.weights /= np.sum(self.weights)

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MeanShiftLitePF(ParticleFilter):
    """A variation of particle filter tracker that uses the color distribution of the patch."""

    def __init__(self, frame, template, **kwargs):
        """Initialize Mean Shift Lite particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above. There is one
        element that is added called num_bins which is explained in the problem set documentation.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MeanShiftLitePF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        pass

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initialize MD particle filter object (parameters same as ParticleFilter).
        The documentation for this class is the same as the ParticleFilter above.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        pass

    # TODO: Override render() if desired (shouldn't have to, ideally)
