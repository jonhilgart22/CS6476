"""Problem Set 7: Particle Filter Tracking"""

from ps7 import *

# I/O directories
input_dir = "input"
output_dir = "output"


# Driver/helper code
def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """

    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    print template_rect

    # Loop over video (until last frame or Ctrl + C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                x = template_rect['x']
                y = template_rect['y']
                w = template_rect['w']
                h = template_rect['h']
                center = (x + w/2, y + h/2)
                print center
                print np.average(pf.particles, axis=0)
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)

            if False:  # For debugging, it displays every frame
                out_frame = frame.copy()
                pf.render(out_frame)
                cv2.imshow('Tracking', out_frame)
                cv2.waitKey(1)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break


def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    num_particles = 300  # Define the number of particles
    sigma_mse = 10  # Define a value for sigma when calculating the MSE
    sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    # TODO: Implement ParticleFilter
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  # suggested template window (dict)
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            94: os.path.join(output_dir, 'ps7-1-a-3.png'),
            171: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

    # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)
    # RESULT: converge faster?
    num_particles = 200  # Define the number of particles
    sigma_mse = 10  # Define a value for sigma when calculating the MSE
    sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    # TODO: Implement ParticleFilter
    template_rect = {'x': 340.8751, 'y': 195.1776, 'w': 63.5404, 'h': 89.0504}  # suggested template window (dict)
    # template_rect = {'x': 275.8751, 'y': 125.1776, 'w': 193.5404, 'h': 229.0504}  # suggested template window (dict)
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-1-a1-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a1-2.png'),
            94: os.path.join(output_dir, 'ps7-1-a1-3.png'),
            171: os.path.join(output_dir, 'ps7-1-a1-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

    # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
    # RESULT:
    # low sigma_mse: converge faster, because similarity (weight) becomes higher
    # high sigma_mse: converge slower, because similarity (weight) becomes lower
    num_particles = 300  # Define the number of particles
    sigma_mse = 4  # Define a value for sigma when calculating the MSE
    sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    # TODO: Implement ParticleFilter
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  # suggested template window (dict)
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-1-a2-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a2-2.png'),
            94: os.path.join(output_dir, 'ps7-1-a2-3.png'),
            171: os.path.join(output_dir, 'ps7-1-a2-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

    # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)
    # RESULT:
    # low num_particles: might converge to local optima i.e. wrong patch
    # high num_particles: slower to run
    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define a value for sigma when calculating the MSE
    sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    # TODO: Implement ParticleFilter
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  # suggested template window (dict)
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-1-a3-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a3-2.png'),
            94: os.path.join(output_dir, 'ps7-1-a3-3.png'),
            171: os.path.join(output_dir, 'ps7-1-a3-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

    # 1b
    # You may define new values for num_particles, sigma_mse, and sigma_dyn
    num_particles = 300  # Define the number of particles
    sigma_mse = 10  # Define a value for sigma when calculating the MSE
    sigma_dyn = 10  # Define a value for sigma when adding noise to the particles movement
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}
    run_particle_filter(ParticleFilter,
        os.path.join(input_dir, "noisy_debate.mp4"),
        template_rect,
        {
            14: os.path.join(output_dir, 'ps7-1-b-1.png'),
            94: os.path.join(output_dir, 'ps7-1-b-2.png'),
            530: os.path.join(output_dir, 'ps7-1-b-3.png')
        },
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

    # 2a
    # You may define new values for num_particles, sigma_mse, and sigma_dyn
    num_particles = 3000  # Define the number of particles
    sigma_mse = 6  # Define a value for sigma when calculating the MSE
    sigma_dyn = 30  # Define a value for sigma when adding noise to the particles movement
    alpha = .37  # Define a value for alpha
    # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
    # TODO: Run it on pres_debate.mp4 to track Romney's left hand, tweak parameters to track up to frame 140
    # template_rect = {'x': 520, 'y': 375, 'w': 85, 'h': 130}  # TODO: Define the hand coordinate values
    # template_rect = {'x': 545, 'y': 385, 'w': 60, 'h': 95}  # TODO: Define the hand coordinate values
    template_rect = {'x': 545, 'y': 390, 'w': 60, 'h': 90}  # TODO: Define the hand coordinate values
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
        },  # frames to save, mapped to filenames, and 'template'
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
        template_coords=template_rect)  # Add more if you need to

    # 2b
    # You may define new values for num_particles, sigma_mse, sigma_dyn, and alpha
    num_particles = 4000  # Define the number of particles
    sigma_mse = 6  # Define a value for sigma when calculating the MSE
    sigma_dyn = 30  # Define a value for sigma when adding noise to the particles movement
    alpha = .37  # Define a value for alpha
    # TODO: Run AppearanceModelPF on noisy_debate.mp4, tweak parameters to track hand up to frame 140
    template_rect = {'x': 545, 'y': 390, 'w': 60, 'h': 90}  # Define the template window values
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "noisy_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
            15: os.path.join(output_dir, 'ps7-2-b-2.png'),
            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
            140: os.path.join(output_dir, 'ps7-2-b-4.png')
        },  # frames to save, mapped to filenames, and 'template'
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn, alpha=alpha,
        template_coords=template_rect)  # Add more if you need to

    # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)

    # 3a
    # You may define new values for num_particles, sigma_mse, and sigma_dyn
    num_particles = 500  # Define the number of particles
    sigma_mse = .5  # Define a value for sigma when calculating the MSE
    sigma_dyn = 8  # Define a value for sigma when adding noise to the particles movement
    hist_bins_num = 8
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}
    run_particle_filter(MeanShiftLitePF,
        os.path.join(input_dir, "pres_debate.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
            94: os.path.join(output_dir, 'ps7-3-a-3.png'),
            171: os.path.join(output_dir, 'ps7-3-a-4.png')
        },
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        hist_bins_num=hist_bins_num,
        template_coords=template_rect)  # Add more if you need to

    # 3b
    # You may define new values for num_particles, sigma_mse, sigma_dyn, and hist_bins_num
    num_particles = 400  # Define the number of particles
    sigma_mse = .5  # Define a value for sigma when calculating the MSE
    sigma_dyn = 8  # Define a value for sigma when adding noise to the particles movement
    hist_bins_num = 16
    template_rect = {'x': 540, 'y': 390, 'w': 60, 'h': 100}  # Define the template window values
    run_particle_filter(MeanShiftLitePF,
        os.path.join(input_dir, "pres_debate.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
            15: os.path.join(output_dir, 'ps7-3-b-2.png'),
            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
            140: os.path.join(output_dir, 'ps7-3-b-4.png')
        },
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        hist_bins_num=hist_bins_num,
        template_coords=template_rect)  # Add more if you need to

    # 4: Discussion problems. See problem set document.

    # # 5: Implement a more sophisticated model to deal with occlusions and size/perspective changes
    num_particles = 100  # Define the number of particles
    sigma_mse = 8  # Define a value for sigma when calculating the MSE
    sigma_dyn = 3  # Define a value for sigma when adding noise to the particles movement
    template_rect = {'x': 235, 'y': 60, 'w': 55, 'h': 220}  # Define the template window values
    run_particle_filter(MDParticleFilter,
        os.path.join(input_dir, "pedestrians.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-5-a-1.png'),
            40: os.path.join(output_dir, 'ps7-5-a-2.png'),
            100: os.path.join(output_dir, 'ps7-5-a-3.png'),
            240: os.path.join(output_dir, 'ps7-5-a-4.png')
        },
        num_particles=num_particles, sigma_mse=sigma_mse, sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to

if __name__ == '__main__':
    main()
