# About Project

Generating unique stimulus videos for pigeons.

You can use either the `pygeon.py` file from the terminal *or* the jupyter notebook. I would recommend using the notebook for editing parameters and such, but if satisfied with existing parameters or expanding features, you might want to use the og-python file.

## Output

The program outputs 3 videos: 1 video will function as the fixation point screen (camera 3) and the other 2 will randomly switch between displaying a randomly selected shape. This code relies heavily on the skimage library for the generation and implementation.

**time key**: a video key is generated for each set of videos (contains shape, screen, & time info). See `output/18-03-2021_time_key.csv` for example.

## 2 Stimulus generators

`stimulus_generator.ipynb`: create videos with visual stimuli. Duration of time shown differs between fixation and stimuli.

`simple_stim_gen.ipynb`: creates stimuli and fixation points with the same durations;

- 10 second for all shapes displayed followed by 5 seconds for stimuli and 15 sec. for fixation pt.

## Installation

Currently, the project has only been implemented with macOS.

### Dependencies

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/)
2. Clone pygeon repo
3. From Anaconda Navigator select **import**
    - All required packages will be installed.
    - This may take a while, since this environment is set up to run other CV software.

`conda env create -f environment.yml`

## Essential packages

[Matplotlib](https://matplotlib.org/)
[Numpy](https://numpy.org/)
[OpenCV](https://opencv.org/)
[Scipy](https://www.scipy.org/)
[Skimage](https://scikit-image.org/)
[Pandas](https://pandas.pydata.org/)