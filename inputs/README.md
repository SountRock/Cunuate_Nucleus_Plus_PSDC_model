## About the datasets

The datasets were generated using TouchSim, a model of human mechanoreceptors.

Why synthetic? Because there is very little open data available for optimization, and TouchSim allows producing sufficiently natural input patterns, with a more realistic distribution of RA and SA mechanoreceptor responses.

The datasets were manually selected based on the naturalness of the input pattern generation - the main factor being smooth transitions in frequency dynamics. They were collected for 10 neurons, but can be scaled up to 100 neurons if good mixing is ensured.

__Touchsim GitHub:__ https://github.com/hsaal/touchsim?ysclid=mkb2ey0n9p100347384 

inputs\genPatterns.py - script for generating synthetic synchronous and asynchronous patterns