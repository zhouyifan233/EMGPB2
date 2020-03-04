A python implementation of paper "Switching Kalman filters". This method is also called Expectation Maximisation Generalised Pseudo Bayes 2 (EMGPB-2).

Estimate parameters of a linear dynamical system from multiple iid discrete-time observation sequences. 
 
Under Construction...

Usage
----------------------
"test_emgpb2_estimation": Simulate a path using given parameters (two models). Estimate the parameters using EMGPB2. Includes two scenarios constant velocity and random walk.  

"create_test_data": create a track for testing. The track will be stored in folder "data/".

"EMGPB2 example CV.ipynb" (Jupyter notebook): An example to show improvement of using EMGPB2 considering Constant Velocity model (CV) compared to unknown (guessed) parameters. 
This example is supposed to run with Stone-soup which provides an IMM implementation.  

"EMGPB2 example RW.ipynb" (Jupyter notebook): An example to show improvement of using EMGPB2 considering Random Walk model (RW) compared to unknown (guessed) parameters. 
This example is supposed to run with Stone-soup which provides an IMM implementation.  


Dependencies
----------------------
Python (v3.7)  
Numpy  
Scipy  
Stone-soup (https://github.com/dstl/Stone-Soup)

References
----------------------
[1] Ghahramani, Zoubin, and Geoffrey E. Hinton. Parameter estimation for linear dynamical systems. Technical Report CRG-TR-96-2, University of Totronto, Dept. of Computer Science, 1996.

[2] Murphy, Kevin P. "Switching kalman filters." (1998): 21.


Acknowledgement
----------------------
This implementation is base on Josh Coates' work.  


