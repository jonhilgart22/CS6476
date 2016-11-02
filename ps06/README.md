#Problem Set 6: Optic Flow

##Description
We discussed optic flow as the problem of computing a dense flow field where a flow field is a vector field <u(x,y), v(x,y)>. We discussed a standard method — Hierarchical Lucas and Kanade — for computing this field. The basic idea is to compute a single translational vector over a window centered about a point that best predicts the change in intensities over that window by looking at the image gradients. For this problem set you will implement the necessary elements to compute the LK optic flow field. This will include the necessary functions to create a Gaussian pyramid.

## Setup
Clone this repository recursively:
`git clone --recursive https://github.gatech.edu/omscs6476/ps06.git`

(If your version of git does not support recurse clone, then clone without the option and tun `git submodule init` and `git submodule update`).

##Instructions
If you have questions, please post them to Piazza.
[Problem Set 6: Optic Flow](https://docs.google.com/document/d/1NOS8_2RoKTcVAG73BsqwRWLiKmge5KAX-yCRmpNF3Is/edit?usp=sharing)
