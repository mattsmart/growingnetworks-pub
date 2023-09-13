# growingnetworks
Building networks from oscillations on nodes

## Requirements 
* Python 3.8 (for compaitibility with `pygraphviz`, see below). 
* Install the python packages in requirements.txt using `python -m pip install -r requirements.txt`
* Install Graphviz 7.0.1 (used with `networkx` package to visualize graphs)
* Install package `pygraphviz` as described here https://pygraphviz.github.io/documentation/stable/install  
note: on Windows, the steps described there fail on Python 3.10 but work on Python 3.8 -- use this command:  
`python -m pip install --global-option="build_ext" --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib" pygraphviz==1.10`  
