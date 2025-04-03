## Two-stage Vehicle Trajectory Prediction Method Based on Adaptive Map Retrieval(TAMR)

<p align="center"><img src="fig/TAMR.png" width="600" /></p>

## Running
python train.py -m TAMR

python test.py -m TAMR  --weight=/absolute/path/to/45.000.ckpt --split=test/val

python test1vis.py -m TAMR --weight=/absolute/path/to/45.000.ckpt --split=val


## Visualization
<p align="center"><img src="fig/go straight.png" width="600" /></p>
<p align="center"><img src="fig/turn left.png" width="600" /></p>
<p align="center"><img src="fig/turn right.png" width="600" /></p>

