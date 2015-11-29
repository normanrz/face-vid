# face-vid
A research project for facial expression detection by utilizing state of the art depp convultional neural networks.

### Installation

- Install Caffe and PyCaffe
- Install OpenCV and OpenCV for Python 2.7
- Install NodeJS / NPM for Web Demo
- Install Python 2.7 dependencies

```
pip install -r requirements.txt
pip install -r web-server/requirements.txt
```
- Install the Web-Servers dependencies

```
cd web-server
npm install -g webpack
npm install
```
- Download the MMI AOA FACS Dataset from [http://www.mmifacedb.eu/](http://www.mmifacedb.eu/) (ca. 329 videos)


### Frame Extraction
To extract all frames of MMI videos + preprocess them use the `extract_frames.py` script:

```
python extract_frames.py <videos_path> <output_path>

e.g.
python extract_frames.py /datasets/mmi/Sessions /datasets/mmi/hdf5-output
```

THis will extract all FACS annotated frame per video, crop the frames to only contain faces, equalize the colors, calculate the optical flow, subtract the global means and save everything as a train/test split HDF5 database.


### Modell Training

To fine-tune the spatial AlexNet network run:
```
./face-vid-nets/train_finetune_network.sh
```

To train the optical flow temporal network run:
```
./face-vid-nets/train_flow_network.sh
```


### Web Demo
The project includes a web demo to easily record videos of yourself using your webcam and start a prediction. 

Start the web-server with:
```
cd web-server
python server.py 
```

The demo will be running on port 9000: [http://localhost:9000](http://localhost:9000) 

##### Trouble Shooting
- Demo requires a very modern browser with WebRTC. Successfully tested with Chrome
- Have NodeJS and Webpack installed
- Install all NodeJS dependencies 
- Compile ES6 + React code `npm run build`
- Train networks before starting demo

