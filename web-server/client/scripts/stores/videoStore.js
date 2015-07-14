import _ from "lodash"
import alt from '../alt';
import VideoActions from '../actions/videoActions';
import API from "../lib/api"

import VideoThumbnail1 from "../../images/examples/video1.png"
import VideoThumbnail2 from "../../images/examples/video2.png"
import VideoThumbnail3 from "../../images/examples/video3.png"


class VideoStore {

  constructor() {
    this.bindActions(VideoActions);

    this.isUploading = false;
    this.isInvalidFile = false;
    this.exampleVideos = [
      {
        id : 1,
        thumbnail : "/images/examples/video1.png",
        title : "Happiness",
        activity : "Action Units: 12, 17, 18, 25, 45, 6"
      },
      {
        id : 2,
        thumbnail : "/images/examples/video2.png",
        title : "Anger",
        activity : "Action Units: 4, 24, 45"
      },
      {
        id : 3,
        thumbnail : "/images/examples/video3.png",
        title : "Disgust",
        activity : "Action Units: 4, 9, 10, 11, 17, 38"
      }
    ];
  }

  static getExampleVideos() {
    return this.getState().exampleVideos;
  }

  onUploadVideo(videoFile) {

    const payload = {
      video : videoFile
    };

    API.postVideo(payload)
    this.isUploading = true;
  }

  onUseExample(videoId) {

    API.getPredictionForExample(videoId);
    this.isUploading = true;
  }

  onReceivePrediction() {
    this.isUploading = false;
    this.isInvalidFile = false;
  }

  onReceiveUploadError() {
    this.isUploading = false;
    this.isInvalidFile = true;
  }

};

export default alt.createStore(VideoStore, "VideoStore");