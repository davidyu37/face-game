let faceapi;
let video;
let detections;
let width = 360;
let height = 280;
let canvas, ctx;
let collecting = false;
let data = [];
let collectBtn = null;
let brain = null;
let label = "raisedbrow";
let labelBtn = null;

// relative path to your models from window.location.pathname
const detection_options = {
  withLandmarks: true,
  withDescriptors: false,
  Mobilenetv1Model: "models",
  FaceLandmarkModel: "models",
  FaceRecognitionModel: "models"
};

async function make() {
  // get the video
  video = await getVideo();

  canvas = createCanvas(width, height);
  ctx = canvas.getContext("2d");

  faceapi = ml5.faceApi(video, detection_options, modelReady);
  loadBrain();
}

function loadBrain() {
  let options = {
    inputs: 136,
    outputs: 2,
    task: "classification"
  };
  brain = ml5.neuralNetwork(options, function() {
    console.log("brain loaded");
  });
}
// call app.map.init() once the DOM is loaded
window.addEventListener("DOMContentLoaded", function() {
  make();
  collectBtn = addButton("collect", toggleDataCollection);
  labelBtn = addButton("label", toggleLabel);
  addButton("save", saveData);
});

function addButton(id, action) {
  const btnElem = document.getElementById(id);
  btnElem.addEventListener("click", action);

  return btnElem;
}

function toggleLabel() {
  if (label === "raisedbrow") {
    label = "normal";
    labelBtn.innerText = "Normal";
    return;
  }
  label = "raisedbrow";
  labelBtn.innerText = "Raised";
}

function toggleDataCollection() {
  if (collecting) {
    collectBtn.innerText = "Collect";
  } else {
    collectBtn.innerText = "Stop";
  }
  collecting = !collecting;
}

function saveData() {
  brain.saveData();
}

function modelReady() {
  console.log("ready!");
  faceapi.detect(gotResults);
}

function gotResults(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  if (collecting && result[0]) {
    console.log(result[0]);
    const {
      landmarks: { positions }
    } = result[0];
    recordData(positions);
  }

  detections = result;

  // Clear part of the canvas
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);

  ctx.drawImage(video, 0, 0, width, height);

  if (detections) {
    if (detections.length > 0) {
      drawBox(detections);
      drawLandmarks(detections);
    }
  }
  faceapi.detect(gotResults);
}

function recordData(positions) {
  console.log(positions);
  const inputs = [];
  positions.forEach(position => {
    const { _x, _y } = position;
    inputs.push(_x);
    inputs.push(_y);
  });

  brain.addData(inputs, [label]);

  // const rightEyeBrow = parts.rightEyeBrow;
  // const leftEyeBrow = parts.leftEyeBrow;
  // const inputs = [];

  // rightEyeBrow.forEach(point => {
  //   const { x, y } = point;
  //   inputs.push(x);
  //   inputs.push(y);
  // });

  // leftEyeBrow.forEach(point => {
  //   const { x, y } = point;
  //   inputs.push(x);
  //   inputs.push(y);
  // });

  // brain.addData(inputs, [label]);
}

function drawBox(detections) {
  for (let i = 0; i < detections.length; i++) {
    const alignedRect = detections[i].alignedRect;
    const x = alignedRect._box._x;
    const y = alignedRect._box._y;
    const boxWidth = alignedRect._box._width;
    const boxHeight = alignedRect._box._height;

    ctx.beginPath();
    ctx.rect(x, y, boxWidth, boxHeight);
    ctx.strokeStyle = "#a15ffb";
    ctx.stroke();
    ctx.closePath();
  }
}

function drawLandmarks(detections) {
  for (let i = 0; i < detections.length; i++) {
    const mouth = detections[i].parts.mouth;
    const nose = detections[i].parts.nose;
    const leftEye = detections[i].parts.leftEye;
    const rightEye = detections[i].parts.rightEye;
    const rightEyeBrow = detections[i].parts.rightEyeBrow;
    const leftEyeBrow = detections[i].parts.leftEyeBrow;

    drawPart(mouth, true);
    drawPart(nose, false);
    drawPart(leftEye, true);
    drawPart(leftEyeBrow, false);
    drawPart(rightEye, true);
    drawPart(rightEyeBrow, false);
  }
}

function drawPart(feature, closed) {
  ctx.beginPath();
  for (let i = 0; i < feature.length; i++) {
    const x = feature[i]._x;
    const y = feature[i]._y;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  if (closed === true) {
    ctx.closePath();
  }
  ctx.stroke();
}

// Helper Functions
async function getVideo() {
  // Grab elements, create settings, etc.
  const videoElement = document.createElement("video");
  videoElement.setAttribute("style", "display: none;");
  videoElement.width = width;
  videoElement.height = height;
  document.body.appendChild(videoElement);

  // Create a webcam capture
  const capture = await navigator.mediaDevices.getUserMedia({ video: true });
  videoElement.srcObject = capture;
  videoElement.play();

  return videoElement;
}

function createCanvas(w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  document.body.appendChild(canvas);
  return canvas;
}