let brain;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 74,
    outputs: 2,
    task: "classification",
    debug: true
  };
  brain = ml5.neuralNetwork(options);
  brain.loadData("openmouth.json", dataReady);
}

function dataReady() {
  brain.normalizeData();
  brain.train({ epochs: 500 }, finished);
}

function finished() {
  console.log("model trained", brain);
  brain.save();
}
