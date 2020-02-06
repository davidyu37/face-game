let brain;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 136,
    outputs: 2,
    task: "classification",
    debug: true
  };
  brain = ml5.neuralNetwork(options);
  brain.loadData("raisedbrow1.json", dataReady);
}

function dataReady() {
  brain.normalizeData();
  brain.train({ epochs: 500 }, finished);
}

function finished() {
  console.log("model trained", brain);
  brain.save();
}
