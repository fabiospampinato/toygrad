
/* IMPORT */

import {NeuralNetwork} from '../dist/index.js';

/* MAIN */

const nn = new NeuralNetwork ({
  inputLayer: 2,
  hiddenLayer: 10,
  outputLayer: 1,
  learningRate: .1,
  activation: 'sigmoid',

  layers: [
    {
      neurons: 2,
      activation: 'sigmoid'
    },
    {
      neurons: 4,
      activation: 'sigmoid'
    },
    {
      neurons: 1,
      activation: 'sigmoid'

    }
  ]
});

for ( let i = 1; i < 50_000; i++ ) {
  // nn.trainSingle([[0,0], [0,1], [1,0], [1,1]], [[0],[1],[1],[0]])
  nn.trainSingle([0, 0], [0]);
  nn.trainSingle([0, 1], [1]);
  nn.trainSingle([1, 0], [1]);
  nn.trainSingle([1, 1], [0]);
}

// for ( let i = 1; i < 10_000; i++ ) {
//   nn.train([0, 0], [0]);
//   nn.train([0, 1], [1]);
//   nn.train([1, 0], [1]);
//   nn.train([1, 1], [0]);
// }

console.log(nn.predict([0,0]));
console.log(nn.predict([1,0]));
console.log(nn.predict([0,1]));
console.log(nn.predict([1,1]));

/* EXPORT */

export default nn;
