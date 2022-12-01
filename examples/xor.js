
/* IMPORT */

import {NeuralNetwork} from '../dist/index.js';

/* MAIN */

const nn = new NeuralNetwork ({
  learningRate: .1,
  layers: [
    {
      inputs: 2,
      outputs: 400,
      activation: 'tanh'
    },
    {
      inputs: 400,
      outputs: 1,
      activation: 'tanh'
    }
  ]
});

nn.trainLoop ( 800, () => {
  return nn.trainMultiple ( [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]] );
});

// for ( let i = 1; i < 50_000; i++ ) {
//   nn.trainSingle ( [0, 0], [0] );
//   nn.trainSingle ( [0, 1], [1] );
//   nn.trainSingle ( [1, 0], [1] );
//   nn.trainSingle ( [1, 1], [0] );
// }

// for ( let i = 1; i < 50_000; i++ ) {
//   nn.trainMultiple ( [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]] );
// }

console.log ( '' );
console.log ( '0^0 ->', nn.infer ( [0, 0] )[0] );
console.log ( '1^0 ->', nn.infer ( [1, 0] )[0] );
console.log ( '0^1 ->', nn.infer ( [0, 1] )[0] );
console.log ( '1^1 ->', nn.infer ( [1, 1] )[0] );
console.log ( '' );

/* EXPORT */

export default nn;
