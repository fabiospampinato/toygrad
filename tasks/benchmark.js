
/* IMPORT */

import benchmark from 'benchloop';
import {NeuralNetwork} from '../dist/index.js';

/* HELPERS */

const xorOptions = {
  learningRate: .1,
  layers: [
    {
      inputs: 2,
      outputs: 4,
      activation: 'sigmoid'
    },
    {
      inputs: 4,
      outputs: 1,
      activation: 'sigmoid'
    }
  ]
};

/* MAIN */

benchmark.defaultOptions = Object.assign ( benchmark.defaultOptions, {
  log: 'compact'
});

benchmark.group ( 'xor', () => {

  benchmark ({
    name: 'trainSingle',
    iterations: 1,
    fn: () => {
      const nn = new NeuralNetwork ( xorOptions );
      for ( let i = 0; i < 500_000; i++ ) {
        nn.trainSingle ( [0, 0], [0] );
        nn.trainSingle ( [0, 1], [1] );
        nn.trainSingle ( [1, 0], [1] );
        nn.trainSingle ( [1, 1], [0] );
      }
    }
  });

  benchmark ({
    name: 'trainMultiple',
    iterations: 1,
    fn: () => {
      const nn = new NeuralNetwork ( xorOptions );
      for ( let i = 0; i < 500_000; i++ ) {
        nn.trainMultiple ( [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]] );
      }
    }
  });

});
