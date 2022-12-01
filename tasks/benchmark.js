
/* IMPORT */

import benchmark from 'benchloop';
import Matrix from '../dist/matrix.js';
import {NeuralNetwork} from '../dist/index.js';

/* HELPERS */

const xorOptions = {
  learningRate: 1,
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
      const input1 = [0, 0];
      const input2 = [0, 1];
      const input3 = [1, 0];
      const input4 = [1, 1];
      const output1 = [0];
      const output2 = [1];
      const output3 = [1];
      const output4 = [0];
      for ( let i = 0; i < 500_000; i++ ) {
        nn.trainSingle ( input1, output1 );
        nn.trainSingle ( input2, output2 );
        nn.trainSingle ( input3, output3 );
        nn.trainSingle ( input4, output4 );
      }
    }
  });

  benchmark ({
    name: 'trainMultiple',
    iterations: 1,
    fn: () => {
      const nn = new NeuralNetwork ( xorOptions );
      const inputs = Matrix.from ( [[0, 0], [0, 1], [1, 0], [1, 1]] );
      const outputs = Matrix.from ( [[0], [1], [1], [0]] );
      for ( let i = 0; i < 500_000; i++ ) {
        nn.trainMultiple ( inputs, outputs );
      }
    }
  });

});
