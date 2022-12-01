
/* IMPORT */

import {describe} from 'fava';
import {relu, sigmoid, softplus, tanh} from '../dist/activations.js';
import {abs, add, clone, create, divide, each, each2, map, mean, multiply, product, random, reduce, scale, subtract, sum, transpose} from '../dist/ops.js';
import {encode, decode} from '../dist/weights.js';
import NeuralNetwork from '../dist/neural_network.js';
import ExampleXOR from '../examples/xor.js';

/* MAIN */

describe ( 'Toygrad', () => {

  describe ( 'activations', () => {

    describe ( 'relu', it => {

      //TODO

    });

    describe ( 'sigmoid', it => {

      //TODO

    });

    describe ( 'softplus', it => {

      //TODO

    });

    describe ( 'tanh', it => {

      //TODO

    });

  });

  describe ( 'ops', () => {

    describe ( 'abs', it => {

      it ( 'does not mutate arguments', t => {

        const input = [[-1]];
        const output = abs ( input );

        t.not ( input, output );
        t.deepEqual ( input, [[-1]] );

      });

      it ( 'makes every value positive', t => {

        const input = [
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ];

        const output = [
          [1, 2, 3],
          [1, 2, 3],
          [0, 0.123, 0]
        ];

        t.deepEqual ( abs ( input ), output );

      });

    });

    describe ( 'add', it => {

      it ( 'adds a matrix from another', t => {

        const inputA = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const inputB = [
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ];

        const output = [
          [2.5, 4.5, 6.5],
          [-6.5, -4.5, -2.5]
        ];

        t.deepEqual ( add ( inputA, inputB ), output );

      });

    });

    describe ( 'clone', it => {

      it ( 'clones a matrix', t => {

        const matrix = random ( 10, 20, -1, 1 );
        const cmatrix = clone ( matrix );

        t.true ( cmatrix.length === 10 );
        t.true ( cmatrix.every ( row => row.length === 20 ) );
        t.not ( matrix, cmatrix );

        each2 ( matrix, cmatrix, ( a, b ) => {
          t.is ( a, b );
        });

      });

    });

    describe ( 'create', it => {

      it ( 'creates an empty matrix', t => {

        const matrix = create ( 10, 20 );

        t.true ( matrix.length === 10 );
        t.true ( matrix.every ( row => row.length === 20 ) );
        t.is ( sum ( matrix ), 0 );

      });

      it ( 'creates a matrix filled with a single value', t => {

        const matrix = create ( 10, 20, 1 );

        t.is ( sum ( matrix ), 200 );

      });

    });

    describe ( 'divide', it => {

      it ( 'divides a matrix from another', t => {

        const inputA = [
          [100, 200, 300],
          [-300, -200, -100]
        ];

        const inputB = [
          [1, 10, 100],
          [100, 10, 1]
        ];

        const output = [
          [100, 20, 3],
          [-3, -20, -100]
        ];

        t.deepEqual ( divide ( inputA, inputB ), output );

      });

    });

    describe ( 'each', it => {

      it ( 'iterates over every value', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const received = [];
        const expected = [1, 2, 3, -3, -2, -1];
        const result = each ( input, x => received.push ( x ) );

        t.is ( result, undefined );
        t.deepEqual ( received, expected );

      });

    });

    describe ( 'map', it => {

      it ( 'does not mutate arguments', t => {

        const input = [[-1]];
        const output = map ( input, x => x * 2 );

        t.not ( input, output );
        t.deepEqual ( input, [[-1]] );

      });

      it ( 'maps over every value', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const output = [
          [2, 4, 6],
          [-6, -4, -2]
        ];

        t.deepEqual ( map ( input, x => x * 2 ), output );

      });

    });

    describe ( 'mean', it => {

      it ( 'returns the mean of the values', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ];

        t.deepEqual ( mean ( input ), 3.21 / 9 );

      });

    });

    describe ( 'multiply', it => {

      it ( 'multiplies a matrix from another', t => {

        const inputA = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const inputB = [
          [1, 2, 3],
          [3, 2, 1]
        ];

        const output = [
          [1, 4, 9],
          [-9, -4, -1]
        ];

        t.deepEqual ( multiply ( inputA, inputB ), output );

      });

    });

    describe ( 'product', it => {

      it ( 'multiples a matrix from another, rows by columns', t => {

        const inputA = [
          [1, 2],
          [4, 3]
        ];

        const inputB = [
          [1, 2, 3],
          [3, -4, 7]
        ];

        const output = [
          [7, -6, 17],
          [13, -4, 33]
        ];

        t.deepEqual ( product ( inputA, inputB ), output );

      });

    });

    describe ( 'random', it => {

      it ( 'instantiates a random matrix', t => {

        const matrix = random ( 1000, 1000, -1, 1 );

        t.true ( sum ( matrix ) < 2000 );
        t.true ( sum ( matrix ) > -2000 );

      });

    });

    describe ( 'reduce', it => {

      it ( 'reduces over every value', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ];

        t.deepEqual ( reduce ( input, ( acc, x ) => acc + x, 0 ), 3.21 );

      });

    });

    describe ( 'scale', it => {

      it ( 'does not mutate arguments', t => {

        const input = [[-1]];
        const output = scale ( input, 2 );

        t.not ( input, output );
        t.deepEqual ( input, [[-1]] );

      });

      it ( 'multiplies every value by a factor', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const output = [
          [2, 4, 6],
          [-6, -4, -2]
        ];

        t.deepEqual ( scale ( input, 2 ), output );

      });

    });

    describe ( 'subtract', it => {

      it ( 'subtracts a matrix from another', t => {

        const inputA = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const inputB = [
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ];

        const output = [
          [-.5, -.5, -.5],
          [.5, .5, .5]
        ];

        t.deepEqual ( subtract ( inputA, inputB ), output );

      });

    });

    describe ( 'sum', it => {

      it ( 'sums every value', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ];

        t.deepEqual ( sum ( input ), 3.21 );

      });

    });

    describe ( 'transpose', it => {

      it ( 'does not mutate arguments', t => {

        const input = [[-1]];
        const output = scale ( input, 2 );

        t.not ( input, output );
        t.deepEqual ( input, [[-1]] );

      });

      it ( 'transposes a matrix', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1]
        ];

        const output = [
          [1, -3],
          [2, -2],
          [3, -1]
        ];

        t.deepEqual ( transpose ( input ), output );

      });

    });

  });

  describe ( 'nn', it => {

    it ( 'can export the model as options and import it again', t => {

      const nn1 = new NeuralNetwork ({
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
        ],
        learningRate: .1,
        precision: 'float64'
      });

      const i00 = nn1.infer ( [0, 0] )[0];
      const i10 = nn1.infer ( [1, 0] )[0];
      const i01 = nn1.infer ( [0, 1] )[0];
      const i11 = nn1.infer ( [1, 1] )[0];

      const nn2 = new NeuralNetwork ( nn1.exportAsOptions () );

      const o00 = nn2.infer ( [0, 0] )[0];
      const o10 = nn2.infer ( [1, 0] )[0];
      const o01 = nn2.infer ( [0, 1] )[0];
      const o11 = nn2.infer ( [1, 1] )[0];

      t.is ( i00, o00 );
      t.is ( i10, o10 );
      t.is ( i01, o01 );
      t.is ( i11, o11 );

    });

    it ( 'can export the model as a standalone function', t => {

      const nn1 = new NeuralNetwork ({
        learningRate: .1,
        layers: [
          {
            inputs: 2,
            outputs: 4,
            activation: 'sigmoid',
            weights: [
              [0, .5, 1, 1.5],
              [-0, -.5, -1, -1.5]
            ]
          },
          {
            inputs: 4,
            outputs: 1,
            activation: 'sigmoid',
            weights: [
              [0],
              [.5],
              [1],
              [1.5]
            ]
          }
        ]
      });

      const i00 = nn1.infer ( [0, 0] )[0];
      const i10 = nn1.infer ( [1, 0] )[0];
      const i01 = nn1.infer ( [0, 1] )[0];
      const i11 = nn1.infer ( [1, 1] )[0];

      const fn = nn1.exportAsFunction ();

      const o00 = fn ( [0, 0] )[0];
      const o10 = fn ( [1, 0] )[0];
      const o01 = fn ( [0, 1] )[0];
      const o11 = fn ( [1, 1] )[0];

      t.is ( i00, o00 );
      t.is ( i10, o10 );
      t.is ( i01, o01 );
      t.is ( i11, o11 );

    });

    it ( 'throws if the number of layers is incorrect', t => {

      try {

        new NeuralNetwork ({
          learningRate: 1,
          layers: []
        });

      } catch ( error ) {

        t.is ( error.message, 'Only a fixed 2 layers of weights are supported for now, sorry' );

      }

    });

    it ( 'throws if layers\' inputs and ouputs don\'t match up', t => {

      try {

        new NeuralNetwork ({
          learningRate: 1,
          layers: [
            {
              inputs: 1,
              outputs: 5,
              activation: 'sigmoid'
            },
            {
              inputs: 4,
              outputs: 1,
              activation: 'sigmoid'
            }
          ]
        });

      } catch ( error ) {

        t.is ( error.message, 'The number of outputs of a layer must match the number of inputs of the next layer' );

      }

    });

  });

  describe ( 'weights', () => {

    describe ( 'encoding', it => {

      it.todo ( 'support encoding and decoding, with float16 precision' );

      it ( 'support encoding and decoding, with float32 precision', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.5, 1.25]
        ];

        const encoded = encode ( input, 'float32' );
        const decoded = decode ( encoded, 'float32' );

        t.is ( encoded.length, 74 );
        t.deepEqual ( input, decoded );

      });

      it ( 'support encoding and decoding, with float64 precision', t => {

        const input = [
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ];

        const encoded = encode ( input, 'float64' );
        const decoded = decode ( encoded, 'float64' );

        t.is ( encoded.length, 146 );
        t.deepEqual ( input, decoded );

      });

    });

  });

  describe ( 'examples', () => {

    describe ( 'xor', it => {

      it ( 'works', t => {

        t.true ( ExampleXOR.infer ( [0, 0] )[0] < .1 );
        t.true ( ExampleXOR.infer ( [1, 0] )[0] > .9 );
        t.true ( ExampleXOR.infer ( [0, 1] )[0] > .9 );
        t.true ( ExampleXOR.infer ( [1, 1] )[0] < .1 );

      });

    });

  });

});
