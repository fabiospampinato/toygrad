
/* IMPORT */

import {describe} from 'fava';
import {relu, sigmoid, softplus, tanh} from '../dist/activations.js';
import {abs, add, divide, each, map, mean, multiply, product, random, reduce, scale, subtract, sum, transpose} from '../dist/ops.js';
import {encode, decode} from '../dist/weights.js';
import Matrix from '../dist/matrix.js';
import NeuralNetwork from '../dist/neural_network.js';
import ExampleXOR from '../examples/xor.js';

/* HELPERS */

const distance = ( x, y ) => Math.abs ( x - y );

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

      it ( 'makes every value positive', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = Matrix.from ([
          [1, 2, 3],
          [1, 2, 3],
          [0, 0.123, 0]
        ]);

        t.deepEqual ( abs ( input ).buffer, output.buffer );

      });

    });

    describe ( 'add', it => {

      it ( 'adds a matrix from another', t => {

        const inputA = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = Matrix.from ([
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ]);

        const output = Matrix.from ([
          [2.5, 4.5, 6.5],
          [-6.5, -4.5, -2.5]
        ]);

        t.deepEqual ( add ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'divide', it => {

      it ( 'divides a matrix from another', t => {

        const inputA = Matrix.from ([
          [100, 200, 300],
          [-300, -200, -100]
        ]);

        const inputB = Matrix.from ([
          [1, 10, 100],
          [100, 10, 1]
        ]);

        const output = Matrix.from ([
          [100, 20, 3],
          [-3, -20, -100]
        ]);

        t.deepEqual ( divide ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'each', it => {

      it ( 'iterates over every value', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const received = [];
        const expected = [1, 2, 3, -3, -2, -1];
        const result = each ( input, x => received.push ( x ) );

        t.is ( result, undefined );
        t.deepEqual ( received, expected );

      });

    });

    describe ( 'map', it => {

      it ( 'maps over every value', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = Matrix.from ([
          [2, 4, 6],
          [-6, -4, -2]
        ]);

        t.deepEqual ( map ( input, x => x * 2 ).buffer, output.buffer );

      });

    });

    describe ( 'mean', it => {

      it ( 'returns the mean of the values', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ]);

        t.true ( distance ( mean ( input ), 3.21 / 9 ) < 0.000001 );

      });

    });

    describe ( 'multiply', it => {

      it ( 'multiplies a matrix from another', t => {

        const inputA = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = Matrix.from ([
          [1, 2, 3],
          [3, 2, 1]
        ]);

        const output = Matrix.from ([
          [1, 4, 9],
          [-9, -4, -1]
        ]);

        t.deepEqual ( multiply ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'product', it => {

      it ( 'multiples a matrix from another, rows by columns', t => {

        const inputA = Matrix.from ([
          [1, 2],
          [4, 3]
        ]);

        const inputB = Matrix.from ([
          [1, 2, 3],
          [3, -4, 7]
        ]);

        const output = Matrix.from ([
          [7, -6, 17],
          [13, -4, 33]
        ]);

        t.deepEqual ( product ( inputA, inputB ).buffer, output.buffer );

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

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ]);

        t.true ( distance ( reduce ( input, ( acc, x ) => acc + x, 0 ), 3.21 ) < 0.000001 );

      });

    });

    describe ( 'scale', it => {

      it ( 'multiplies every value by a factor', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = Matrix.from ([
          [2, 4, 6],
          [-6, -4, -2]
        ]);

        t.deepEqual ( scale ( input, 2 ).buffer, output.buffer );

      });

    });

    describe ( 'subtract', it => {

      it ( 'subtracts a matrix from another', t => {

        const inputA = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = Matrix.from ([
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ]);

        const output = Matrix.from ([
          [-.5, -.5, -.5],
          [.5, .5, .5]
        ]);

        t.deepEqual ( subtract ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'sum', it => {

      it ( 'sums every value', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ]);

        t.true ( distance ( sum ( input ), 3.21 ) < 0.000001 );

      });

    });

    describe ( 'transpose', it => {

      it ( 'transposes a matrix', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = Matrix.from ([
          [1, -3],
          [2, -2],
          [3, -1]
        ]);

        t.deepEqual ( transpose ( input ).buffer, output.buffer );

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
            activation: 'tanh'
          },
          {
            inputs: 4,
            outputs: 1,
            activation: 'tanh'
          }
        ],
        learningRate: .1
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
            activation: 'tanh'
          },
          {
            inputs: 4,
            outputs: 1,
            activation: 'tanh'
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
          learningRate: .1,
          layers: []
        });

      } catch ( error ) {

        t.is ( error.message, 'Only a fixed 2 layers of weights are supported for now, sorry' );

      }

    });

    it ( 'throws if layers\' inputs and ouputs don\'t match up', t => {

      try {

        new NeuralNetwork ({
          learningRate: .1,
          layers: [
            {
              inputs: 1,
              outputs: 5,
              activation: 'tanh'
            },
            {
              inputs: 4,
              outputs: 1,
              activation: 'tanh'
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

      it ( 'support encoding and decoding', t => {

        const input = Matrix.from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.5, 1.11]
        ]);

        const encoded = encode ( input );
        const decoded = decode ( encoded );

        t.is ( encoded.length, 74 );
        t.deepEqual ( input.buffer, decoded.buffer );

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
