
/* IMPORT */

import {describe} from 'fava';
import {relu, sigmoid, softplus, tanh} from '../dist/activations.js';
import {encode, decode} from '../dist/encoder.js';
import {abs, add, ceil, column, count, cube, diagonal, divide, each, each2, fill, floor, from, identity, log2, log10, map, map2, mean, mae, max, min, modulo, mse, multiply, ones, pow, product, random, reduce, reduce2, resize, round, row, scale, sign, size, sqrt, square, subtract, sum, trace, transpose, zeros} from '../dist/ops.js';
import Matrix from '../dist/matrix.js';
import NeuralNetwork from '../dist/neural_network.js';
import ExampleXOR from '../examples/xor.js';

/* HELPERS */

const distance = ( x, y ) => Math.abs ( x - y );

/* MAIN */

describe ( 'Toygrad', () => {

  describe ( 'activations', () => {

    //TODO: identity
    //TODO: leakyrelu
    //TODO: relu
    //TODO: sigmoid
    //TODO: softmax
    //TODO: softplus
    //TODO: tanh
    //TODO: custom

  });

  describe ( 'ops', () => {

    describe ( 'abs', it => {

      it ( 'makes every value positive', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, Math.abs );

        t.deepEqual ( abs ( input ).buffer, output.buffer );

      });

    });

    describe ( 'add', it => {

      it ( 'adds a matrix to another', t => {

        const inputA = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = from ([
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ]);

        const output = from ([
          [2.5, 4.5, 6.5],
          [-6.5, -4.5, -2.5]
        ]);

        t.deepEqual ( add ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'ceil', it => {

      it ( 'rounds up every value', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, Math.ceil );

        t.deepEqual ( ceil ( input ).buffer, output.buffer );

      });

    });

    describe ( 'column', it => {

      it ( 'gets a single column from a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const col0 = from ([
          [1],
          [-3]
        ]);

        const col1 = from ([
          [2],
          [-2]
        ]);

        const col2 = from ([
          [3],
          [-1]
        ]);

        t.deepEqual ( column ( input, 0 ).buffer, col0.buffer );
        t.deepEqual ( column ( input, 1 ).buffer, col1.buffer );
        t.deepEqual ( column ( input, 2 ).buffer, col2.buffer );

      });

    });

    describe ( 'count', it => {

      it ( 'returns the number of values in the matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        t.deepEqual ( count ( input ), 6 );

      });

    });

    describe ( 'cube', it => {

      it ( 'raises every value to the 3rd power', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = map ( input, x => Math.pow ( x, 3 ) );

        t.deepEqual ( cube ( input ).buffer, output.buffer );

      });

    });

    describe ( 'diagonal', it => {

      it ( 'returns values in the diagonal of a square matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = from ([
          [1, 0],
          [0, -2]
        ]);

        t.deepEqual ( diagonal ( input ).buffer, output.buffer );

      });

    });

    describe ( 'divide', it => {

      it ( 'divides a matrix by another', t => {

        const inputA = from ([
          [100, 200, 300],
          [-300, -200, -100]
        ]);

        const inputB = from ([
          [1, 10, 100],
          [100, 10, 1]
        ]);

        const output = from ([
          [100, 20, 3],
          [-3, -20, -100]
        ]);

        t.deepEqual ( divide ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'each', it => {

      it ( 'iterates over every value', t => {

        const input = from ([
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

    //TODO: each2

    describe ( 'fill', it => {

      it ( 'replaces every value with a fixed one', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, () => -1 );

        t.deepEqual ( fill ( input, -1 ).buffer, output.buffer );

      });

    });

    describe ( 'floor', it => {

      it ( 'rounds down every value', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, Math.floor );

        t.deepEqual ( floor ( input ).buffer, output.buffer );

      });

    });

    describe ( 'from', it => {

      it ( 'creates a matrix from plain arrays', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, Math.floor );

        t.deepEqual ( floor ( input ).buffer, output.buffer );

      });

    });

    describe ( 'identity', it => {

      it ( 'creates an identity matrix', t => {

        const output = from ([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ])

        t.deepEqual ( identity ( 3, 3 ).buffer, output.buffer );

      });

    });

    describe ( 'log2', it => {

      it ( 'calculates the log2 of every value', t => {

        const input = from ([
          [1, 2, 3],
          [4, 5, 6]
        ]);

        const output = map ( input, Math.log2 );

        t.deepEqual ( log2 ( input ).buffer, output.buffer );

      });

    });

    describe ( 'log10', it => {

      it ( 'calculates the log10 of every value', t => {

        const input = from ([
          [1, 2, 3],
          [4, 5, 6]
        ]);

        const output = map ( input, Math.log10 );

        t.deepEqual ( log10 ( input ).buffer, output.buffer );

      });

    });

    describe ( 'map', it => {

      it ( 'maps over every value', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = from ([
          [2, 4, 6],
          [-6, -4, -2]
        ]);

        t.deepEqual ( map ( input, x => x * 2 ).buffer, output.buffer );

      });

    });

    //TODO: map2

    describe ( 'mean', it => {

      it ( 'calculats the mean of a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, 9, 0]
        ]);

        t.deepEqual ( mean ( input ), 1 );

      });

    });

    describe ( 'mae', it => {

      it ( 'calculats the mean absolute error between two matrices', t => {

        const input = from ([
          [1, 2],
          [-1, -2]
        ]);

        const output = from ([
          [1, 4],
          [-3, -2]
        ]);

        t.deepEqual ( mae ( input, output ), 1 );

      });

    });

    describe ( 'max', it => {

      it ( 'gets the maximum value of a matrix', t => {

        const input = from ([
          [1, 2],
          [-1, -2]
        ]);

        t.deepEqual ( max ( input ), 2 );

      });

    });

    describe ( 'min', it => {

      it ( 'gets the minimum value of a matrix', t => {

        const input = from ([
          [1, 2],
          [-1, -2]
        ]);

        t.deepEqual ( min ( input ), -2 );

      });

    });

    describe ( 'modulo', it => {

      it ( 'calculates the modulo for every value', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, x => x % 2 );

        t.deepEqual ( modulo ( input, 2 ).buffer, output.buffer );

      });

    });

    describe ( 'mse', it => {

      it ( 'calculats the mean square error between two matrices', t => {

        const input = from ([
          [1, 2],
          [-1, -2]
        ]);

        const output = from ([
          [1, 4],
          [-3, -2]
        ]);

        t.deepEqual ( mse ( input, output ), 2 );

      });

    });

    describe ( 'multiply', it => {

      it ( 'multiplies a matrix from another', t => {

        const inputA = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = from ([
          [1, 2, 3],
          [3, 2, 1]
        ]);

        const output = from ([
          [1, 4, 9],
          [-9, -4, -1]
        ]);

        t.deepEqual ( multiply ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'ones', it => {

      it ( 'returns a matrix filled with ones', t => {

        const output = fill ( new Matrix ( 5, 10 ), 1 );

        t.deepEqual ( ones ( 5, 10 ), output );

      });

    });

    describe ( 'pow', it => {

      it ( 'raises every value to a power', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = map ( input, x => Math.pow ( x, 7 ) );

        t.deepEqual ( pow ( input, 7 ).buffer, output.buffer );

      });

    });

    describe ( 'product', it => {

      it ( 'multiples a matrix from another, rows by columns', t => {

        const inputA = from ([
          [1, 2],
          [4, 3]
        ]);

        const inputB = from ([
          [1, 2, 3],
          [3, -4, 7]
        ]);

        const output = from ([
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

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ]);

        t.true ( distance ( reduce ( input, ( acc, x ) => acc + x, 0 ), 3.21 ) < 0.000001 );

      });

    });

    //TODO: reduce2

    describe ( 'resize', it => {

      it ( 'resizes a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const smaller = from ([
          [1, 2],
          [-1, -2]
        ]);

        const bigger = from ([
          [1, 2, 3, 0],
          [-1, -2, -3, 0],
          [0, -0.123, -0, 0],
          [0, 0, 0, 0]
        ]);

        t.deepEqual ( resize ( input, 2, 2 ).buffer, smaller.buffer );
        t.deepEqual ( resize ( input, 4, 4 ).buffer, bigger.buffer );

      });

    });

    describe ( 'round', it => {

      it ( 'rounds every value', t => {

        const input = from ([
          [1, 2, 3],
          [-1, -2, -3],
          [0, -0.123, -0]
        ]);

        const output = map ( input, Math.round );

        t.deepEqual ( round ( input ).buffer, output.buffer );

      });

    });

    describe ( 'row', it => {

      it ( 'gets a single column from a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const row0 = from ([
          [1, 2, 3]
        ]);

        const row1 = from ([
          [-3, -2, -1]
        ]);

        t.deepEqual ( row ( input, 0 ).buffer, row0.buffer );
        t.deepEqual ( row ( input, 1 ).buffer, row1.buffer );

      });

    });

    describe ( 'scale', it => {

      it ( 'multiplies every value by a factor', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = from ([
          [2, 4, 6],
          [-6, -4, -2]
        ]);

        t.deepEqual ( scale ( input, 2 ).buffer, output.buffer );

      });

    });

    describe ( 'sign', it => {

      it ( 'gets a single column from a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = from ([
          [1, 1, 1],
          [-1, -1, -1]
        ]);

        t.deepEqual ( sign ( input ).buffer, output.buffer );

      });

    });

    describe ( 'size', it => {

      it ( 'returns the dimensions of the matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        t.deepEqual ( size ( input ), [2, 3] );

      });

    });

    describe ( 'sqrt', it => {

      it ( 'calculates the square root of every value', t => {

        const input = from ([
          [1, 2, 3],
          [4, 5, 6]
        ]);

        const output = map ( input, Math.sqrt );

        t.deepEqual ( sqrt ( input ).buffer, output.buffer );

      });

    });

    describe ( 'square', it => {

      it ( 'raises every value to the 2nd power', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = map ( input, x => Math.pow ( x, 2 ) );

        t.deepEqual ( square ( input ).buffer, output.buffer );

      });

    });

    describe ( 'subtract', it => {

      it ( 'subtracts a matrix from another', t => {

        const inputA = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const inputB = from ([
          [1.5, 2.5, 3.5],
          [-3.5, -2.5, -1.5]
        ]);

        const output = from ([
          [-.5, -.5, -.5],
          [.5, .5, .5]
        ]);

        t.deepEqual ( subtract ( inputA, inputB ).buffer, output.buffer );

      });

    });

    describe ( 'sum', it => {

      it ( 'sums every value', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1],
          [1, 1.1, 1.11]
        ]);

        t.true ( distance ( sum ( input ), 3.21 ) < 0.000001 );

      });

    });

    describe ( 'trace', it => {

      it ( 'returns the sum of values in the diagonal of a square matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        t.deepEqual ( trace ( input ), -1 );

      });

    });

    describe ( 'transpose', it => {

      it ( 'transposes a matrix', t => {

        const input = from ([
          [1, 2, 3],
          [-3, -2, -1]
        ]);

        const output = from ([
          [1, -3],
          [2, -2],
          [3, -1]
        ]);

        t.deepEqual ( transpose ( input ).buffer, output.buffer );

      });

    });

    describe ( 'zeros', it => {

      it ( 'returns a matrix filled with zeros', t => {

        const output = new Matrix ( 5, 10 );

        t.deepEqual ( zeros ( 5, 10 ), output );

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

      for ( let i = 1; i < 1000; i++ ) {
        nn1.trainSingle ( [0, 0], [0] );
        nn1.trainSingle ( [0, 1], [1] );
        nn1.trainSingle ( [1, 0], [1] );
        nn1.trainSingle ( [1, 1], [0] );
      }

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

      for ( let i = 1; i < 1000; i++ ) {
        nn1.trainSingle ( [0, 0], [0] );
        nn1.trainSingle ( [0, 1], [1] );
        nn1.trainSingle ( [1, 0], [1] );
        nn1.trainSingle ( [1, 1], [0] );
      }

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

    it ( 'throws if layers\' inputs and ouputs do not match up', t => {

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

  describe ( 'encoder', it => {

    it ( 'support encoder and decoding', t => {

      const input = from ([
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
