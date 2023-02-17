
// DATASET: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
// DEMO: https://playground.solidjs.com/anonymous/7b1e3399-8b30-4fdb-9675-b85a5ab2d6dc

/* IMPORT */

import _ from 'lodash';
import {map, matrix, zeros} from 'mathjs';
import fs from 'node:fs';
import {NeuralNetwork, Tensor, Trainers} from '../dist/index.js';

/* HELPERS */

const FULL_SIZE = 28;
const CROP_SIZE = 24;

const EPOCHS_LIMIT = 2;
const TRAIN_LIMIT = Infinity;
const TEST_LIMIT = Infinity;

/* PARSE */

const argmax = arr => {
  return arr.indexOf ( Math.max ( ...Array.from ( arr ) ) );
};

const crop = input => {
  if ( FULL_SIZE === CROP_SIZE ) return new Tensor ( FULL_SIZE, FULL_SIZE, 1, new Float32Array ( input ) );
  const offsetRows = Math.round ( Math.random () * ( FULL_SIZE - CROP_SIZE ) );
  const offsetCols = Math.round ( Math.random () * ( FULL_SIZE - CROP_SIZE ) );
  const matrixInput = matrix ( _.chunk ( input, FULL_SIZE ) );
  const matrixBase = matrix ( zeros ( [CROP_SIZE, CROP_SIZE] ) );
  const matrixOutput = map ( matrixBase, ( _, [row, column] ) => matrixInput.get ( [row + offsetRows, column + offsetCols] ) );
  return new Tensor ( CROP_SIZE, CROP_SIZE, 1, new Float32Array ( matrixOutput.toArray ().flat () ) );
};

const parseCSV = csv => {
  const values = csv.split ( ',' ).map ( Number );
  const output = values[0];
  const input = crop ( values.slice ( 1 ).map ( x => x / 255 ) );
  return {input, output};
};

const TRAIN_SET = fs.readFileSync ( './examples/mnist_train.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );
const TEST_SET = fs.readFileSync ( './examples/mnist_test.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );

/* TRAIN */

const nn = new NeuralNetwork ({
  layers: [
    { type: 'input', sx: CROP_SIZE, sy: CROP_SIZE, sz: 1 },
    { type: 'conv', sx: 5, filters: 6, stride: 1, pad: 2, bias: 0.1 },
    { type: 'relu' },
    { type: 'pool', sx: 2, stride: 2 },
    { type: 'conv', sx: 5, filters: 12, stride: 1, pad: 2, bias: 0.1 },
    { type: 'relu' },
    { type: 'pool', sx: 3, stride: 3 },
    { type: 'dense', filters: 10 },
    { type: 'softmax' }
  ]
});

const trainer = new Trainers.Adadelta ( nn, {
  batchSize: 10,
  l2decay: 0.001
});

for ( let epoch = 0; epoch < EPOCHS_LIMIT; epoch++ ) {
  const batch = _.shuffle ( TRAIN_SET );
  for ( let i = 0, l = Math.min ( TRAIN_LIMIT, batch.length ) - 1; i < l; i += 1 ) {
    if ( i % 500 === 0 ) console.log ( i );
    const sample = batch[i];
    trainer.train ( sample.input, sample.output );
  }
}

/* TEST */

let pass = 0;
let fail = 0;

for ( let i = 0, l = Math.min ( TEST_LIMIT, TEST_SET.length ); i < l; i++ ) {
  const sample = TEST_SET[i];
  const actual = argmax ( nn.forward ( sample.input, false ).w );
  const expected = sample.output;
  if ( expected === actual ) {
    pass += 1;
  } else {
    fail += 1;
  }
}

console.log ( 'Pass:', pass );
console.log ( 'Fail:', fail );
console.log ( 'Success:', ( pass * 100 ) / ( pass + fail ) );

/* EXPORT */

export default nn;
