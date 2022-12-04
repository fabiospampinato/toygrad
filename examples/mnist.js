
/* IMPORT */

import _ from 'lodash';
import fs from 'node:fs';
import {from, map} from '../dist/ops.js';
import Matrix from '../dist/matrix.js';
import {NeuralNetwork} from '../dist/index.js';

/* HELPERS */

const SIZE = 28;
const CSIZE = 28;

/* MAIN */

// DATASET: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
// DEMO: https://playground.solidjs.com/anonymous/0a49c7e6-224b-4e00-9d98-bd121ba1cd34

const nn = new NeuralNetwork ({
  learningRate: .1,
  layers: [
    {
      inputs: CSIZE * CSIZE,
      outputs: 64,
      activation: 'leakyrelu'
    },
    {
      inputs: 64,
      outputs: 10,
      activation: 'softmax'
    }
  ]
});

/* TRAIN */

const crop = input => {
  if ( SIZE === CSIZE ) return input;
  const offsetRows = Math.round ( Math.random () * ( SIZE - CSIZE ) );
  const offsetCols = Math.round ( Math.random () * ( SIZE - CSIZE ) );
  const matrixInput = from ( _.chunk ( input, SIZE ) );
  const matrixBase = new Matrix ( CSIZE, CSIZE );
  const matrixPopulated = map ( matrixBase, ( _, row, col ) => matrixInput.get ( row + offsetRows, col + offsetCols ) );
  return Array.from ( matrixPopulated.buffer );
};

const parseCSV = csv => {
  const values = csv.split ( ',' ).map ( Number );
  const output = new Array ( 10 ).fill ( 0 ).map ( ( _, i ) => ( i === values[0] ) ? 1 : 0 );
  const input = crop ( values.slice ( 1 ).map ( x => x / 255 ) );
  return {input, output};
};

const TRAIN_SET = fs.readFileSync ( './examples/mnist_train.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );
const TEST_SET = fs.readFileSync ( './examples/mnist_test.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );

nn.trainLoop ( 3, () => {
  const batch = _.shuffle ( TRAIN_SET );
  for ( let i = 0, l = batch.length - 1; i < l; i += 1 ) {
    const slice =  batch.slice ( i, i + 1 );
    const inputs = slice.map ( x => x.input );
    const outputs = slice.map ( x => x.output );
    nn.trainMultiple ( inputs, outputs );
  }
});

/* TEST */

let pass = 0;
let fail = 0;

for ( let i = 0, l = TEST_SET.length; i < l; i++ ) {
  const output = nn.infer ( TEST_SET[i].input );
  const actual = output.indexOf ( Math.max ( ...output ) );
  const expected = TEST_SET[i].output.indexOf ( 1 );
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
