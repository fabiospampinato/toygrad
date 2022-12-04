
/* IMPORT */

import _ from 'lodash';
import fs from 'node:fs';
import {NeuralNetwork} from '../dist/index.js';

/* MAIN */

// DATASET: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
// DEMO: https://playground.solidjs.com/anonymous/1c71105e-dd96-483d-887c-023bab2649a4

const nn = new NeuralNetwork ({
  learningRate: .05,
  layers: [
    {
      inputs: 28 * 28,
      outputs: 20,
      activation: 'leakyrelu'
    },
    {
      inputs: 20,
      outputs: 10,
      activation: 'softmax'
    }
  ]
});

/* TRAIN */

const parseCSV = csv => {
  const values = csv.split ( ',' ).map ( Number );
  const output = new Array ( 10 ).fill ( 0 ).map ( ( _, i ) => ( i === values[0] ) ? 1 : 0 );
  const input = values.slice ( 1 ).map ( x => x / 255 );
  return {input, output};
};

const TRAIN_SET = fs.readFileSync ( './examples/mnist_train.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );
const TEST_SET = fs.readFileSync ( './examples/mnist_test.csv', 'utf8' ).split ( '\n' ).slice ( 1, -1 ).map ( parseCSV );

nn.trainLoop ( 5, () => {
  const batch = _.shuffle ( TRAIN_SET );
  for ( let i = 0, l = batch.length - 1; i < l; i += 1 ) {
    const slice =  batch.slice ( i, i + 1 );
    const inputs = slice.map ( x => x.input );
    const outputs = slice.map ( x => x.output );
    debugger;
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
