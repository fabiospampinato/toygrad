
/* IMPORT */

import {NeuralNetwork, Tensor, Trainers} from '../dist/index.js';

/* HELPERS */

const argmax = arr => {
  return arr.indexOf ( Math.max ( ...Array.from ( arr ) ) );
};

const toTensor = arr => {
  return new Tensor ( 1, 1, 2, new Float32Array ( arr ) );
};

/* TRAIN */

const nn = new NeuralNetwork ({
  layers: [
    { type: 'input', sx: 1, sy: 1, sz: 2 },
    { type: 'dense', filters: 4 },
    { type: 'tanh' },
    { type: 'dense', filters: 2 },
    { type: 'softmax' }
  ]
});

const trainer = new Trainers.Adadelta ( nn, {
  batchSize: 4
});

for ( let i = 0, l = 50_000; i < l; i++ ) {
  trainer.train ( toTensor ( [0, 0] ), 0 );
  trainer.train ( toTensor ( [1, 0] ), 1 );
  trainer.train ( toTensor ( [0, 1] ), 1 );
  trainer.train ( toTensor ( [1, 1] ), 0 );
}

/* TEST */

console.log ( '' );
console.log ( '0^0 ->', argmax ( nn.forward ( toTensor ( [0, 0] ), false ).w ) );
console.log ( '1^0 ->', argmax ( nn.forward ( toTensor ( [1, 0] ), false ).w ) );
console.log ( '0^1 ->', argmax ( nn.forward ( toTensor ( [0, 1] ), false ).w ) );
console.log ( '1^1 ->', argmax ( nn.forward ( toTensor ( [1, 1] ), false ).w ) );
console.log ( '' );

/* EXPORT */

export default nn;
