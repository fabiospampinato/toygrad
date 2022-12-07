# Toygrad

A toy library for building simple neural networks which can be serialized to compact JSON.

This is basically a fork of [@karphathy](https://github.com/karpathy)'s wonderful [ConvNetJS](https://github.com/karpathy/convnetjs), but stripped down a bit, updated for ES6+ and TS, and with better support for saving the model in a compact format, for production use cases.

## Install

```sh
npm install --save toygrad
```

## Usage

Let's build a model that learns the XOR function:

```ts
import fs from 'node:fs';
import {NeuralNetwork, Tensor, Trainers} from 'toygrad';

// Some helper functions that we'll need

const argmax = arr => {
  return arr.indexOf ( Math.max ( ...Array.from ( arr ) ) );
};

const toTensor = arr => {
  return new Tensor ( 1, 1, 2, new Float32Array ( arr ) );
};

// Initializing the structure of the network

const nn = new NeuralNetwork ({
  layers: [
    { type: 'input', sx: 1, sy: 1, sz: 2 },
    { type: 'dense', filters: 4 },
    { type: 'tanh' },
    { type: 'dense', filters: 2 },
    { type: 'softmax' }
  ]
});

// Initializing the trainer algorithm

const trainer = new Trainers.Adadelta ( nn, {
  batchSize: 4
});

// Training for 50_000 epochs

for ( let i = 0, l = 50_000; i < l; i++ ) {
  trainer.train ( toTensor ( [0, 0] ), 0 );
  trainer.train ( toTensor ( [1, 0] ), 1 );
  trainer.train ( toTensor ( [0, 1] ), 1 );
  trainer.train ( toTensor ( [1, 1] ), 0 );
}

// Inferring

console.log ( argmax ( nn.forward ( toTensor ( [0, 0] ), false ).w ) ); // => 0
console.log ( argmax ( nn.forward ( toTensor ( [1, 0] ), false ).w ) ); // => 1
console.log ( argmax ( nn.forward ( toTensor ( [0, 1] ), false ).w ) ); // => 1
console.log ( argmax ( nn.forward ( toTensor ( [1, 1] ), false ).w ) ); // => 0

// Exporting as options, with 16 bits of precision per weight, for saving the neural network

const options = nn.getAsOptions ( 'f16' );
const clone = new NeuralNetwork ( options );

// Saving the network to disk, for future usage

fs.writeFileSync ( 'xor.standalone.json', JSON.stringify ( options ) );
```

## Thanks

- Huge thanks to [@liashchynskyi](https://github.com/liashchynskyi), who wrote some very core code used in a previous version of this library, and an amazing [article](https://dev.to/liashchynskyi/creating-of-neural-network-using-javascript-in-7minutes-o21) about building your own neural network from scratch.
- Huge thanks to [@karphathy](https://github.com/karpathy) and the other maintainers of [ConvNetJS](https://github.com/karpathy/convnetjs), who basically wrote all the interesting code in this library.

## License

MIT © Andrej Karpathy
