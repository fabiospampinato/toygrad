# Toygrad

A toy library for building simple feed-forward neural networks which can be exported to standalone compact JS functions.

## Install

```sh
npm install --save toygrad
```

## Usage

Let's build a model that learns the XOR function:

```ts
import fs from 'node:fs';
import {NeuralNetwork} from 'toygrad';

// Initializing the structure of the network

const nn = new NeuralNetwork ({
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

// Training for 50_000 epochs, with some logging

nn.trainLoop ( 50_000, () => {
  const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const outputs = [[0], [1], [1], [0]];
  nn.trainMultiple ( inputs, outputs );
});

// Inferring

console.log ( nn.infer ( [0, 0] )[0] ); // => 0.03047387662095156
console.log ( nn.infer ( [1, 0] )[0] ); // => 0.9748001385066861
console.log ( nn.infer ( [0, 1] )[0] ); // => 0.9730739075567443
console.log ( nn.infer ( [1, 1] )[0] ); // => 0.02092264031622114

// Exporting as options, for cloning the neural network

const opts = nn.exportAsOptions ();
const clone = new NeuralNetwork ( opts );

// Exporting as a standalone function, for production

const xor = nn.exportAsFunction ();

console.log ( xor ( [0, 0] )[0] ); // => 0.03047387662095156
console.log ( xor ( [1, 0] )[0] ); // => 0.9748001385066861
console.log ( xor ( [0, 1] )[0] ); // => 0.9730739075567443
console.log ( xor ( [1, 1] )[0] ); // => 0.02092264031622114

// Saving the standalone function to disk

fs.writeFileSync ( 'xor.standalone.js', xor.toString () );
```

## Thanks

- Huge thanks to [@liashchynskyi](https://github.com/liashchynskyi), who actually wrote some very core code used in this library, and an amazing [article](https://dev.to/liashchynskyi/creating-of-neural-network-using-javascript-in-7minutes-o21) about building your own neural network from scratch.

## License

MIT © Fabio Spampinato
