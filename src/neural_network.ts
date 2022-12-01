
/* IMPORT */

import * as Activations from './activations';
import {abs, add, map, mean, mse, multiply, product, random, scale, subtract, transpose} from './ops';
import {fusedAddProductScale, fusedMultiplyMapActivation} from './ops';
import {encode, decode} from './weights';
import Matrix from './matrix';
import type {Identity, Vector, Options, ResultForward, ResultBackward, ResultTrain} from './types';

/* MAIN */

//TODO: Generalize this to an arbitrary number of layers

class NeuralNetwork {

  /* VARIABLES */

  options: Options;

  activation0: Identity<number>;
  activation1: Identity<number>;
  activation0d: Identity<number>;
  activation1d: Identity<number>;

  weights0: Matrix;
  weights1: Matrix;

  /* CONSTRUCTOR */

  constructor ( options: Options ) {

    if ( options.layers.length !== 2 ) throw new Error ( 'Only a fixed 2 layers of weights are supported for now, sorry' );
    if ( !options.layers.slice ( 0, 1 ).every ( ( layer, i ) => layer.outputs === options.layers[i + 1].inputs ) ) throw new Error ( 'The number of outputs of a layer must match the number of inputs of the next layer' );

    const layer0 = options.layers[0];
    const layer1 = options.layers[1];
    const weights0 = layer0.weights;
    const weights1 = layer1.weights;
    const activation0 = Activations[layer0.activation];
    const activation1 = Activations[layer1.activation];

    this.options = options;

    this.activation0 = x => activation0 ( x, false );
    this.activation1 = x => activation1 ( x, false );
    this.activation0d = x => activation0 ( x, true );
    this.activation1d = x => activation1 ( x, true );

    this.weights0 = weights0 ? decode ( weights0 ) : random ( layer0.inputs, layer0.outputs, -1, 1 );
    this.weights1 = weights1 ? decode ( weights1 ) : random ( layer1.inputs, layer1.outputs, -1, 1 );

  }

  /* TRAINING/INFERENCE API */

  forward ( inputs: Vector[] ): ResultForward {

    const weighted0 = product ( Matrix.from ( inputs ), this.weights0 );
    const activated0 = map ( weighted0, this.activation0 );
    const weighted1 = product ( activated0, this.weights1 );
    const activated1 = map ( weighted1, this.activation1 );
    const result: ResultForward = [weighted0, weighted1, activated0, activated1];

    return result;

  }

  backward ( inputs: Vector[], outputs: Vector[], forward: Matrix[] ): ResultBackward {

    const [weighted0, weighted1, activated0, activated1] = forward;

    const error1 = subtract ( Matrix.from ( outputs ), activated1 );
    // const gradient1 = multiply ( error1, map ( activated1, this.activation1d ) );
    const gradient1 = fusedMultiplyMapActivation ( error1, activated1, this.activation1d );
    const error0 = product ( gradient1, transpose ( this.weights1 ) );
    // const gradient0 = multiply ( error0, map ( activated0, this.activation0d ) );
    const gradient0 = fusedMultiplyMapActivation ( error0, activated0, this.activation0d );
    const result: ResultBackward = [error0, error1, gradient0, gradient1];

    // this.weights1 = add ( this.weights1, product ( transpose ( activated0 ), scale ( gradient1, this.options.learningRate ) ) );
    this.weights1 = fusedAddProductScale ( this.weights1, transpose ( activated0 ), gradient1, this.options.learningRate );
    // this.weights0 = add ( this.weights0, product ( transpose ( Matrix.from ( inputs ) ), scale ( gradient0, this.options.learningRate ) ) );
    this.weights0 = fusedAddProductScale ( this.weights0, transpose ( Matrix.from ( inputs ) ), gradient0, this.options.learningRate );

    return result;

  }

  trainSingle ( input: Vector, output: Vector ): ResultTrain {

    return this.trainMultiple ( [input], [output] );

  }

  trainMultiple ( inputs: Vector[], outputs: Vector[] ): ResultTrain {

    const forward = this.forward ( inputs );
    const backward = this.backward ( inputs, outputs, forward );

    return [inputs, outputs, forward, backward];

  }

  trainLoop ( iterations: number, train: ( i: number ) => ResultTrain | void ): void {

    const logEnabled = true;
    const logStep = Math.floor ( iterations / 1000 );

    for ( let i = 0, s = 0; i < iterations; i++ ) {

      const result = train ( i );

      if ( logEnabled && result && ( i % logStep ) === 0 ) {

        const percentage = ( ++s / 10 ).toFixed ( 1 );
        const error = mse ( result[2][3], Matrix.from ( result[1] ) );

        console.log ( `${percentage}% -`, error );

      }

    }

  }

  infer ( input: Vector ): Vector {

    return Array.from ( this.forward ( [input] )[3].buffer );

  }

  /* EXPORT API */

  exportAsFunction (): (( input: Vector ) => Vector) {

    const fn = [
      `(function _ ( input ) {` +
        `if ( !_._ ) {` +
          `_._ = true;` +
          `${Matrix.toString ()};` +
          `_.M = Matrix;` +
          `_.p = ${product.toString ()};` +
          `_.m = ${map.toString ()};` +
          `_.d = ${decode.toString ()};` +
          `_.a0 = ${Activations[this.options.layers[0].activation].toString ()};` +
          `_.a1 = ${Activations[this.options.layers[1].activation].toString ()};` +
          `_.w0 = _.d ( '${encode ( this.weights0 )}' );` +
          `_.w1 = _.d ( '${encode ( this.weights1 )}' );` +
        `}` +
        `return Array.from ( _.m ( _.p ( _.m ( _.p ( _.M.from ( [input] ), _.w0 ), _.a0 ), _.w1 ), _.a1 ).buffer );` +
      `})`
    ];

    return eval.call ( undefined, fn.join ( '' ) );

  }

  exportAsOptions (): Options {

    return {
      layers: [
        {
          inputs: this.options.layers[0].inputs,
          outputs: this.options.layers[0].outputs,
          activation: this.options.layers[0].activation,
          weights: encode ( this.weights0 )
        },
        {
          inputs: this.options.layers[1].inputs,
          outputs: this.options.layers[1].outputs,
          activation: this.options.layers[1].activation,
          weights: encode ( this.weights1 )
        }
      ],
      learningRate: this.options.learningRate
    };

  }

}

/* EXPORT */

export default NeuralNetwork;
