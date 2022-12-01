
/* IMPORT */

import * as Activations from './activations';
import {abs, add, clone, map, mean, multiply, product, random, scale, subtract, transpose} from './ops';
import {isString} from './utils';
import {encode, decode} from './weights';
import type {Identity, Matrix, Precision, Vector, Options, ResultForward, ResultBackward, ResultTrain} from './types';

/* MAIN */

class NeuralNetwork {

  /* VARIABLES */

  options: Options;
  precision: Precision;

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
    this.precision = options.precision || 'float32';

    this.activation0 = x => activation0 ( x, false );
    this.activation1 = x => activation1 ( x, false );
    this.activation0d = x => activation0 ( x, true );
    this.activation1d = x => activation1 ( x, true );

    this.weights0 = weights0 ? ( isString ( weights0 ) ? decode ( weights0, this.precision ): weights0 ) : random ( layer0.inputs, layer0.outputs, -1, 1 );
    this.weights1 = weights1 ? ( isString ( weights1 ) ? decode ( weights1, this.precision ): weights1 ) : random ( layer1.inputs, layer1.outputs, -1, 1 );

  }

  /* TRAINING/INFERENCE API */

  forward ( inputs: Vector[] ): ResultForward {

    const weighted0 = product ( inputs, this.weights0 );
    const activated0 = map ( weighted0, this.activation0 );
    const weighted1 = product ( activated0, this.weights1 );
    const activated1 = map ( weighted1, this.activation1 );
    const result: ResultForward = [weighted0, weighted1, activated0, activated1];

    return result;

  }

  backward ( inputs: Vector[], outputs: Vector[], forward: Matrix[] ): ResultBackward {

    const [weighted0, weighted1, activated0, activated1] = forward;

    const error1 = subtract ( outputs, activated1 );
    const gradient1 = multiply ( error1, map ( weighted1, this.activation0d ) );
    const error0 = product ( gradient1, transpose ( this.weights1 ) );
    const gradient0 = multiply ( error0, map ( weighted0, this.activation1d ) );
    const result: ResultBackward = [error0, error1, gradient0, gradient1];

    this.weights1 = add ( this.weights1, product ( transpose ( activated0 ), scale ( gradient1, this.options.learningRate ) ) );
    this.weights0 = add ( this.weights0, product ( transpose ( inputs ), scale ( gradient0, this.options.learningRate ) ) );

    return result;

  }

  trainSingle ( input: Vector, output: Vector ): ResultTrain {

    return this.trainMultiple ( [input], [output] );

  }

  trainMultiple ( inputs: Vector[], outputs: Vector[] ): ResultTrain {

    const forward = this.forward ( inputs );
    const backward = this.backward ( inputs, outputs, forward );

    return [forward, backward];

  }

  trainLoop ( iterations: number, train: ( i: number ) => ResultTrain | void ): void {

    const logEnabled = true;
    const logStep = Math.floor ( iterations / 1000 );

    for ( let i = 0, s = 0; i < iterations; i++ ) {

      const result = train ( i );

      if ( logEnabled && result && ( i % logStep ) === 0 ) {

        const percentage = ( ++s / 10 ).toFixed ( 1 );
        const error = mean ( abs ( result[1][1] ) );

        console.log ( `${percentage}% -`, error );

      }

    }

  }

  infer ( input: Vector ): Vector {

    return this.forward ( [input] )[3][0];

  }

  /* EXPORT API */

  exportAsFunction (): (( input: Vector ) => Vector) {

    const fn = [
      `(function _ ( input ) {` +
        `if ( !_._ ) {` +
          `_._ = true;` +
          `_.p = ${product.toString ()};` +
          `_.m = ${map.toString ()};` +
          `_.d = ${decode.toString ()};` +
          `_.a0 = ${Activations[this.options.layers[0].activation].toString ()};` +
          `_.a1 = ${Activations[this.options.layers[1].activation].toString ()};` +
          `_.w0 = _.d ( '${encode ( this.weights0, this.precision )}' );` +
          `_.w1 = _.d ( '${encode ( this.weights1, this.precision )}' );` +
        `}` +
        `return _.m ( _.p ( _.m ( _.p ( [input], _.w0 ), _.a0 ), _.w1 ), _.a1 )[0];` +
      `})`
    ];

    return eval ( fn.join ( '' ) );

  }

  exportAsOptions (): Options {

    return {
      layers: [
        {
          inputs: this.options.layers[0].inputs,
          outputs: this.options.layers[0].outputs,
          activation: this.options.layers[0].activation,
          weights: clone ( this.weights0 )
        },
        {
          inputs: this.options.layers[1].inputs,
          outputs: this.options.layers[1].outputs,
          activation: this.options.layers[1].activation,
          weights: clone ( this.weights1 )
        }
      ],
      learningRate: this.options.learningRate,
      precision: this.options.precision
    };

  }

}

/* EXPORT */

export default NeuralNetwork;
