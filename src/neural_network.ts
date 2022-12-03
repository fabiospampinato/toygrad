
/* IMPORT */

import * as Activations from './activations';
import {encode, decode} from './encoder';
import {abs, add, from, map, map2, mean, mse, multiply, product, random, scale, subtract, transpose} from './ops';
import {fusedAddProductScale, fusedAddScale, fusedMultiplyMapActivation, fusedProductBiased} from './ops';
import Matrix from './matrix';
import type {Identity, Vector, Options, ResultForward, ResultBackward, ResultTrain} from './types';

/* MAIN */

//TODO: Generalize this to an arbitrary number of layers
//TODO: Support a softmax layer
//TODO: Support custom activation functions

class NeuralNetwork {

  /* VARIABLES */

  options: Options;

  activation0: Identity<number>;
  activation1: Identity<number>;
  activation0d: Identity<number>;
  activation1d: Identity<number>;

  biases0: Matrix;
  biases1: Matrix;
  weights0: Matrix;
  weights1: Matrix;

  /* CONSTRUCTOR */

  constructor ( options: Options ) {

    if ( options.layers.length !== 2 ) throw new Error ( 'Only a fixed 2 layers of weights are supported for now, sorry' );

    if ( !options.layers.slice ( 0, 1 ).every ( ( layer, i ) => layer.outputs === options.layers[i + 1].inputs ) ) throw new Error ( 'The number of outputs of a layer must match the number of inputs of the next layer' );

    const layer0 = options.layers[0];
    const layer1 = options.layers[1];
    const biases0 = layer0.biases;
    const biases1 = layer1.biases;
    const weights0 = layer0.weights;
    const weights1 = layer1.weights;
    const activation0 = Activations[layer0.activation];
    const activation1 = Activations[layer1.activation];

    this.options = options;

    this.activation0 = x => activation0 ( x, false );
    this.activation1 = x => activation1 ( x, false );
    this.activation0d = x => activation0 ( x, true );
    this.activation1d = x => activation1 ( x, true );

    this.biases0 = biases0 ? decode ( biases0 ) : new Matrix ( 1, layer0.outputs );
    this.biases1 = biases1 ? decode ( biases1 ) : new Matrix ( 1, layer1.outputs );
    this.weights0 = weights0 ? decode ( weights0 ) : random ( layer0.inputs, layer0.outputs, -1, 1 );
    this.weights1 = weights1 ? decode ( weights1 ) : random ( layer1.inputs, layer1.outputs, -1, 1 );

  }

  /* TRAINING/INFERENCE API */

  forward ( inputs: Vector[] ): ResultForward {

    // const weighted0 = add ( product ( from ( inputs ), this.weights0 ), this.biases0 );
    const weighted0 = fusedProductBiased ( from ( inputs ), this.weights0, this.biases0 );
    const activated0 = map ( weighted0, this.activation0 );
    // const weighted1 = add ( product ( activated0, this.weights1 ), this.biases1 );
    const weighted1 = fusedProductBiased ( activated0, this.weights1, this.biases1 );
    const activated1 = map ( weighted1, this.activation1 );
    const result: ResultForward = [weighted0, weighted1, activated0, activated1];

    return result;

  }

  backward ( inputs: Vector[], outputs: Vector[], forward: Matrix[] ): ResultBackward {

    const {learningRate} = this.options;
    const [weighted0, weighted1, activated0, activated1] = forward;

    const error1 = subtract ( from ( outputs ), activated1 );
    // const gradient1 = multiply ( error1, map ( activated1, this.activation1d ) );
    const gradient1 = fusedMultiplyMapActivation ( error1, activated1, this.activation1d );
    const error0 = product ( gradient1, transpose ( this.weights1 ) );
    // const gradient0 = multiply ( error0, map ( activated0, this.activation0d ) );
    const gradient0 = fusedMultiplyMapActivation ( error0, activated0, this.activation0d );
    const result: ResultBackward = [error0, error1, gradient0, gradient1];

    // this.biases1 = add ( this.biases1, scale ( gradient1, learningRate / 2 ) );
    this.biases1 = fusedAddScale ( this.biases1, gradient1, learningRate / 2 );
    // this.biases0 = add ( this.biases0, scale ( gradient0, learningRate / 2 ) );
    this.biases0 = fusedAddScale ( this.biases0, gradient0, learningRate / 2 );
    // this.weights1 = add ( this.weights1, product ( transpose ( activated0 ), scale ( gradient1, learningRate ) ) );
    this.weights1 = fusedAddProductScale ( this.weights1, transpose ( activated0 ), gradient1, learningRate );
    // this.weights0 = add ( this.weights0, product ( transpose ( from ( inputs ) ), scale ( gradient0, learningRate ) ) );
    this.weights0 = fusedAddProductScale ( this.weights0, transpose ( from ( inputs ) ), gradient0, learningRate );

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
        const error = mse ( result[2][3], from ( result[1] ) );

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
          `_.f = ${from.toString ()};` +
          `_.m = ${map.toString ()};` +
          `_.pb = ${fusedProductBiased.toString ()};` +
          `_.d = ${decode.toString ()};` +
          `_.a0 = ${Activations[this.options.layers[0].activation].toString ()};` +
          `_.a1 = ${Activations[this.options.layers[1].activation].toString ()};` +
          `_.b0 = _.d ( '${encode ( this.biases0 )}' );` +
          `_.b1 = _.d ( '${encode ( this.biases1 )}' );` +
          `_.w0 = _.d ( '${encode ( this.weights0 )}' );` +
          `_.w1 = _.d ( '${encode ( this.weights1 )}' );` +
        `}` +
        `return Array.from ( _.m ( _.pb ( _.m ( _.pb ( _.f ( [input] ), _.w0, _.b0 ), _.a0 ), _.w1, _.b1 ), _.a1 ).buffer );` +
      `})`
    ];

    return eval.call ( undefined, fn.join ( '' ) );

  }

  exportAsGraphviz (): string {

    const layer0 = this.options.layers[0];
    const layer1 = this.options.layers[1];

    const lines: string[] = [];

    lines.push ( 'digraph {' );
    lines.push ( 'rankdir=LR' );
    lines.push ( 'splines=line' );

    lines.push ( 'subgraph cluster_input {' );
    lines.push ( 'label="Input"' );
    for ( let i = 0; i < layer0.inputs; i++ ) {
      lines.push ( `"i${i}" [label="" shape="circle"]` );
    }
    lines.push ( '}' );

    lines.push ( 'subgraph cluster_hidden {' );
    lines.push ( 'label="Hidden"' );
    for ( let h = 0; h < layer0.outputs; h++ ) {
      lines.push ( `"h${h}" [label="" shape="circle"]` );
    }
    lines.push ( '}' );

    lines.push ( 'subgraph cluster_output {' );
    lines.push ( 'label="Output"' );
    for ( let o = 0; o < layer1.outputs; o++ ) {
      lines.push ( `"o${o}" [label="" shape="circle"]` );
    }
    lines.push ( '}' );

    for ( let i = 0; i < layer0.inputs; i++ ) {
      for ( let h = 0; h < layer0.outputs; h++ ) {
        lines.push ( `"i${i}" -> "h${h}"` );
      }
    }

    for ( let h = 0; h < layer0.outputs; h++ ) {
      for ( let o = 0; o < layer1.outputs; o++ ) {
        lines.push ( `"h${h}" -> "o${o}"` );
      }
    }

    lines.push ( '}' );

    return lines.join ( '\n' );

  }

  exportAsOptions (): Options {

    return {
      layers: [
        {
          inputs: this.options.layers[0].inputs,
          outputs: this.options.layers[0].outputs,
          activation: this.options.layers[0].activation,
          biases: encode ( this.biases0 ),
          weights: encode ( this.weights0 )
        },
        {
          inputs: this.options.layers[1].inputs,
          outputs: this.options.layers[1].outputs,
          activation: this.options.layers[1].activation,
          biases: encode ( this.biases1 ),
          weights: encode ( this.weights1 )
        }
      ],
      learningRate: this.options.learningRate
    };

  }

}

/* EXPORT */

export default NeuralNetwork;
