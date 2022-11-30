
/* IMPORT */

import * as Activations from './activations';
import {abs, add, map, mean, multiply, product, random, scale, subtract, transpose} from './ops';
import type {ActivationFN, Matrix, Vector, Options} from './types';

/* MAIN */

class NeuralNetwork {

  /* VARIABLES */

  options: Options;
  activations: ActivationFN[];
  weights: Matrix[];

  /* CONSTRUCTOR */

  constructor ( options: Options ) {

    if ( options.layers.length < 2 ) throw new Error ( 'You need at least 1 input layer and 1 output layer' );

    this.options = options;
    this.activations = options.layers.map ( layer => Activations[layer.activation] );
    this.weights = options.layers.slice ( 0, -1 ).map ( ( layer, i ) => random ( layer.neurons, options.layers[i + 1].neurons, -1, 1 ) );

  }

  /* API */

  trainSingle ( input: Vector, target: Vector, _log: boolean = false ): void {

    /* FORWARD PASS */

    const layersLogits: Matrix[] = [];
    const layersActivated: Matrix[] = [];

    let layer = [input];
    for ( let i = 0, l = this.weights.length; i < l; i++ ) {
      const logits = product ( layer, this.weights[i] );
      const activated = layer = map ( logits, x => this.activations[i + 1]( x, false ) );
      layersLogits.push ( logits );
      layersActivated.push ( activated );
    }

    // let layer = [input];
    // const input_layer = [input];
    // const hidden_layer_logits = product(input_layer, this.weights0);
    // const hidden_layer_activated = map(hidden_layer_logits,v => this.activation(v, false));
    // const output_layer_logits = product(hidden_layer_activated, this.weights1);
    // const output_layer_activated = map(output_layer_logits,v => this.activation(v, false));

    /* BACKWARD PASS */

    const layersErrors: Matrix[] = [];
    const layersGradients: Matrix[] = [];

    layer = [target];
    for ( let i = this.weights.length - 1; i >= 0; i-- ) {
      const fn = ( i === this.weights.length - 1 ) ? subtract : product;
      const error = fn ( layer, ( i === this.weights.length - 1 ) ? layersActivated[i] : transpose ( this.weights[i + 1] ) );
      const gradient = layer = multiply ( error, map ( layersLogits[i], x => this.activations[i + 1]( x, true ) ) );
      layersErrors.unshift ( error );
      layersGradients.unshift ( gradient );
    }

    for ( let i = this.weights.length - 1; i >= 0; i-- ) {
      this.weights[i] = add ( this.weights[i], product ( transpose ( layersActivated[i - 1] || [input] ), scale ( layersGradients[i], this.options.learningRate ) ) );
    }

    // const output_error = subtract(target, output_layer_activated);
    // const output_delta = multiply(output_error, map(output_layer_logits,v => this.activation(v, true)));
    // const hidden_error = product(output_delta, transpose(this.weights1));
    // const hidden_delta = multiply(hidden_error, map(hidden_layer_logits,v => this.activation(v, true)));

    // this.weights1 = add(this.weights1, product(transpose(hidden_layer_activated), scale(output_delta, this.options.learningRate)));
    // this.weights0 = add(this.weights0, product(transpose(input_layer), scale(hidden_delta, this.options.learningRate)));

    /* LOGGING */

    if ( _log ) {
      console.log ( `Mean error: ${mean ( abs ( layersErrors[layersErrors.length - 1] ) )}`);
    }

  }

  predict ( input: Vector ): Vector {

    let layer = [input];
    for ( let i = 0, l = this.weights.length; i < l; i++ ) {
      const logits = product ( layer, this.weights[i] );
      const activated = layer = map ( logits, x => this.activations[i + 1]( x, false ) );
    }
    return layer[0];

    // const input_layer = [input];
    // const hidden_layer = map(product(input_layer, this.weights0),v => this.activation(v, false));
    // const output_layer = map(product(hidden_layer, this.weights1),v => this.activation(v, false));
    // return output_layer[0] || [];
  }

  // predict(input) {
  //   // let input_layer = input;
  //   // let hidden_layer = product(input_layer, this.weights0).map(v => this.activation(v, false));
  //   // let output_layer = product(hidden_layer, this.weights1).map(v => this.activation(v, false));

  //   const hiddenlayer2: number[] = [];
  //   for ( let ci = 0, cl = this.weights0._data.length; ci < cl; ci++ ) {
  //     for ( let i = 0, l = this.weights0._data[ci].length; i < l; i++ ) {
  //       hiddenlayer2[i] ||= 0;
  //       hiddenlayer2[i] += this.weights0._data[ci][i] * input._data[0][ci];
  //     }
  //   }
  //   const hiddenlayer3 = hiddenlayer2.map(v => this.activation(v, false));

  //   const outputlayer2: number[] = [];
  //   for ( let ci = 0, cl = this.weights1._data.length; ci < cl; ci++ ) {
  //     for ( let i = 0, l = this.weights1._data[ci].length; i < l; i++ ) {
  //       outputlayer2[i] ||= 0;
  //       outputlayer2[i] += this.weights1._data[ci][i] * hiddenlayer3[ci];
  //     }
  //   }
  //   const outputlayer3 = outputlayer2.map(v => this.activation(v, false));

  //   return outputlayer3;

  // }

}

class _NeuralNetwork {

  /* VARIABLES */

  options: Options;
  activation;
  weights0;
  weights1;

  /* CONSTRUCTOR */

  constructor ( options: Options ) {
    this.options = options;
    this.activation = Activations[options.activation];
    this.weights0 = random ( options.inputLayer, options.hiddenLayer, -1.0, 1.0 );
    this.weights1 = random ( options.hiddenLayer, options.outputLayer, -1.0, 1.0 );
  }

  /* API */

  trainSingle ( input: Vector, target: Vector ) {
    target = [target]
    const input_layer = [input];
    const hidden_layer_logits = product(input_layer, this.weights0);
    const hidden_layer_activated = map(hidden_layer_logits,v => this.activation(v, false));
    const output_layer_logits = product(hidden_layer_activated, this.weights1);
    const output_layer_activated = map(output_layer_logits,v => this.activation(v, false));

    const output_error = subtract(target, output_layer_activated);
    const output_delta = multiply(output_error, map(output_layer_logits,v => this.activation(v, true)));
    const hidden_error = product(output_delta, transpose(this.weights1));
    const hidden_delta = multiply(hidden_error, map(hidden_layer_logits,v => this.activation(v, true)));

    this.weights1 = add(this.weights1, product(transpose(hidden_layer_activated), scale(output_delta, this.options.learningRate)));
    this.weights0 = add(this.weights0, product(transpose(input_layer), scale(hidden_delta, this.options.learningRate)));

    // if (i % 10000 == 0) {
    //   console.log(`Error: ${mean(abs(output_error))}`);
    // }
  }

  predict ( input: Vector ): Vector {
    const input_layer = [input];
    const hidden_layer = map(product(input_layer, this.weights0),v => this.activation(v, false));
    const output_layer = map(product(hidden_layer, this.weights1),v => this.activation(v, false));
    return output_layer[0] || [];
  }

  // predict(input) {
  //   // let input_layer = input;
  //   // let hidden_layer = product(input_layer, this.weights0).map(v => this.activation(v, false));
  //   // let output_layer = product(hidden_layer, this.weights1).map(v => this.activation(v, false));

  //   const hiddenlayer2: number[] = [];
  //   for ( let ci = 0, cl = this.weights0._data.length; ci < cl; ci++ ) {
  //     for ( let i = 0, l = this.weights0._data[ci].length; i < l; i++ ) {
  //       hiddenlayer2[i] ||= 0;
  //       hiddenlayer2[i] += this.weights0._data[ci][i] * input._data[0][ci];
  //     }
  //   }
  //   const hiddenlayer3 = hiddenlayer2.map(v => this.activation(v, false));

  //   const outputlayer2: number[] = [];
  //   for ( let ci = 0, cl = this.weights1._data.length; ci < cl; ci++ ) {
  //     for ( let i = 0, l = this.weights1._data[ci].length; i < l; i++ ) {
  //       outputlayer2[i] ||= 0;
  //       outputlayer2[i] += this.weights1._data[ci][i] * hiddenlayer3[ci];
  //     }
  //   }
  //   const outputlayer3 = outputlayer2.map(v => this.activation(v, false));

  //   return outputlayer3;

  // }

}

/* EXPORT */

export default NeuralNetwork;
