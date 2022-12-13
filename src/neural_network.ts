
/* IMPORT */

import Layers from '~/layers';
import Abstract from '~/layers/abstract';
import AbstractInput from '~/layers/abstract_input';
import AbstractHidden from '~/layers/abstract_hidden';
import AbstractOutput from '~/layers/abstract_output';
import type Tensor from '~/tensor';
import type {LayersAbstract, NeuralNetworkLayers, NeuralNetworkLayersDescriptions, NeuralNetworkLayersResolved, NeuralNetworkOptions, ParamsAndGrads, Precision} from '~/types';

/* MAIN */

class NeuralNetwork {

  /* VARIABLES */

  layers: NeuralNetworkLayersResolved;
  options: NeuralNetworkOptions;

  /* CONSTRUCTOR */

  constructor ( options: NeuralNetworkOptions ) {

    this.layers = this.resolve ( options.layers ) as any; //TSC
    this.options = options;

    if ( this.layers.length < 2 ) throw new Error ( 'At least one input layer and one output layer are required' );

    if ( !this.layers.slice ( 0, 1 ).every ( layer => layer instanceof AbstractInput ) ) throw new Error ( 'The first layer must be an input layer' );

    if ( !this.layers.slice ( -1 ).every ( layer => layer instanceof AbstractOutput ) ) throw new Error ( 'The last layer must be an output layer' );

    if ( !this.layers.slice ( 1, -1 ).every ( layer => layer instanceof AbstractHidden ) ) throw new Error ( 'The layers between the first and the last must be hidden layers' );

  }

  /* API */

  cost ( input: Tensor, output: Tensor ): number {

    this.forward ( input, false );

    return this.layers[this.layers.length - 1].backward ( output );

  }

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    let result = this.layers[0].forward ( input, isTraining );

    for ( let i = 1, l = this.layers.length; i < l; i++ ) {

      result = this.layers[i].forward ( result, isTraining );

    }

    return result;

  }

  backward ( output: Tensor ): number {

    const loss: number = this.layers[this.layers.length - 1].backward ( output );

    for ( let i = this.layers.length - 2; i >= 1; i-- ) {

      this.layers[i].backward ( output );

    }

    return loss;

  }

  resolve ( layers: NeuralNetworkLayersDescriptions ): Abstract[] {

    const resolveLayer = ( layer: NeuralNetworkLayers | LayersAbstract, prev?: Abstract ) => {

      if ( layer instanceof Abstract ) return layer;

      switch ( layer.type ) {
        /* INPUT */
        case 'input': return new Layers.Input ( layer, prev );
        /* HIDDEN */
        case 'conv': return new Layers.Conv ( layer, prev );
        case 'dense': return new Layers.Dense ( layer, prev );
        case 'dropout': return new Layers.Dropout ( layer, prev );
        case 'pool': return new Layers.Pool ( layer, prev );
        /* ACTIVATION */
        case 'leakyrelu': return new Layers.LeakyRelu ( layer, prev );
        case 'maxout': return new Layers.Maxout ( layer, prev );
        case 'relu': return new Layers.Relu ( layer, prev );
        case 'sigmoid': return new Layers.Sigmoid ( layer, prev );
        case 'tanh': return new Layers.Tanh ( layer, prev );
        /* OUTPUT */
        case 'regression': return new Layers.Regression ( layer, prev );
        case 'softmax': return new Layers.Softmax ( layer, prev );
        /* DEFAULT */
        default: throw new Error ( 'Unknown layer type' );
      }

    };

    let prev: Abstract | undefined;

    return layers.map ( layer => {

      return prev = resolveLayer ( layer, prev );

    });

  }

  getAsOptions ( precision: Precision = 'f32' ): NeuralNetworkOptions {

    return {
      layers: this.layers.map ( layer => layer.getAsOptions ( precision ) )
    } as NeuralNetworkOptions; //TSC

  }

  getParamsAndGrads (): ParamsAndGrads[] {

    return this.layers.flatMap ( layer => layer.getParamsAndGrads () );

  }

}

/* EXPORT */

export default NeuralNetwork;
