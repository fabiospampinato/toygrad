
/* IMPORT */

import Abstract from '~/trainers/abstract';
import type NeuralNetwork from '~/neural_network';
import type {TrainerSGDOptions} from '~/types';

/* MAIN */

class SGD extends Abstract {

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerSGDOptions ) {

    super ( nn, {
      method: 'sgd',
      ...options
    });

  }

}

/* EXPORT */

export default SGD;
