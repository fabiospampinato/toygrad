
/* IMPORT */

import Abstract from '~/trainers/abstract';
import type NeuralNetwork from '~/neural_network';
import type {TrainerAdamOptions} from '~/types';

/* MAIN */

class Adam extends Abstract {

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerAdamOptions ) {

    super ( nn, {
      method: 'adam',
      ...options
    });

  }

}

/* EXPORT */

export default Adam;
