
/* IMPORT */

import Abstract from '~/trainers/abstract';
import type NeuralNetwork from '~/neural_network';
import type {TrainerNesterovOptions} from '~/types';

/* MAIN */

class Nesterov extends Abstract {

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerNesterovOptions ) {

    super ( nn, {
      method: 'nesterov',
      ...options
    });

  }

}

/* EXPORT */

export default Nesterov;
