
/* IMPORT */

import Abstract from '~/trainers/abstract';
import type NeuralNetwork from '~/neural_network';
import type {TrainerAdagradOptions} from '~/types';

/* MAIN */

class Adagrad extends Abstract {

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerAdagradOptions ) {

    super ( nn, {
      method: 'adagrad',
      ...options
    });

  }

}

/* EXPORT */

export default Adagrad;
