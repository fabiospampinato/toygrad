
/* IMPORT */

import AbstractActivationElementwise from '~/layers/abstract_activation_elementwise';
import type {SigmoidOptions} from '~/types';

/* MAIN */

class Sigmoid extends AbstractActivationElementwise<SigmoidOptions> {

  /* API */

  activationForward ( x: number ): number {

    return 1 / ( 1 + Math.exp ( -x ) );

  }

  activationBackward ( x: number, dx: number ): number {

    return x * ( 1 - x ) * dx;

  }

}

/* EXPORT */

export default Sigmoid;
