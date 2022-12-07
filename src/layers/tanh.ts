
/* IMPORT */

import AbstractActivationElementwise from '~/layers/abstract_activation_elementwise';
import type {TanhOptions} from '~/types';

/* MAIN */

class Tanh extends AbstractActivationElementwise<TanhOptions> {

  /* API */

  activationForward ( x: number ): number {

    return Math.tanh ( x );

  }

  activationBackward ( x: number, dx: number ): number {

    return ( 1 - ( x ** 2 ) ) * dx;

  }

}

/* EXPORT */

export default Tanh;
