
/* IMPORT */

import AbstractActivationElementwise from '~/layers/abstract_activation_elementwise';
import type {LeakyReluOptions} from '~/types';

/* MAIN */

class LeakyRelu extends AbstractActivationElementwise<LeakyReluOptions> {

  /* API */

  activationForward ( x: number ): number {

    return Math.max ( .01 * x, x );

  }

  activationBackward ( x: number, dx: number ): number {

    return ( x <= 0 ) ? 0 : dx;

  }

}

/* EXPORT */

export default LeakyRelu;
