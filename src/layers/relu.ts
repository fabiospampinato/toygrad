
/* IMPORT */

import AbstractActivationElementwise from '~/layers/abstract_activation_elementwise';
import type {ReluOptions} from '~/types';

/* MAIN */

class Relu extends AbstractActivationElementwise<ReluOptions> {

  /* API */

  activationForward ( x: number ): number {

    return ( x <= 0 ) ? 0 : x;

  }

  activationBackward ( x: number, dx: number ): number {

    return ( x <= 0 ) ? 0 : dx;

  }

}

/* EXPORT */

export default Relu;
