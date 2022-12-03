
/* IMPORT */

import {map, sum} from './ops';
import type Matrix from './matrix';

/* MAIN */

const leakyrelu = ( x: number, derivative: boolean ): number => {
  if ( derivative ) {
    return ( x < 0 ) ? .01 : 1;
  } else {
    return Math.max ( .1 * x, x );
  }
};

const relu = ( x: number, derivative: boolean ): number => {
  if ( derivative ) {
    return ( x < 0 ) ? 0 : 1;
  } else {
    return ( x < 0 ) ? 0 : x;
  }
};

const sigmoid = ( x: number, derivative: boolean ): number => {
  const fx = 1 / ( 1 + Math.exp ( -x ) );
  if ( derivative ) {
    return fx * ( 1 - fx );
  } else {
    return fx;
  }
};

const softmax = Object.assign (( x: Matrix, derivative: boolean ): Matrix => {
  const exp = map ( x, Math.exp );
  const total = sum ( exp );
  if ( derivative ) {
    return map ( exp, x => ( x / total ) * ( 1 - ( x / total ) ) );
  } else {
    return map ( exp, x => x / total );
  }
}, { multi: true } );

const softplus = ( x: number, derivative: boolean ): number => {
  if ( derivative ) {
    return 1 / ( 1 + Math.exp ( -x ) );
  } else {
    return Math.log ( 1 + Math.exp ( x ) );
  }
};

const tanh = ( x: number, derivative: boolean ): number => {
  const fx = Math.tanh ( x );
  if ( derivative ) {
    return 1 - ( fx ** 2 );
  } else {
    return fx;
  }
};

/* EXPORT */

export {leakyrelu, relu, sigmoid, softmax, softplus, tanh};
