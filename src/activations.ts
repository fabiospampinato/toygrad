
/* MAIN */

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

const softplus = ( x: number, derivative: boolean ): number => {
  if ( derivative ) {
    return 1 / ( 1 + Math.exp ( -x ) );
  } else {
    return Math.log ( 1 + Math.exp ( x ) );
  }
};

const tanh = ( x: number, derivative: boolean ): number => {
  const fx = 2 / ( 1 + Math.exp ( -2 * x ) ) - 1;
  if ( derivative ) {
    return 1 - ( fx ** 2 );
  } else {
    return fx;
  }
};

/* EXPORT */

export {relu, sigmoid, softplus, tanh};
