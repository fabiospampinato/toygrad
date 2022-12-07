
/* MAIN */

const isBuffer = ( value: unknown ): value is Float32Array => {

  return ( value instanceof Float32Array );

};

const isNumber = ( value: unknown ): value is number => {

  return ( typeof value === 'number' );

};

const isString = ( value: unknown ): value is string => {

  return ( typeof value === 'string' );

};

const isUndefined = ( value: unknown ): value is undefined => {

  return ( typeof value === 'undefined' );

};

const randn = ( mean: number, stdev: number ): number => {

  //URL: https://stackoverflow.com/a/36481059/1420197

  const u = 1 - Math.random ();
  const v = Math.random ();
  const z = Math.sqrt ( -2 * Math.log ( u ) ) * Math.cos ( 2 * Math.PI * v );
  const rand = ( z * stdev + mean );

  return rand;

};

const range = ( start: number, end: number ): number[] => {

  const length = end - start;

  return Array.from ( {length}, ( _, i ) => i + start );

};

/* EXPORT */

export {isBuffer, isNumber, isString, isUndefined, randn, range};
