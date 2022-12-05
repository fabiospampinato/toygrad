
/* IMPORT */

import Matrix from './matrix';
import type {Precision} from './types';

/* MAIN */

const encode = ( matrix: Matrix, precision: Precision ): string => {

  const uint8 = new Uint8Array ( matrix.buffer.buffer );

  let encoded = `${precision}|${matrix.rows}|${matrix.cols}|`;
  let bytes = '';

  if ( precision === 'f32' ) {

    for ( let i = 0, l = uint8.length; i < l; i += 1 ) {
      bytes += String.fromCharCode ( uint8[i] );
    }

  } else if ( precision === 'f16' ) {

    for ( let i = 0, l = uint8.length; i < l; i += 4 ) {
      bytes += String.fromCharCode ( uint8[i + 2] );
      bytes += String.fromCharCode ( uint8[i + 3] );
    }

  } else {

    throw new Error ( 'Unsupported precision' );

  }

  encoded += btoa ( bytes );

  return encoded;

};

const decode = ( encoded: string ): Matrix => {

  const parts =  encoded.split ( '|' );
  const precision = parts[0];
  const rows = Number ( parts[1] );
  const cols = Number ( parts[2] );
  const bytes = atob ( parts[3] );

  const matrix = new Matrix ( rows, cols );
  const uint8 = new Uint8Array ( matrix.buffer.buffer );

  if ( precision === 'f32' ) {

    for ( let s = 0, i = 0, l = bytes.length; i < l; s += 4, i += 4 ) {
      uint8[s + 0] = bytes[i + 0].charCodeAt ( 0 );
      uint8[s + 1] = bytes[i + 1].charCodeAt ( 0 );
      uint8[s + 2] = bytes[i + 2].charCodeAt ( 0 );
      uint8[s + 3] = bytes[i + 3].charCodeAt ( 0 );
    }

  } else if ( precision === 'f16' ) {

    for ( let s = 0, i = 0, l = bytes.length; i < l; s += 4, i += 2 ) {
      uint8[s + 0] = 0;
      uint8[s + 1] = 0;
      uint8[s + 2] = bytes[i + 0].charCodeAt ( 0 );
      uint8[s + 3] = bytes[i + 1].charCodeAt ( 0 );
    }

  } else {

    throw new Error ( 'Unsupported precision' );

  }

  return matrix;

};

/* EXPORT */

export {encode, decode};
