
/* IMPORT */

import Matrix from './matrix';
import type {Precision} from './types';

/* MAIN */

//TODO: Make the encoding more efficient, we are throwing away some usable bits by encoding to hex characters

const encode = ( matrix: Matrix, precision: Precision ): string => {

  const uint8 = new Uint8Array ( matrix.buffer.buffer );

  let encoded = `${precision}|${matrix.rows}|${matrix.cols}|`;

  if ( precision === 'f32' ) {

    for ( let i = 0, l = uint8.length; i < l; i += 1 ) {
      encoded += uint8[i].toString ( 16 ).padStart ( 2, '0' );
    }

  } else if ( precision === 'f16' ) {

    for ( let i = 0, l = uint8.length; i < l; i += 4 ) {
      encoded += uint8[i + 2].toString ( 16 ).padStart ( 2, '0' );
      encoded += uint8[i + 3].toString ( 16 ).padStart ( 2, '0' );
    }

  } else {

    throw new Error ( 'Unsupported precision' );

  }

  return encoded;

};

const decode = ( encoded: string ): Matrix => {

  const parts = encoded.split ( '|' );
  const precision = parts[0];
  const rows = Number ( parts[1] );
  const cols = Number ( parts[2] );
  const bytes = parts[3];

  const matrix = new Matrix ( rows, cols );
  const uint8 = new Uint8Array ( matrix.buffer.buffer );

  if ( precision === 'f32' ) {

    for ( let s = 0, i = 0, l = bytes.length; i < l; s += 4, i += 8 ) {
      uint8[s + 0] = parseInt ( bytes.slice ( i, i + 2 ), 16 );
      uint8[s + 1] = parseInt ( bytes.slice ( i + 2, i + 4 ), 16 );
      uint8[s + 2] = parseInt ( bytes.slice ( i + 4, i + 6 ), 16 );
      uint8[s + 3] = parseInt ( bytes.slice ( i + 6, i + 8 ), 16 );
    }

  } else if ( precision === 'f16' ) {

    for ( let i = 0, l = bytes.length; i < l; i += 4 ) {
      uint8[i + 0] = 0;
      uint8[i + 1] = 0;
      uint8[i + 2] = parseInt ( bytes.slice ( i, i + 2 ), 16 );
      uint8[i + 3] = parseInt ( bytes.slice ( i + 2, i + 4 ), 16 );
    }

  } else {

    throw new Error ( 'Unsupported precision' );

  }

  return matrix;

};

/* EXPORT */

export {encode, decode};
