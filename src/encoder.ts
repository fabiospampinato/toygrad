
/* IMPORT */

import Matrix from './matrix';

/* MAIN */

//TODO: Make the encoding more efficient, we are throwing away some usable bits by encoding to hex characters

const encode = ( matrix: Matrix ): string => {

  const uint8 = new Uint8Array ( matrix.buffer.buffer );
  const bytesPerRow = matrix.cols * 4;

  let encoded = '';

  for ( let ri = 0, rl = uint8.length; ri < rl; ri += bytesPerRow ) {
    for ( let i = ri, l = ri + bytesPerRow; i < l; i += 1 ) {
      encoded += uint8[i].toString ( 16 ).padStart ( 2, '0' );
    }
    encoded += '|';
  }

  return encoded.slice ( 0, -1 );

};

const decode = ( encoded: string ): Matrix => {

  const parts = encoded.split ( '|' );
  const rows = parts.length;
  const cols = parts[0].length / 8;
  const matrix = new Matrix ( rows, cols );
  const uint8 = new Uint8Array ( matrix.buffer.buffer );

  for ( let ri = 0, rl = parts.length; ri < rl; ri++ ) {
    for ( let i = 0, l = parts[ri].length; i < l; i += 2 ) {
      const index = ri * ( cols * 4 ) + ( i / 2 );
      uint8[index] = parseInt ( parts[ri].slice ( i, i + 2 ), 16 );
    }
  }

  return matrix;

};

/* EXPORT */

export {encode, decode};
