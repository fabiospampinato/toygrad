
/* IMPORT */

import type {Matrix} from './types';

/* MAIN */

//TODO: Make the encoding more efficient, we are throwing away some usable bits by encoding to hex characters
//TODO: Make the precision configurable (Float64Array -> perfect, Float32Array -> great, Float16Array -> good)

const encode = ( matrix: Matrix ): string => {

  return matrix.map ( row => {

    const {buffer} = new Float32Array ( row );
    const uint8 = new Uint8Array ( buffer );
    const hex = Array.from ( uint8 ).map ( byte => byte.toString ( 16 ).padStart ( 2, '0' ) ).join ( '' );

    return hex;

  }).join ( '|' );

};

const decode = ( encoded: string ): Matrix => {

  return encoded.split ( '|' ).map ( row => {

    const length = row.length / 2;
    const byteLength = length + ( ( length % 4 ) ? 4 - ( length % 4 ) : 0 );
    const buffer = new ArrayBuffer ( byteLength );
    const uint8 = new Uint8Array ( buffer );
    const float32 = new Float32Array ( buffer );

    for ( let i = 0, l = row.length; i < l; i += 2 ) {

      uint8[i / 2] = parseInt ( row.slice ( i, i + 2 ), 16 );

    }

    return Array.from ( float32 ).slice ( 0, length );

  });

};

/* EXPORT */

export {encode, decode};
