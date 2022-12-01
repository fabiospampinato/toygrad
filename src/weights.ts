
/* IMPORT */

import type {Matrix, Precision} from './types';

/* MAIN */

//TODO: Make the encoding more efficient, we are throwing away some usable bits by encoding to hex characters
//TODO: Actually implement "float16" encoding

const encode = ( matrix: Matrix, precision: Precision ): string => {

  const TypedArray = ( precision === 'float64' ) ? Float64Array : Float32Array;

  return matrix.map ( row => {

    const {buffer} = new TypedArray ( row );
    const uint8 = new Uint8Array ( buffer );
    const hex = Array.from ( uint8 ).map ( byte => byte.toString ( 16 ).padStart ( 2, '0' ) ).join ( '' );

    return hex;

  }).join ( '|' );

};

const decode = ( encoded: string, precision: Precision ): Matrix => {

  const TypedArray = ( precision === 'float64' ) ? Float64Array : Float32Array;

  return encoded.split ( '|' ).map ( row => {

    const length = row.length / 2;
    const bytesPerElement = TypedArray.BYTES_PER_ELEMENT;
    const byteLength = length + ( ( length % bytesPerElement ) ? bytesPerElement - ( length % bytesPerElement ) : 0 );
    const buffer = new ArrayBuffer ( byteLength );
    const uint8 = new Uint8Array ( buffer );
    const float32 = new TypedArray ( buffer );

    for ( let i = 0, l = row.length; i < l; i += 2 ) {

      uint8[i / 2] = parseInt ( row.slice ( i, i + 2 ), 16 );

    }

    return Array.from ( float32 ).slice ( 0, length );

  });

};

/* EXPORT */

export {encode, decode};
