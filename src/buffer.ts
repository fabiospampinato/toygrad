
/* IMPORT */

import type {Precision} from '~/types';

/* MAIN */

class Buffer extends Float32Array {

  /* STATIC API */

  static encode ( buffer: Float32Array, precision: Precision ): string {

    const uint8 = new Uint8Array ( buffer.buffer );

    let encoded = `${precision}|${buffer.length}|`;
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

  }

  static decode ( encoded: string ): Buffer {

    const parts =  encoded.split ( '|' );
    const precision = parts[0];
    const length = Number ( parts[1] );
    const bytes = atob ( parts[2] );

    const buffer = new Buffer ( length );
    const uint8 = new Uint8Array ( buffer.buffer );

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

    return buffer;

  }

}

/* EXPORT */

export default Buffer;
