
/* IMPORT */

import Buffer from '~/buffer';
import type {Precision} from '~/types';

/* MAIN */

const Encoder = {

  /* API */

  encode: ( buffer: Float32Array, precision: Precision ): string => {

    const uint8 = new Uint8Array ( buffer.buffer );

    let bytes = '';

    if ( precision === 'f32' ) {

      for ( let i = 0, l = uint8.length; i < l; i += 1 ) {
        bytes += String.fromCharCode ( uint8[i] );
      }

      return `4${btoa ( bytes )}`;

    } else if ( precision === 'f16' ) {

      for ( let i = 0, l = uint8.length; i < l; i += 4 ) {
        bytes += String.fromCharCode ( uint8[i + 2] );
        bytes += String.fromCharCode ( uint8[i + 3] );
      }

      return `2${btoa ( bytes )}`;

    } else {

      throw new Error ( 'Unsupported precision' );

    }

  },

  decode: ( encoded: string ): Float32Array => {

    const precision = Number ( encoded[0] );
    const bytes = atob ( encoded.slice ( 1 ) );

    const buffer = new Buffer ( bytes.length / precision );
    const uint8 = new Uint8Array ( buffer.buffer );

    if ( precision === 4 ) {

      for ( let s = 0, i = 0, l = bytes.length; i < l; s += 4, i += 4 ) {
        uint8[s + 0] = bytes[i + 0].charCodeAt ( 0 );
        uint8[s + 1] = bytes[i + 1].charCodeAt ( 0 );
        uint8[s + 2] = bytes[i + 2].charCodeAt ( 0 );
        uint8[s + 3] = bytes[i + 3].charCodeAt ( 0 );
      }

    } else if ( precision === 2 ) {

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

};

/* EXPORT */

export default Encoder;
