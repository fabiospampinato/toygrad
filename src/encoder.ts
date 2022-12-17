
/* IMPORT */

import Buffer from '~/buffer';
import type {Precision} from '~/types';

/* MAIN */

const Encoder = {

  /* API */

  encode: ( buffer: Float32Array, precision: Precision ): string => {

    const uint8 = new Uint8Array ( buffer.buffer );

    if ( precision === 'f32' ) {

      let bytes1 = '';
      let bytes2 = '';
      let bytes3 = '';
      let bytes4 = '';

      for ( let i = 0, l = uint8.length; i < l; i += 4 ) {

        bytes1 += String.fromCharCode ( uint8[i + 0] );
        bytes2 += String.fromCharCode ( uint8[i + 1] );
        bytes3 += String.fromCharCode ( uint8[i + 2] );
        bytes4 += String.fromCharCode ( uint8[i + 3] );

      }

      return `4${btoa ( `${bytes1}${bytes2}${bytes3}${bytes4}` )}`;

    } else if ( precision === 'f16' ) {

      let bytes3 = '';
      let bytes4 = '';

      for ( let i = 0, l = uint8.length; i < l; i += 4 ) {

        bytes3 += String.fromCharCode ( uint8[i + 2] );
        bytes4 += String.fromCharCode ( uint8[i + 3] );

      }

      return `2${btoa ( `${bytes3}${bytes4}` )}`;

    } else {

      throw new Error ( 'Unsupported precision' );

    }

  },

  decode: ( encoded: string ): Float32Array => {

    const precision = Number ( encoded[0] );
    const bytes = atob ( encoded.slice ( 1 ) );
    const bytesChunkLength = bytes.length / precision;

    const buffer = new Buffer ( bytes.length / precision );
    const uint8 = new Uint8Array ( buffer.buffer );

    if ( precision === 4 ) {

      const bytes1 = bytes.slice ( bytesChunkLength * 0, bytesChunkLength * 1 );
      const bytes2 = bytes.slice ( bytesChunkLength * 1, bytesChunkLength * 2 );
      const bytes3 = bytes.slice ( bytesChunkLength * 2, bytesChunkLength * 3 );
      const bytes4 = bytes.slice ( bytesChunkLength * 3, bytesChunkLength * 4 );

      for ( let s = 0, i = 0, l = bytesChunkLength; i < l; s += 4, i += 1 ) {

        uint8[s + 0] = bytes1[i].charCodeAt ( 0 );
        uint8[s + 1] = bytes2[i].charCodeAt ( 0 );
        uint8[s + 2] = bytes3[i].charCodeAt ( 0 );
        uint8[s + 3] = bytes4[i].charCodeAt ( 0 );

      }

    } else if ( precision === 2 ) {

      const bytes3 = bytes.slice ( bytesChunkLength * 0, bytesChunkLength * 1 );
      const bytes4 = bytes.slice ( bytesChunkLength * 1, bytesChunkLength * 2 );

      for ( let s = 0, i = 0, l = bytesChunkLength; i < l; s += 4, i += 1 ) {

        uint8[s + 0] = 0;
        uint8[s + 1] = 0;
        uint8[s + 2] = bytes3[i].charCodeAt ( 0 );
        uint8[s + 3] = bytes4[i].charCodeAt ( 0 );

      }

    } else {

      throw new Error ( 'Unsupported precision' );

    }

    return buffer;

  }

};

/* EXPORT */

export default Encoder;
