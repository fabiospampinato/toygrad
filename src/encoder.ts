
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

    } else if ( precision === 'f8' ) {

      const max = Math.max ( ...Array.from ( buffer ).map ( Math.abs ) );
      const scale = Number ( ( 126 / max ).toFixed ( 1 ) );

      if ( max > 127 || scale < 1 ) throw new Error ( 'Unsupported encoding, max value out of range' );

      let bytes = '';

      for ( let i = 0, l = buffer.length; i < l; i += 1 ) {

        bytes += String.fromCharCode ( Math.trunc ( ( buffer[i] * scale ) + 127 ) );

      }

      return `1${scale}|${btoa ( bytes )}`;

    } else {

      throw new Error ( 'Unsupported precision' );

    }

  },

  decode: ( encoded: string ): Float32Array => {

    const precision = Number ( encoded[0] );
    const separatorIndex = encoded.indexOf ( '|' );
    const bytesIndex = ( separatorIndex >= 0 ) ? separatorIndex + 1 : 1;
    const bytes = atob ( encoded.slice ( bytesIndex ) );
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

    } else if ( precision === 1 ) {

      const scale = Number ( encoded.slice ( 1, separatorIndex ) );

      for ( let i = 0, l = bytesChunkLength; i < l; i += 1 ) {

        buffer[i] = ( bytes[i].charCodeAt ( 0 ) - 127 ) / scale;

      }

    } else {

      throw new Error ( 'Unsupported precision' );

    }

    return buffer;

  }

};

/* EXPORT */

export default Encoder;
