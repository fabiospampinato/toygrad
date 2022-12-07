
/* IMPORT */

import Abstract from '~/layers/abstract';
import type {AbstractOutputOptions} from '~/types';

/* MAIN */

class AbstractOutput<T extends AbstractOutputOptions = AbstractOutputOptions>  extends Abstract<T> {}

/* EXPORT */

export default AbstractOutput;
