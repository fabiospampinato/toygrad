
/* IMPORT */

import Abstract from '~/layers/abstract';
import type {AbstractInputOptions} from '~/types';

/* MAIN */

class AbstractInput<T extends AbstractInputOptions = AbstractInputOptions> extends Abstract<T> {}

/* EXPORT */

export default AbstractInput;
