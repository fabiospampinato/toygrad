
/* IMPORT */

import Abstract from '~/layers/abstract';
import type {AbstractHiddenOptions} from '~/types';

/* MAIN */

class AbstractHidden<T extends AbstractHiddenOptions = AbstractHiddenOptions> extends Abstract<T> {}

/* EXPORT */

export default AbstractHidden;
