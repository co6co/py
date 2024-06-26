import { withInstall } from '../../utils';

import hello from './src/hello';

export const Hello = withInstall(hello);
export default Hello;

export * from './src';
