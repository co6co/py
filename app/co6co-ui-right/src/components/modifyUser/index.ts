import { withInstall } from 'co6co';

import modifyUser from './src/modifyUser';

export const ModifyUser = withInstall(modifyUser);
export default ModifyUser;

export * from './src';
