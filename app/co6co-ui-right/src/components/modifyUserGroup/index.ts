import { withInstall } from 'co6co';

import modifyUserGroup from './src/modifyUserGroup';

export const ModifyUserGroup = withInstall(modifyUserGroup);
export default ModifyUserGroup;

export * from './src';
