import { withInstall } from 'co6co';

import userTable, { ViewFeatures } from './src/userTable';
export const UserTableView = withInstall(userTable, { features: ViewFeatures });
//export default UserTableView;

export * from './src';
