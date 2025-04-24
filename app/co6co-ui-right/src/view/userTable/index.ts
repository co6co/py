import { withInstall } from 'co6co';

import userTable from './src/userTable';

export const UserTableView = withInstall(userTable);
//export default UserTableView;

export * from './src';
