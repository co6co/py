import { withInstall } from 'co6co';

import taskTable from './src/taskTable';

export const TaskTableView = withInstall(taskTable);
//export default UserTableView;

export * from './src';
