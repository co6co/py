import { withInstall } from 'co6co';

import taskTable, { TaskTableViewFeatures } from './src/taskTable';
export const TaskTableView = withInstall(taskTable, {
	features: TaskTableViewFeatures,
});
export * from './src';
