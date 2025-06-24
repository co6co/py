import { withInstall } from 'co6co';

import taskTable, { Features } from './src/taskTable';
export const TaskTableView = withInstall(taskTable, {
	features: Features,
});
export * from './src';
