import { withInstall } from 'co6co';

import dynamicTable, { Features } from './src/dynamicTable';

export const DynamicTableView = withInstall(dynamicTable, {
	features: Features,
});
//export default UserTableView;

export * from './src';
