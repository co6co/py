import { withInstall } from 'co6co';

import dynamicTable, { ViewFeatures } from './src/dynamicTable';

export const DynamicTableView = withInstall(dynamicTable, {
	features: ViewFeatures,
});
//export default UserTableView;

export * from './src';
