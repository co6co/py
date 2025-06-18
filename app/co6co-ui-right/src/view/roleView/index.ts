import { withInstall } from 'co6co';

import roleView, { ViewFeatures } from './src/roleView';

export const RoleView = withInstall(roleView, { features: ViewFeatures });
//export default RoleView;

export * from './src';
